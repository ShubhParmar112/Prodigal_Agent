"""The agent loop.

bootstrap -> open -> (reply | nudge)* -> close -> finalise

The loop is UI-agnostic: it emits Events and returns the text to show. Whether
the chat happens in a terminal or a browser is somebody else's problem.
"""
from __future__ import annotations

import json
import re

from . import prompts
from .events import AGENT, ERROR, OUTCOME, SYSTEM, TOOL, Event
from .llm import LLMError, ToolCallInvalid
from .state import ConversationState, now_iso
from .tools import SCHEMAS, ToolBox, describe

MAX_TOOL_ROUNDS = 6

# Narration the model writes when it carries on past the end of its own message.
_META = [re.compile(p, re.I) for p in (
    r"user'?s?\s+response", r"candidate'?s?\s+response", r"not yet provided",
    r"\(\s*waiting", r"awaiting\s+(?:your\s+)?(?:reply|response)",
    r"no\s+(?:reply|response)\s+yet",
)]

# A sentence, with its offset kept so the original text can be sliced intact.
_SENTENCE = re.compile(r"[^.!?]+[.!?]+|\S[^.!?]*$")

# Which field a question is about. Used to notice the agent asking the same
# thing twice - wording alone is useless, because "Which college are you
# attending?" and "What is the name of your institute?" share almost no words.
_TOPIC_WORDS = {
    "email": ("email", "e-mail", "mail address"),
    "college": ("college", "university", "institute", "studying at", "study at"),
    "degree": ("degree", "course", "programme", "program", "b.tech", "btech"),
    "branch": ("branch", "specialis", "specializ", "stream", "major", "discipline"),
    "graduation_year": ("graduat", "passing out", "pass out", "final year"),
    "cgpa": ("cgpa", "gpa", "grade point", "marks", "score"),
    "preferred_location": ("location", "city", "based", "mumbai", "bengaluru", "remote"),
    "available_from": ("start", "available", "join", "begin", "when could you"),
    "internship_duration_months": ("duration", "how long", "months", "how many month"),
}


def topic_of(text):
    """Best guess at which field a message is asking about, or None."""
    if not text:
        return None
    low = text.lower()
    best, score = None, 0
    for field, words in _TOPIC_WORDS.items():
        hits = sum(1 for w in words if w in low)
        if hits > score:
            best, score = field, hits
    return best


def tidy(text):
    """Clean model output before the candidate ever sees it.

    gpt-oss does not reliably stop at the end of one chat message. It stacks
    several questions together, invents the candidate's reply and answers it,
    or loops a sentence until it runs out of tokens. Everything here works by
    finding where to cut and slicing the original string - never by splitting
    and rejoining, which would mangle "e.g." and "B.Tech".
    """
    if not text:
        return text

    text = re.sub(r"([.!?,])\1{2,}", r"\1", text.strip())
    text = re.sub(r"([?!])\.+", r"\1", text)  # "6?......" -> "6?"

    cuts = []

    # 1. narration about a turn that has not happened
    for pattern in _META:
        found = pattern.search(text)
        if found:
            cuts.append(found.start())

    # 2. more than one question in a message - keep only the first
    marks = [m.start() for m in re.finditer(r"\?", text)]
    if len(marks) > 1:
        cuts.append(marks[0] + 1)

    # 3. a sentence repeating itself. Short fragments are skipped so that
    #    abbreviations like "B." are never mistaken for sentences.
    seen = set()
    for match in _SENTENCE.finditer(text):
        sentence = match.group().strip().lower()
        if len(sentence) < 20:
            continue
        if sentence in seen:
            cuts.append(match.start())
            break
        seen.add(sentence)

    if cuts:
        text = text[:min(cuts)]

    return re.sub(r"[ \t]{2,}", " ", text).strip()


class Agent:
    def __init__(self, candidate_id, db, llm, emit=None):
        self.db = db
        self.llm = llm
        self.emit = emit or (lambda event: None)
        self.state = ConversationState(candidate_id=candidate_id)
        self.tools = ToolBox(db, self.state, self.emit)
        self.messages = [{"role": "system", "content": prompts.system_prompt()}]
        self._last_topic = None      # what the previous message asked about
        self._saved_when_asked = 0   # how much was stored at that point

    # ------------------------------------------------------------ lifecycle

    def bootstrap(self):
        """Read the record and decide whether a conversation is needed at all.

        Returns (needs_chat, reason).
        """
        record = self.db.get_candidate(self.state.candidate_id)
        self.state.remember(record)

        missing, suspect = self.state.missing(), self.state.suspect()
        self.emit(Event(SYSTEM, "Loaded {} (version {})".format(
            record.get("full_name", "?"), record.get("version")),
            data={"missing": missing, "suspect": suspect}))

        if self.state.nothing_to_do():
            reason = ("{} has every mandatory field filled and nothing looks wrong, so there is "
                      "nothing to ask. No conversation started, no changes written.").format(
                          record.get("full_name", self.state.candidate_id))
            self.emit(Event(SYSTEM, reason))
            return False, reason

        parts = []
        if missing:
            parts.append("{} missing: {}".format(len(missing), ", ".join(missing)))
        if suspect:
            parts.append("{} suspect: {}".format(
                len(suspect), ", ".join(s["field"] for s in suspect)))
        return True, "; ".join(parts)

    def open(self):
        """Produce the agent's first message."""
        self.messages.append({"role": "user", "content": prompts.briefing(self.state)})
        return self._turn()

    def reply(self, text):
        """Feed the candidate's message in and get the agent's answer back."""
        self.state.add_message("candidate", text)
        self.messages.append({"role": "user", "content": text})
        return self._turn()

    def nudge(self):
        """The candidate has gone quiet; send one gentle follow-up."""
        self.messages.append({"role": "user", "content": prompts.IDLE_NUDGE})
        return self._turn()

    def close(self, reason="the chat was closed", gave_up=False):
        """Let the model record an outcome, then write everything to the database."""
        if not self.state.ended:
            self.state.end_reason = reason
            self.messages.append({
                "role": "user",
                "content": prompts.IDLE_GIVE_UP if gave_up else prompts.WRAP_UP,
            })
            try:
                self._turn(announce=False)
            except LLMError as exc:
                self.emit(Event(ERROR, "could not reach the model while closing: {}".format(exc)))
            self.state.ended = True
        return self.finalise()

    # ----------------------------------------------------------- the loop

    def _turn(self, announce=True):
        """One exchange: call the model, run any tools it asks for, repeat."""
        nudged = False
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                reply = self.llm.complete(self.messages, SCHEMAS)
            except ToolCallInvalid:
                # Usually save_fields called with an empty placeholder value.
                # Coach it and let the loop retry rather than breaking the chat.
                self.emit(Event(SYSTEM, "invalid tool call from the model; asking it to retry"))
                self.messages.append({"role": "user", "content": prompts.BAD_TOOL_CALL})
                continue
            except LLMError as exc:
                # The candidate is sitting there waiting, so they must be told
                # something. Returning silently is how the chat appears to hang.
                self.emit(Event(ERROR, str(exc)))
                return self._apologise(announce)

            if reply.note:
                self.emit(Event(SYSTEM, reply.note))

            self.messages.append(self._assistant_message(reply))

            if not reply.wants_tools:
                text = tidy(reply.content)

                # Asking about the same field again, with nothing collected in
                # between, is the "idk" loop: rephrasing the question does not
                # help, so make it change tack instead.
                topic = topic_of(text)
                if (announce and not nudged and topic and topic == self._last_topic
                        and len(self.state.saved) == self._saved_when_asked):
                    nudged = True
                    self.emit(Event(SYSTEM,
                                    "asked about {} again; nudging it to change tack".format(topic)))
                    self.messages.append({"role": "user",
                                          "content": prompts.repeated_question(topic)})
                    continue

                if text and announce:
                    self._last_topic = topic
                    self._saved_when_asked = len(self.state.saved)
                    self.state.add_message("agent", text)
                    self.emit(Event(AGENT, text))
                return text

            refused, did_nothing = {}, []
            for call in reply.tool_calls:
                self.emit(Event(TOOL, describe(call["name"], call["args"]),
                                data={"name": call["name"], "args": call["args"]}))
                result = self.tools.run(call["name"], call["args"])
                self.emit(Event(TOOL, "  -> {}".format(self._summarise(result)),
                                data={"result": result}, meta={"result": True}))
                if isinstance(result, dict) and result.get("rejected"):
                    refused.update(result["rejected"])
                elif isinstance(result, dict) and not result.get("ok") and result.get("error"):
                    did_nothing.append(result["error"])
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

            # A tool result alone is too easy to skim past - the model will thank
            # the candidate for a value that was never stored. Spell it out.
            if refused and not self.state.ended:
                self.messages.append({"role": "user",
                                      "content": prompts.rejection_followup(refused)})
            elif did_nothing and not self.state.ended:
                # e.g. save_fields with no values - otherwise it just tries again.
                self.messages.append({"role": "user",
                                      "content": prompts.tool_did_nothing(did_nothing)})

            if self.state.ended:
                # end_conversation was called; anything it said alongside still counts
                closing = tidy(reply.content)
                if closing and announce:
                    self.state.add_message("agent", closing)
                    self.emit(Event(AGENT, closing))
                return closing

        self.emit(Event(ERROR, "the model kept calling tools without replying; stopping the turn"))
        return self._apologise(announce)

    def _apologise(self, announce=True):
        """Say something to the candidate when the model could not be reached."""
        text = ("Sorry, I am having a technical problem at my end. "
                "Can I message you again shortly?")
        if announce:
            self.state.add_message("agent", text)
            self.emit(Event(AGENT, text))
        return text

    @staticmethod
    def _assistant_message(reply):
        message = {"role": "assistant", "content": reply.content or None}
        if reply.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"],
                                 "arguments": json.dumps(call["args"], default=str)},
                }
                for call in reply.tool_calls
            ]
        return message

    @staticmethod
    def _summarise(result):
        if not isinstance(result, dict):
            return str(result)[:120]
        if result.get("ok"):
            if result.get("saved"):
                return "saved " + ", ".join(result["saved"])
            if result.get("status"):
                return "outcome " + result["status"]
            return "ok"
        return "rejected: {}".format(result.get("rejected") or result.get("error"))[:160]

    # ------------------------------------------------------------ finalise

    def finalise(self):
        """Write collection_status, notes, attempt_count and last_contacted_at."""
        state = self.state

        if not state.outcome:
            state.outcome = self._infer_outcome()
            state.outcome_notes = state.outcome_notes or self._infer_notes()
            self.emit(Event(SYSTEM, "the model did not set an outcome; inferred '{}'".format(
                state.outcome)))

        tracking = {
            "collection_status": state.outcome,
            "last_contacted_at": now_iso(),
            # attempt_count counts outbound contact attempts, so it goes up once per
            # conversation that was actually opened - not only on no_answer.
            "attempt_count": (state.record.get("attempt_count") or 0) + 1,
        }
        if state.outcome_notes:
            tracking["notes"] = state.outcome_notes

        try:
            updated, note = self.db.save(state.candidate_id, tracking, state.record)
            state.remember(updated)
            if note:
                self.emit(Event(SYSTEM, note))
        except Exception as exc:  # the run must still produce a transcript
            self.emit(Event(ERROR, "could not write the outcome: {}".format(exc)))

        self.emit(Event(OUTCOME, state.outcome, data={
            "notes": state.outcome_notes,
            "written": state.saved,
            "still_needed": state.still_needed(),
        }))

        path = state.dump(http_log=self.db.log)
        self.emit(Event(SYSTEM, "transcript written to {}.json / .txt".format(path)))
        return state.outcome

    def _infer_outcome(self):
        if not self.state.saved and not self.state.transcript:
            return "no_answer"
        candidate_spoke = any(m["role"] == "candidate" for m in self.state.transcript)
        if not candidate_spoke:
            return "no_answer"
        if not self.state.still_needed():
            return "completed"
        return "in_progress"

    def _infer_notes(self):
        outstanding = self.state.still_needed()
        if not outstanding:
            return "All mandatory fields collected and validated."
        return "Chat ended before completion. Still needed: {}.".format(", ".join(outstanding))
