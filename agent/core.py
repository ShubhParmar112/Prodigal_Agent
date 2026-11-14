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
from .llm import LLMError
from .state import ConversationState, now_iso
from .tools import SCHEMAS, ToolBox, describe

MAX_TOOL_ROUNDS = 6

# Narration the model sometimes writes instead of stopping at its own message.
_META = re.compile(
    r"^\(?\s*(?:user'?s?\s+response|candidate'?s?\s+response|awaiting|waiting for|"
    r"no response|not yet provided|no reply)", re.I)


def tidy(text):
    """Clean model output before the candidate ever sees it.

    gpt-oss will occasionally keep generating past the end of its own message
    and start narrating the turn it has not seen - rows of dots, a
    parenthetical aside, or 'User's response not yet provided.' repeated until
    it runs out of tokens. Stop sequences catch most of it; this catches the
    rest, and stops at the first repeated sentence.
    """
    if not text:
        return text

    text = re.sub(r"([.!?,])\1{2,}", r"\1", text.strip())
    text = re.sub(r"([?!])\.+", r"\1", text)  # "6?......" -> "6?"

    kept, seen = [], set()
    for sentence in re.findall(r"[^.!?]+[.!?]*", text):
        stripped = sentence.strip()
        if not stripped:
            continue
        if _META.match(stripped) or "not yet provided" in stripped.lower():
            break                       # everything after this is narration
        key = stripped.lower()
        if key in seen:
            break                       # it has started looping
        seen.add(key)
        kept.append(stripped)

    return re.sub(r"[ \t]{2,}", " ", " ".join(kept)).strip()


class Agent:
    def __init__(self, candidate_id, db, llm, emit=None):
        self.db = db
        self.llm = llm
        self.emit = emit or (lambda event: None)
        self.state = ConversationState(candidate_id=candidate_id)
        self.tools = ToolBox(db, self.state, self.emit)
        self.messages = [{"role": "system", "content": prompts.system_prompt()}]

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
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                reply = self.llm.complete(self.messages, SCHEMAS)
            except LLMError as exc:
                self.emit(Event(ERROR, str(exc)))
                return ("Sorry, I am having a technical problem at my end. "
                        "Can I message you again shortly?")

            self.messages.append(self._assistant_message(reply))

            if not reply.wants_tools:
                text = tidy(reply.content)
                if text and announce:
                    self.state.add_message("agent", text)
                    self.emit(Event(AGENT, text))
                return text

            for call in reply.tool_calls:
                self.emit(Event(TOOL, describe(call["name"], call["args"]),
                                data={"name": call["name"], "args": call["args"]}))
                result = self.tools.run(call["name"], call["args"])
                self.emit(Event(TOOL, "  -> {}".format(self._summarise(result)),
                                data={"result": result}, meta={"result": True}))
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

            if self.state.ended:
                # end_conversation was called; anything it said alongside still counts
                closing = tidy(reply.content)
                if closing and announce:
                    self.state.add_message("agent", closing)
                    self.emit(Event(AGENT, closing))
                return closing

        self.emit(Event(ERROR, "the model kept calling tools without replying; stopping the turn"))
        return "Sorry, something went wrong at my end. I will have someone follow up with you."

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
