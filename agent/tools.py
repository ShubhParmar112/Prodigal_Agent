"""The tools the LLM can call, and the code behind them.

Schemas are explicit rather than a free-form object: naming every field and its
enum in the schema is what makes tool calls reliable on a mid-size model.

Every write goes through `save_fields`, which validates first and hands failures
back to the model in plain language. That feedback loop - not a script - is how
the agent recovers from bad answers.
"""
from __future__ import annotations

import json
import math
import re

from . import validators as V
from .db import CandidateDBError, ConflictError
from .state import now_iso

SAVE_FIELDS = {
    "type": "function",
    "function": {
        "name": "save_fields",
        "description": (
            "Save one or more confirmed answers to the candidate's record. Call this as soon as "
            "the candidate gives you a clear answer; do not batch everything to the end. Only "
            "include fields the candidate actually told you in this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Valid email address."},
                "college": {"type": "string"},
                "degree": {"type": "string", "enum": V.DEGREES},
                "branch": {"type": "string",
                           "description": "Specialisation, e.g. Computer Engineering."},
                "graduation_year": {"type": "integer", "description": "2026, 2027 or 2028."},
                "cgpa": {"type": "number", "description": "On a 10-point scale, e.g. 8.4."},
                "preferred_location": {"type": "string", "enum": V.LOCATIONS},
                "available_from": {"type": "string",
                                   "description": "YYYY-MM-DD, between 2026-01-01 and 2027-12-31."},
                "internship_duration_months": {"type": "integer", "enum": V.DURATIONS},
                "linkedin_url": {"type": "string"},
                "github_url": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}},
                "full_name": {"type": "string",
                              "description": "Only if the candidate corrects their own name."},
                "reason": {
                    "type": "string",
                    "description": "One line on what the candidate said that led to this save.",
                },
            },
            "required": ["reason"],
        },
    },
}

SET_OUTCOME = {
    "type": "function",
    "function": {
        "name": "set_outcome",
        "description": (
            "Record how this conversation ended. Call this once, near the end. "
            "'completed' only when nothing is outstanding."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["in_progress", "completed", "no_answer", "refused", "needs_human"],
                },
                "notes": {
                    "type": "string",
                    "description": (
                        "Context for the human or the next conversation: what is still needed, "
                        "which fields were refused, any time the candidate suggested calling back."
                    ),
                },
            },
            "required": ["status", "notes"],
        },
    },
}

REFRESH = {
    "type": "function",
    "function": {
        "name": "refresh_candidate",
        "description": (
            "Re-read the candidate's record from the database. Use this if a save failed because "
            "the record changed, or if you need to check what is currently stored."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

FLAG_HUMAN = {
    "type": "function",
    "function": {
        "name": "flag_for_human",
        "description": (
            "Hand this candidate to a human. Use when the record is disputed or contradictory, "
            "the candidate is upset, they say it is not their application, or you simply cannot "
            "resolve the situation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "What happened, for the human."},
            },
            "required": ["reason"],
        },
    },
}

END = {
    "type": "function",
    "function": {
        "name": "end_conversation",
        "description": (
            "Close the chat. Call this after set_outcome, once there is nothing left to say."
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
}

SCHEMAS = [SAVE_FIELDS, SET_OUTCOME, REFRESH, FLAG_HUMAN, END]


_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]


def _compact(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _enum_spellings(field, value):
    """Every way a candidate might have said this enum value."""
    aliases = V._DEGREE_ALIASES if field == "degree" else V._LOCATION_ALIASES
    return [value] + [alias for alias, canon in aliases.items() if canon == value]


def _evidence_ok(field, value, spoken):
    """Is there a trace of `value` in what the candidate actually typed?

    Checked per field type, because the agent is supposed to convert formats
    ("next June" -> 2027-06-01, "comp engg" -> Computer Engineering) and a
    naive string match would reject its own correct work.
    """
    if field == "available_from":
        text = str(value)
        if text[:4] in spoken:                       # the year
            return True
        month = _MONTHS[int(text[5:7]) - 1]
        return month in spoken or month[:3] in spoken

    if field in ("graduation_year", "internship_duration_months", "cgpa"):
        digits = re.sub(r"\D", "", str(value))
        if digits in spoken:
            return True
        # 8.0 is usually said as "8"
        return bool(digits.endswith("0") and digits[:-1] and digits[:-1] in spoken)

    if field in ("degree", "preferred_location"):
        return any(_compact(spelling) in spoken for spelling in _enum_spellings(field, value))

    if isinstance(value, list):
        value = " ".join(str(v) for v in value)

    tokens = [t for t in re.findall(r"[a-z0-9]+", str(value).lower()) if len(t) >= 2]
    if not tokens:
        return _compact(value) in spoken

    # A token counts if it appears, or if its first four characters do - that
    # allows "comp engg" to support "Computer Engineering".
    hits = sum(1 for t in tokens if t in spoken or (len(t) >= 6 and t[:4] in spoken))
    return hits >= max(1, math.ceil(0.5 * len(tokens)))


def _candidate_said(state, field, value):
    """Did the candidate actually provide this value?

    Applied to every save, not only overwrites. Asked for a start date it was
    never given, the model will happily invent one and write a confident
    `reason` to match, so the only reliable check is against the candidate's
    own words.
    """
    spoken = _compact(" ".join(m["text"] for m in state.transcript
                               if m["role"] == "candidate"))
    if not spoken:
        return False
    return _evidence_ok(field, value, spoken)


class ToolBox:
    """Executes tool calls against the database and the conversation state."""

    def __init__(self, db, state, emit):
        self.db = db
        self.state = state
        self.emit = emit

    # ------------------------------------------------------------ dispatch

    def run(self, name, args):
        handler = getattr(self, "_" + name, None)
        if handler is None:
            return {"ok": False, "error": "no tool called '{}'".format(name)}
        try:
            result = handler(args or {})
        except CandidateDBError as exc:
            result = {"ok": False, "error": str(exc),
                      "advice": "Tell the candidate you are having a technical problem."}
        self.state.add_tool_call(name, args, result)
        return result

    # --------------------------------------------------------------- tools

    def _save_fields(self, args):
        args = dict(args)
        reason = args.pop("reason", "")
        fields = {k: v for k, v in args.items() if v not in (None, "", [])}

        if not fields:
            return {"ok": False, "error": "No field values were given, so nothing was saved."}

        cleaned, errors = V.validate_many(fields)

        # Every value must be traceable to something the candidate actually said.
        for name in list(cleaned):
            existing = self.state.record.get(name)
            if existing == cleaned[name]:
                continue
            if _candidate_said(self.state, name, cleaned[name]):
                continue
            cleaned.pop(name)
            if existing in (None, "", []):
                errors[name] = (
                    "Refused: the candidate has not told you a value for {}. You cannot infer it, "
                    "assume a sensible default, or carry it over from another field. Ask them for "
                    "it and save only what they answer."
                ).format(name)
            else:
                errors[name] = (
                    "Refused: this would change {} from '{}' to '{}', but the candidate has not "
                    "told you that value in this chat. Do not repair or guess at existing data. "
                    "Ask them to state it themselves, then save what they say."
                ).format(name, existing, fields.get(name))

        # Nothing survived validation - send the reasons back so the model re-asks.
        if not cleaned:
            return {"ok": False, "rejected": errors,
                    "advice": "Go back to the candidate and resolve these in conversation."}

        # Do not re-send values the record already has.
        payload = self.db.changed_only(cleaned, self.state.record)
        unchanged = [k for k in cleaned if k not in payload]

        written = {}
        note = None
        if payload:
            try:
                updated, note = self.db.save(
                    self.state.candidate_id, payload, self.state.record)
            except ConflictError as exc:
                return {"ok": False, "error": str(exc),
                        "advice": "Call refresh_candidate, then try saving again."}
            self.state.remember(updated)
            self.state.note_save(payload)
            written = payload

        result = {"ok": True, "saved": written, "reason": reason}
        if unchanged:
            result["already_matched"] = unchanged
        if errors:
            result["rejected"] = errors
            result["advice"] = "The rest was saved. Resolve the rejected ones with the candidate."
        if note:
            result["recovery"] = note
        result["still_needed"] = self.state.still_needed()
        return result

    def _set_outcome(self, args):
        status = args.get("status", "")
        ok, cleaned_status, err = V.v_collection_status(status)
        if not ok:
            return {"ok": False, "error": err}

        notes = args.get("notes") or ""
        ok, cleaned_notes, err = V.v_notes(notes)
        if not ok:
            return {"ok": False, "error": err}

        # Guard against an over-eager 'completed'.
        outstanding = self.state.still_needed()
        if cleaned_status == "completed" and outstanding:
            return {
                "ok": False,
                "error": "Cannot mark this completed - still outstanding: {}.".format(
                    ", ".join(outstanding)),
                "advice": (
                    "Either collect those, or use 'in_progress' or 'refused' with notes "
                    "explaining why they are not filled."
                ),
            }

        self.state.outcome = cleaned_status
        self.state.outcome_notes = cleaned_notes
        return {"ok": True, "status": cleaned_status,
                "note": "Outcome recorded. It is written to the database when the chat closes."}

    def _refresh_candidate(self, _args):
        record = self.db.get_candidate(self.state.candidate_id)
        self.state.remember(record)
        return {
            "ok": True,
            "version": record.get("version"),
            "missing": self.state.missing(),
            "suspect": [s["field"] for s in self.state.suspect()],
        }

    def _flag_for_human(self, args):
        reason = args.get("reason") or "unspecified"
        self.state.needs_human_reason = reason
        self.state.outcome = "needs_human"
        self.state.outcome_notes = "Escalated by the agent: " + reason
        return {"ok": True, "note": "Flagged for a human. Close the chat politely now."}

    def _end_conversation(self, args):
        self.state.ended = True
        self.state.end_reason = args.get("reason") or "agent ended the chat"
        return {"ok": True}


def describe(name, args):
    """One-line human description of a tool call, for the UI trace."""
    args = args or {}
    if name == "save_fields":
        fields = {k: v for k, v in args.items() if k != "reason" and v not in (None, "", [])}
        return "save_fields {}".format(json.dumps(fields, ensure_ascii=False))
    if name == "set_outcome":
        return "set_outcome -> {}".format(args.get("status"))
    if name == "flag_for_human":
        return "flag_for_human: {}".format(args.get("reason", "")[:60])
    if name == "end_conversation":
        return "end_conversation"
    return name
