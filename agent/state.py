"""Where state lives.

One ConversationState per run. It holds the latest record (and its version),
what the agent has written, what the candidate refused, and the transcript that
is dumped to runs/ when the conversation ends.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .validators import missing_fields, suspect_fields

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def now_iso(offset: str = None) -> str:
    """ISO 8601 with a timezone offset, as last_contacted_at requires."""
    offset = offset or os.getenv("AGENT_TIMEZONE_OFFSET", "+05:30")
    sign = -1 if offset.startswith("-") else 1
    hours, _, minutes = offset.lstrip("+-").partition(":")
    tz = timezone(sign * timedelta(hours=int(hours), minutes=int(minutes or 0)))
    return datetime.now(tz).isoformat(timespec="seconds")


@dataclass
class ConversationState:
    candidate_id: str
    record: dict = field(default_factory=dict)
    started_at: str = field(default_factory=now_iso)

    saved: dict = field(default_factory=dict)      # field -> value actually written
    refused: list = field(default_factory=list)    # fields the candidate declined
    outcome: Optional[str] = None                  # collection_status written
    outcome_notes: Optional[str] = None
    needs_human_reason: Optional[str] = None
    ended: bool = False
    end_reason: Optional[str] = None

    transcript: list = field(default_factory=list)  # [{role, text}]
    tool_calls: list = field(default_factory=list)  # [{name, args, result}]

    # ------------------------------------------------------------- derived

    @property
    def version(self):
        return self.record.get("version")

    @property
    def full_name(self):
        return self.record.get("full_name") or "there"

    def missing(self):
        return missing_fields(self.record)

    def suspect(self):
        return suspect_fields(self.record)

    def still_needed(self):
        """What is left before this record could be called complete."""
        return self.missing() + [s["field"] for s in self.suspect()]

    def nothing_to_do(self) -> bool:
        return not self.missing() and not self.suspect()

    # -------------------------------------------------------------- record

    def remember(self, record: dict):
        if record:
            self.record = record

    def note_save(self, fields: dict):
        self.saved.update(fields)

    # ---------------------------------------------------------- transcript

    def add_message(self, role: str, text: str):
        self.transcript.append({"role": role, "text": text, "at": now_iso()})

    def add_tool_call(self, name: str, args: Any, result: Any):
        self.tool_calls.append({"name": name, "args": args, "result": result, "at": now_iso()})

    # -------------------------------------------------------------- output

    def dump(self, http_log=None, directory: str = None) -> str:
        """Write the transcript, tool calls and API responses to runs/."""
        directory = directory or RUNS_DIR
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.join(directory, "{}_{}".format(self.candidate_id, stamp))

        payload = {
            "candidate_id": self.candidate_id,
            "started_at": self.started_at,
            "ended_at": now_iso(),
            "outcome": self.outcome,
            "outcome_notes": self.outcome_notes,
            "end_reason": self.end_reason,
            "fields_written": self.saved,
            "fields_refused": self.refused,
            "record_at_start_missing": self.missing(),
            "transcript": self.transcript,
            "tool_calls": self.tool_calls,
            "api_calls": http_log or [],
            "final_record": self.record,
        }
        with open(base + ".json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        lines = ["Candidate {}  -  {}".format(self.candidate_id, self.record.get("full_name", "")),
                 "Outcome: {}".format(self.outcome), "-" * 60]
        for message in self.transcript:
            who = "AGENT" if message["role"] == "agent" else "CANDIDATE"
            lines.append("{:<10} {}".format(who + ":", message["text"]))
        lines += ["-" * 60, "Fields written: {}".format(self.saved or "none"),
                  "Notes: {}".format(self.outcome_notes or "none")]
        with open(base + ".txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

        return base
