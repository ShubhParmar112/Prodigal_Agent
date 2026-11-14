"""Events the agent emits so any UI can render the same run.

The core never prints. It emits these, and ui/cli.py and ui/web.py decide how
to show them. That is what lets one agent drive both front ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

AGENT = "agent"          # a message to show the candidate
CANDIDATE = "candidate"  # what the candidate typed
TOOL = "tool"            # the LLM called a tool
HTTP = "http"            # a request to the candidate database
SYSTEM = "system"        # status line for the operator, not the candidate
ERROR = "error"
OUTCOME = "outcome"      # final status written to the database


@dataclass
class Event:
    kind: str
    text: str
    data: Optional[dict] = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "data": self.data, "meta": self.meta}
