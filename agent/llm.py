"""Groq wrapper.

Kept deliberately thin: one `complete()` that takes messages plus tool schemas
and returns a normalised reply, so swapping provider means editing this file
only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "openai/gpt-oss-120b"

# The model sometimes keeps writing past its own message and starts narrating
# the turn it has not seen yet. Cut it off at the first sign of that.
# Groq accepts at most four stop sequences, so tidy() in core.py mops up the rest.
STOP = ["User's response", "User response", "Candidate:", "(Waiting"]


@dataclass
class Reply:
    content: str = ""
    tool_calls: list = field(default_factory=list)  # [{id, name, args}]
    raw: object = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, api_key=None, model=None, temperature=0.6):
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMError("the groq package is not installed: pip install -r requirements.txt") from exc

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add a key "
                           "from https://console.groq.com")
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self._client = Groq(api_key=key)

    def complete(self, messages, tools=None) -> Reply:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            # Enough for a tool call carrying a notes string, but not so much that
            # the model keeps going and narrates the candidate's side of the chat.
            # Below ~400 it starts truncating set_outcome's arguments mid-JSON.
            "max_tokens": 500,
            "stop": STOP,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # groq raises several types; the caller only needs the text
            raise LLMError("the language model call failed: {}".format(exc)) from exc

        message = response.choices[0].message
        calls = []
        for call in (message.tool_calls or []):
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": call.id, "name": call.function.name, "args": args})

        return Reply(content=(message.content or "").strip(), tool_calls=calls, raw=message)
