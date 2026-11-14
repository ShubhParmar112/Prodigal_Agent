"""Groq wrapper.

Kept deliberately thin: one `complete()` that takes messages plus tool schemas
and returns a normalised reply, so swapping provider means editing this file
only.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

DEFAULT_MODEL = "openai/gpt-oss-120b"
# Used automatically when the daily quota on the main model runs out. It has its
# own quota, so a live demo does not die just because testing burned the budget.
FALLBACK_MODEL = "openai/gpt-oss-20b"

# The model sometimes keeps writing past its own message and starts narrating
# the turn it has not seen yet. Cut it off at the first sign of that.
# Groq accepts at most four stop sequences, so tidy() in core.py mops up the rest.
STOP = ["User's response", "User response", "Candidate:", "(Waiting"]


@dataclass
class Reply:
    content: str = ""
    tool_calls: list = field(default_factory=list)  # [{id, name, args}]
    raw: object = None
    note: str = ""   # e.g. "fell back to the smaller model"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(Exception):
    pass


class ToolCallInvalid(LLMError):
    """The model sent tool arguments the provider rejected against our schema.

    Recoverable: the model is told what it got wrong and asked to try again,
    rather than the candidate being told the system is broken.
    """


class RateLimited(LLMError):
    """Groq's free tier is capped per minute AND per day."""

    def __init__(self, message, retry_after=None, daily=False):
        super().__init__(message)
        self.retry_after = retry_after
        self.daily = daily


def _as_rate_limit(exc):
    """Turn a provider exception into RateLimited, or None if it is something else."""
    text = str(exc)
    if "rate_limit" not in text and "429" not in text:
        return None
    wait = re.search(r"try again in ([0-9hms.]+)", text)
    daily = "per day" in text or "TPD" in text
    if daily:
        message = ("the daily token quota for this Groq model is used up"
                   + (" (resets in {})".format(wait.group(1)) if wait else ""))
    else:
        message = ("the model is rate limited"
                   + (", retry in {}".format(wait.group(1)) if wait else ""))
    return RateLimited(message, retry_after=wait.group(1) if wait else None, daily=daily)


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
        self.fallback = os.getenv("GROQ_FALLBACK_MODEL", FALLBACK_MODEL)
        self.temperature = temperature
        self._client = Groq(api_key=key)

    def complete(self, messages, tools=None) -> Reply:
        """Call the model, falling back to the smaller one if the quota is gone."""
        try:
            return self._call(self.model, messages, tools)
        except RateLimited as limited:
            if not (limited.daily and self.fallback and self.fallback != self.model):
                raise
            reply = self._call(self.fallback, messages, tools)
            reply.note = "daily quota for {} is used up; switched to {}".format(
                self.model, self.fallback)
            self.model = self.fallback   # stay on it for the rest of the chat
            return reply

    def _call(self, model, messages, tools=None) -> Reply:
        kwargs = {
            "model": model,
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
            limited = _as_rate_limit(exc)
            if limited:
                raise limited from exc
            text = str(exc)
            if "tool_use_failed" in text or "did not match schema" in text:
                raise ToolCallInvalid(text) from exc
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
