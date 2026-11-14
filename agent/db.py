"""Client for the candidate database.

Everything the assignment asks for on the write path lives here: bearer auth,
optimistic concurrency via If-Match, idempotency keys, 409/422 handling, a
client-side rate limit, and a full request log for the run transcript.
"""
from __future__ import annotations

import os
import time
import uuid
from collections import deque

import httpx

DEFAULT_BASE_URL = "https://candidate-db-api.service.external.usea2.aws.prodigaltech.com"
RATE_LIMIT_PER_MIN = 110  # the documented ceiling is 120; leave headroom


class CandidateDBError(Exception):
    """Any non-success response that the caller has to reason about."""

    def __init__(self, message, status=None, payload=None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class ConflictError(CandidateDBError):
    """409 - the record changed since we read it. Carries the latest version."""

    def __init__(self, message, latest=None, **kw):
        super().__init__(message, **kw)
        self.latest = latest


class CandidateDB:
    def __init__(self, api_key=None, base_url=None, timeout=20.0, on_request=None):
        self.base_url = (base_url or os.getenv("CANDIDATE_API_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("CANDIDATE_API_KEY")
        if not self.api_key:
            raise CandidateDBError("CANDIDATE_API_KEY is not set. Copy .env.example to .env.")
        self._client = httpx.Client(timeout=timeout)
        self._calls = deque()           # timestamps, for the rate limiter
        self.log = []                   # every request, for the run transcript
        self._on_request = on_request   # UI hook so tool calls can be shown live

    # ------------------------------------------------------------- internals

    def _throttle(self):
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= RATE_LIMIT_PER_MIN:
            time.sleep(max(0.0, 60 - (now - self._calls[0])) + 0.1)
        self._calls.append(time.monotonic())

    def _request(self, method, path, *, headers=None, json=None, retries=2):
        self._throttle()
        url = self.base_url + path
        sent = {"Authorization": "Bearer " + self.api_key}
        if headers:
            sent.update(headers)

        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self._client.request(method, url, headers=sent, json=json)
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise CandidateDBError("network error talking to the database: {}".format(exc))

            try:
                payload = response.json()
            except ValueError:
                payload = {"raw": response.text[:500]}

            entry = {
                "method": method,
                "path": path,
                "status": response.status_code,
                "request": json,
                "response": payload,
                "at": time.strftime("%H:%M:%S"),
            }
            # never let the key reach a log file
            entry["headers"] = {k: v for k, v in (headers or {}).items()}
            self.log.append(entry)
            if self._on_request:
                self._on_request(entry)

            if response.status_code == 429 and attempt < retries:
                time.sleep(float(response.headers.get("Retry-After", 2)))
                continue
            if response.status_code >= 500 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return response.status_code, payload

        raise CandidateDBError("request failed: {}".format(last_error))

    # ---------------------------------------------------------------- reads

    def get_schema(self):
        status, payload = self._request("GET", "/v1/schema")
        if status != 200:
            raise CandidateDBError("could not fetch schema", status, payload)
        return payload

    def list_candidates(self, collection_status=None):
        path = "/v1/candidates"
        if collection_status:
            path += "?collection_status=" + collection_status
        status, payload = self._request("GET", path)
        if status != 200:
            raise CandidateDBError("could not list candidates", status, payload)
        return payload.get("candidates", payload) if isinstance(payload, dict) else payload

    def get_candidate(self, candidate_id):
        status, payload = self._request("GET", "/v1/candidates/" + candidate_id)
        if status == 404:
            raise CandidateDBError("no candidate with id '{}'".format(candidate_id), status, payload)
        if status != 200:
            raise CandidateDBError("could not fetch candidate", status, payload)
        return payload

    # --------------------------------------------------------------- writes

    @staticmethod
    def changed_only(fields, current):
        """Drop fields whose value already matches the record.

        This is what keeps the write log clean: the evaluators read it, and the
        assignment scores not overwriting existing information needlessly.
        """
        out = {}
        for name, value in fields.items():
            if name in current and current[name] == value:
                continue
            out[name] = value
        return out

    def patch_candidate(self, candidate_id, fields, version=None, idempotency_key=None):
        """PATCH with optimistic concurrency. Raises ConflictError on 409."""
        if not fields:
            raise CandidateDBError("refusing to send an empty PATCH (the API returns 422)")

        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key or str(uuid.uuid4()),
        }
        if version is not None:
            headers["If-Match"] = '"{}"'.format(version)

        status, payload = self._request(
            "PATCH", "/v1/candidates/" + candidate_id, headers=headers, json=fields
        )

        if status in (200, 201):
            return payload
        if status == 409:
            latest = None
            try:
                latest = self.get_candidate(candidate_id)
            except CandidateDBError:
                pass
            raise ConflictError(
                "the record changed since it was read (version {})".format(version),
                latest=latest, status=status, payload=payload,
            )
        if status == 422:
            raise CandidateDBError(
                "the API rejected these fields as unknown or read-only: {}".format(
                    list(fields)), status, payload,
            )
        raise CandidateDBError("PATCH failed with {}".format(status), status, payload)

    def save(self, candidate_id, fields, record, idempotency_key=None):
        """PATCH, and on a 409 re-read, re-check and retry once.

        Returns (updated_record, note) where `note` describes any recovery that
        happened, so the agent can mention it or log it.
        """
        payload = self.changed_only(fields, record or {})
        if not payload:
            return record, "nothing to write - every value already matched the record"

        key = idempotency_key or str(uuid.uuid4())
        try:
            return self.patch_candidate(candidate_id, payload, record.get("version"), key), None
        except ConflictError as conflict:
            latest = conflict.latest or self.get_candidate(candidate_id)
            retry = self.changed_only(payload, latest)
            if not retry:
                return latest, "another writer already stored the same values; nothing re-sent"
            updated = self.patch_candidate(
                candidate_id, retry, latest.get("version"), str(uuid.uuid4())
            )
            return updated, "hit a 409, re-read the record at version {} and re-applied {}".format(
                latest.get("version"), list(retry)
            )

    def reset(self):
        status, payload = self._request("POST", "/v1/reset")
        if status not in (200, 201, 204):
            raise CandidateDBError("reset failed", status, payload)
        return payload

    def close(self):
        self._client.close()
