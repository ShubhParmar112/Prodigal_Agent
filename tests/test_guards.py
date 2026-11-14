"""Checks on the two things the agent must never get wrong.

No external test runner needed:

    python tests/test_guards.py

1. Validation - the API stores anything, so these rules are the only defence.
2. Evidence - a value may only be saved if the candidate actually supplied it.
   This exists because the model, asked for a start date it was never given,
   invented 2026-08-15 and wrote "Candidate provided start date" as the reason.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import validators as V          # noqa: E402
from agent.tools import _candidate_said    # noqa: E402


class FakeState:
    """Just enough of ConversationState for the evidence check."""

    def __init__(self, *candidate_messages):
        self.transcript = [{"role": "candidate", "text": m} for m in candidate_messages]


# (label, state, field, value, expected)
EVIDENCE_CASES = [
    # the bug this guard was written for
    ("invented start date",     FakeState("yes", "2027", "3"),
     "available_from", "2026-08-15", False),
    ("invented email",          FakeState("yes speaking"),
     "email", "kabir.shah@gmail.com", False),
    ("invented graduation yr",  FakeState("yes", "sure"), "graduation_year", 2027, False),
    ("invented branch",         FakeState("yes"), "branch", "Computer Engineering", False),
    ("invented degree",         FakeState("yes ok"), "degree", "B.Tech", False),
    ("invented cgpa",           FakeState("pretty good"), "cgpa", 8.5, False),
    ("invented location",       FakeState("yeah fine"), "preferred_location", "Remote", False),
    ("candidate said nothing",  FakeState(), "cgpa", 8.0, False),

    # the agent is still allowed to do its job
    ("month and year",          FakeState("i can start from june 2027"),
     "available_from", "2027-06-01", True),
    ("full iso date",           FakeState("2027-06-01"), "available_from", "2027-06-01", True),
    ("graduation year",         FakeState("yes", "2027"), "graduation_year", 2027, True),
    ("duration",                FakeState("3"), "internship_duration_months", 3, True),
    ("email spelled out",       FakeState("sure its kabir.shah@gmail.com"),
     "email", "kabir.shah@gmail.com", True),
    ("abbreviated branch",      FakeState("comp engg"), "branch", "Computer Engineering", True),
    ("degree as 'be'",          FakeState("im doing be"), "degree", "B.E.", True),
    ("degree as 'btech'",       FakeState("btech"), "degree", "B.Tech", True),
    ("cgpa with decimal",       FakeState("8.4"), "cgpa", 8.4, True),
    ("whole-number cgpa",       FakeState("its 8"), "cgpa", 8.0, True),
    ("college partly named",    FakeState("vjti"), "college", "VJTI, Mumbai", True),
    ("bangalore -> Bengaluru",  FakeState("bangalore works"),
     "preferred_location", "Bengaluru", True),
]

# (label, field, value, expected_ok, expected_cleaned)
VALIDATION_CASES = [
    ("cgpa on a 100 scale",   "cgpa", 85, False, None),
    ("cgpa rounds to 2dp",    "cgpa", "8.456", True, 8.46),
    ("degree normalised",     "degree", "btech", True, "B.Tech"),
    ("degree not offered",    "degree", "PhD", False, None),
    ("location normalised",   "preferred_location", "bangalore", True, "Bengaluru"),
    ("year out of range",     "graduation_year", 2031, False, None),
    ("year in range",         "graduation_year", "2027", True, 2027),
    ("date out of window",    "available_from", "2029-06-01", False, None),
    ("date in window",        "available_from", "2027-06-01", True, "2027-06-01"),
    ("duration not offered",  "internship_duration_months", 4, False, None),
    ("email without tld",     "email", "kabir.shah@gmail", False, None),
    ("email spoken aloud",    "email", "shubh at gmail dot com", True, "shubh@gmail.com"),
    ("phone is read-only",    "phone", "+919999999999", False, None),
    ("unknown field",         "favourite_colour", "blue", False, None),
    ("github home page",      "github_url", "https://github.com", False, None),
    ("github profile",        "github_url", "github.com/shubh", True, "https://github.com/shubh"),
]


def main():
    failures = []

    print("evidence guard")
    for label, state, field, value, expected in EVIDENCE_CASES:
        got = _candidate_said(state, field, value)
        ok = got is expected
        failures.append(label) if not ok else None
        print("  {} {:<24} {!r}".format("pass" if ok else "FAIL", label, value))

    print("\nvalidation")
    for label, field, value, expect_ok, expect_clean in VALIDATION_CASES:
        got_ok, got_clean, _ = V.validate_field(field, value)
        ok = got_ok is expect_ok and (not expect_ok or got_clean == expect_clean)
        failures.append(label) if not ok else None
        print("  {} {:<24} {!r} -> {!r}".format(
            "pass" if ok else "FAIL", label, value, got_clean))

    print("\nrecord checks")
    dirty = {"full_name": "X Y", "email": "a@b", "cgpa": 85, "graduation_year": 2031}
    suspects = {s["field"] for s in V.suspect_fields(dirty)}
    for field in ("email", "cgpa", "graduation_year"):
        ok = field in suspects
        failures.append("suspect:" + field) if not ok else None
        print("  {} {} flagged as suspect".format("pass" if ok else "FAIL", field))

    print("\n{} failure(s)".format(len(failures)))
    if failures:
        print("failed:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
