"""Field validation and normalisation.

The candidate API stores whatever it is sent, so every rule in the assignment
schema is enforced here instead. Each validator returns (ok, cleaned, error);
`error` is written for an LLM to read, because it is fed straight back to the
model as a tool result so it can re-ask the candidate naturally.
"""
from __future__ import annotations

import re
from datetime import date, datetime

# ---------------------------------------------------------------- vocabulary

DEGREES = ["B.Tech", "B.E.", "B.Sc", "BCA", "MBA Tech", "MCA", "M.Tech", "Other"]
LOCATIONS = ["Mumbai", "Bengaluru", "Remote", "No preference"]
DURATIONS = [2, 3, 6]
STATUSES = ["pending", "in_progress", "completed", "no_answer", "refused", "needs_human"]

MANDATORY = [
    "full_name", "phone", "email", "college", "degree", "branch",
    "graduation_year", "cgpa", "preferred_location", "available_from",
    "internship_duration_months",
]
# phone is OTP-verified and read-only, so the agent is never asked to fill it
COLLECTABLE = [f for f in MANDATORY if f != "phone"]
OPTIONAL = ["linkedin_url", "github_url", "skills"]
TRACKING = ["collection_status", "attempt_count", "last_contacted_at", "notes"]
READ_ONLY = ["id", "phone", "version", "updated_at"]
WRITABLE = COLLECTABLE + OPTIONAL + TRACKING

GRAD_YEARS = (2026, 2028)
AVAIL_WINDOW = (date(2026, 1, 1), date(2027, 12, 31))

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

_DEGREE_ALIASES = {
    "btech": "B.Tech", "b tech": "B.Tech", "bachelor of technology": "B.Tech",
    "be": "B.E.", "b e": "B.E.", "bachelor of engineering": "B.E.",
    "bsc": "B.Sc", "b sc": "B.Sc", "bachelor of science": "B.Sc",
    "bca": "BCA",
    "mba tech": "MBA Tech", "mbatech": "MBA Tech", "mba-tech": "MBA Tech",
    "mca": "MCA",
    "mtech": "M.Tech", "m tech": "M.Tech", "master of technology": "M.Tech",
    "other": "Other",
}
_LOCATION_ALIASES = {
    "mumbai": "Mumbai", "bombay": "Mumbai",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru",
    "remote": "Remote", "wfh": "Remote", "work from home": "Remote",
    "no preference": "No preference", "any": "No preference",
    "anywhere": "No preference", "none": "No preference", "no pref": "No preference",
}


def _norm(value) -> str:
    """Lowercase, collapse whitespace, drop dots: 'B.Tech' and 'b tech' both -> 'b tech'."""
    return re.sub(r"[\s.]+", " ", str(value).strip().lower()).strip()


# ---------------------------------------------------------------- validators

def _text(value, lo, hi, label):
    if value is None:
        return False, None, label + " is empty."
    text = " ".join(str(value).split())
    if not (lo <= len(text) <= hi):
        return False, None, "{} must be {}-{} characters; got {}.".format(label, lo, hi, len(text))
    return True, text, None


def v_full_name(value):
    return _text(value, 2, 100, "full_name")


def v_college(value):
    return _text(value, 2, 150, "college")


def v_branch(value):
    return _text(value, 2, 100, "branch")


def v_email(value):
    if value is None:
        return False, None, "email is empty."
    text = str(value).strip().replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
    if not _EMAIL_RE.match(text):
        return False, None, (
            "'{}' is not a valid email address. Ask the candidate to repeat it, "
            "including the part after the @.".format(value)
        )
    return True, text, None


def _enum(value, aliases, canon_list, label, hint):
    if value is None:
        return False, None, label + " is empty."
    key = _norm(value)
    hit = aliases.get(key)
    if hit is None:
        for canon in canon_list:
            if _norm(canon) == key:
                hit = canon
                break
    if hit is None:
        return False, None, "'{}' is not an allowed {}. Allowed: {}. {}".format(
            value, label, ", ".join(canon_list), hint
        )
    return True, hit, None


def v_degree(value):
    return _enum(
        value, _DEGREE_ALIASES, DEGREES, "degree",
        "If the candidate's degree genuinely is not in the list, use 'Other'.",
    )


def v_preferred_location(value):
    return _enum(
        value, _LOCATION_ALIASES, LOCATIONS, "preferred_location",
        "Ask the candidate to pick one of those four.",
    )


def v_graduation_year(value):
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return False, None, "'{}' is not a year. Expected a number like 2027.".format(value)
    lo, hi = GRAD_YEARS
    if not (lo <= year <= hi):
        return False, None, (
            "graduation_year {} is outside the accepted range {}-{}. This programme only takes "
            "candidates graduating in those years, so confirm the year with the candidate "
            "before deciding what to do.".format(year, lo, hi)
        )
    return True, year, None


def v_cgpa(value):
    try:
        num = float(str(value).strip().split("/")[0])
    except (TypeError, ValueError):
        return False, None, "'{}' is not a number. CGPA must be numeric, e.g. 8.4.".format(value)
    if 10 < num <= 100:
        return False, None, (
            "CGPA {} is above 10. The database uses a 10-point scale, so this looks like a "
            "percentage or a different scale. Ask the candidate to confirm their CGPA out of 10 "
            "- do not convert it yourself.".format(num)
        )
    if not (0.0 <= num <= 10.0):
        return False, None, "CGPA {} is outside the valid range 0.0-10.0.".format(num)
    return True, round(num, 2), None


def v_internship_duration_months(value):
    try:
        num = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return False, None, "'{}' is not a number of months. Allowed: 2, 3 or 6.".format(value)
    if num not in DURATIONS:
        return False, None, (
            "{} months is not offered. Only 2, 3 or 6 months are available - ask the candidate "
            "which of those suits them.".format(num)
        )
    return True, num, None


def v_available_from(value):
    if value is None:
        return False, None, "available_from is empty."
    text = str(value).strip()
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return False, None, (
            "'{}' is not in YYYY-MM-DD format. Convert the candidate's answer to a real calendar "
            "date first (e.g. 'early June next year' -> 2027-06-01). If their answer is too vague "
            "to pin down, ask for a specific date.".format(value)
        )
    lo, hi = AVAIL_WINDOW
    if not (lo <= parsed <= hi):
        return False, None, (
            "available_from {} is outside the accepted window {} to {}. Ask the candidate for a "
            "start date inside that window.".format(text, lo, hi)
        )
    return True, text, None


def _url(value, host, label, hint):
    if value is None:
        return False, None, label + " is empty."
    text = str(value).strip()
    if not text.startswith(("http://", "https://")):
        text = "https://" + text
    if host not in text.lower():
        return False, None, "'{}' is not a {} URL. {}".format(value, host, hint)
    return True, text, None


def v_linkedin_url(value):
    return _url(value, "linkedin.com", "linkedin_url",
                "Expected something like https://www.linkedin.com/in/their-handle.")


def v_github_url(value):
    ok, cleaned, err = _url(value, "github.com", "github_url",
                            "Expected a profile like https://github.com/their-handle.")
    if not ok:
        return ok, cleaned, err
    if not cleaned.split("github.com", 1)[1].strip("/"):
        return False, None, "That is the GitHub home page, not a profile URL."
    return True, cleaned, None


def v_skills(value):
    if value is None:
        return False, None, "skills is empty."
    if isinstance(value, str):
        items = [p.strip() for p in re.split(r"[,;/]| and ", value)]
    else:
        items = [str(p).strip() for p in value]
    items = [i for i in items if i]
    if not items:
        return False, None, "No skills were recognised in that answer."
    if len(items) > 30:
        return False, None, "At most 30 skills are allowed; got {}.".format(len(items))
    too_long = [i for i in items if len(i) > 50]
    if too_long:
        return False, None, "These look like sentences rather than skills: {}.".format(too_long)
    return True, items, None


def v_collection_status(value):
    key = _norm(value).replace(" ", "_")
    if key not in STATUSES:
        return False, None, "'{}' is not a status. Allowed: {}.".format(value, ", ".join(STATUSES))
    return True, key, None


def v_attempt_count(value):
    try:
        num = int(value)
    except (TypeError, ValueError):
        return False, None, "'{}' is not an integer.".format(value)
    if not (0 <= num <= 50):
        return False, None, "attempt_count {} is outside 0-50.".format(num)
    return True, num, None


def v_notes(value):
    if value is None:
        return True, None, None
    text = str(value).strip()
    if len(text) > 4000:
        return False, None, "notes is {} characters; the maximum is 4000.".format(len(text))
    return True, text, None


def v_last_contacted_at(value):
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return False, None, "'{}' is not ISO 8601, e.g. 2026-11-20T15:30:00+05:30.".format(value)
    if parsed.tzinfo is None:
        return False, None, "last_contacted_at needs a timezone offset, e.g. +05:30."
    return True, text, None


VALIDATORS = {
    "full_name": v_full_name,
    "email": v_email,
    "college": v_college,
    "degree": v_degree,
    "branch": v_branch,
    "graduation_year": v_graduation_year,
    "cgpa": v_cgpa,
    "preferred_location": v_preferred_location,
    "available_from": v_available_from,
    "internship_duration_months": v_internship_duration_months,
    "linkedin_url": v_linkedin_url,
    "github_url": v_github_url,
    "skills": v_skills,
    "collection_status": v_collection_status,
    "attempt_count": v_attempt_count,
    "last_contacted_at": v_last_contacted_at,
    "notes": v_notes,
}


def validate_field(name, value):
    """-> (ok, cleaned_value, error_message)."""
    if name in READ_ONLY:
        return False, None, (
            "'{}' is read-only and can never be written. Phone numbers are OTP-verified; if the "
            "candidate says theirs is wrong, that is a case for a human.".format(name)
        )
    validator = VALIDATORS.get(name)
    if validator is None:
        return False, None, (
            "'{}' is not a field in this database. Writable fields: {}.".format(
                name, ", ".join(WRITABLE))
        )
    return validator(value)


def validate_many(fields):
    """-> (cleaned: dict, errors: dict[field, message])."""
    cleaned, errors = {}, {}
    for name, value in fields.items():
        ok, out, err = validate_field(name, value)
        if ok:
            cleaned[name] = out
        else:
            errors[name] = err
    return cleaned, errors


# ------------------------------------------------------- record-level checks

def _is_empty(value):
    return value is None or (isinstance(value, str) and not value.strip())


def missing_fields(record):
    """Mandatory fields that are empty. phone is excluded - it is always present."""
    return [f for f in COLLECTABLE if _is_empty(record.get(f))]


def suspect_fields(record):
    """Fields that are filled but violate the schema.

    The assignment warns the existing data is not clean, so a record with no
    empty fields is not necessarily a usable one.
    """
    out = []
    for field in COLLECTABLE + OPTIONAL:
        value = record.get(field)
        if _is_empty(value):
            continue
        ok, cleaned, err = validate_field(field, value)
        if not ok:
            out.append({"field": field, "value": value, "problem": err})
        elif cleaned != value:
            out.append({
                "field": field,
                "value": value,
                "problem": "stored as '{}' but the canonical form is '{}'".format(value, cleaned),
                "autofix": cleaned,
            })
    return out


def is_complete(record):
    return not missing_fields(record) and not suspect_fields(record)
