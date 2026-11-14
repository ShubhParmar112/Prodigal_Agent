# ACME candidate information collection agent

An autonomous chat agent that takes one candidate ID, works out what their internship
application is missing, collects it in a natural conversation, validates everything, writes it
back safely, and records how the conversation ended.

Two front ends drive the same agent: a terminal chat and a Streamlit web chat.

---

## Setup and run

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                               # macOS/Linux: cp .env.example .env
python run_cli.py --candidate-id c001
```

Fill `.env` with your two keys before the last step:

| Variable | What it is |
|---|---|
| `CANDIDATE_API_KEY` | The personal key for the candidate database |
| `GROQ_API_KEY` | A free key from [console.groq.com](https://console.groq.com) — this is the agent's brain |

The web UI instead of the terminal:

```bash
streamlit run app.py
```

Useful extras:

```bash
python run_cli.py --list     # your 8 candidates, with what each is missing
python run_cli.py --reset    # restore the workspace to its original data
python tests/test_guards.py  # 48 checks: validation, evidence guard, message cleanup
```

**Ending a chat.** Type `/quit` (or `/exit`) to leave, or `/silence` to simulate going quiet.
Closing the terminal works too. If nothing is typed for `IDLE_TIMEOUT_SECONDS` (default 120) the
agent nudges once, then closes the chat and records the outcome. In the web UI there is an
**End chat** button. However the chat ends, the outcome is always written before the process
exits.

---

## Architecture

```
          ui/cli.py (rich)          ui/web.py (streamlit)
                   \                  /
                    \                /          both only render Events
                     agent/core.py  <-- the loop
                            |
      +---------------------+---------------------+
      |                     |                     |
  agent/llm.py        agent/tools.py      agent/validators.py
   (Groq)             5 tools, guards      every schema rule
                             |
                       agent/db.py
            If-Match · Idempotency-Key · 409 recovery · rate limit
                             |
                   candidate-db-api (HTTP)
```

**The loop.** `bootstrap → open → (reply | nudge)* → close → finalise`. `bootstrap` reads the
record and decides whether a conversation is needed at all. Each turn calls the model; if it
returns tool calls they are executed, their results appended as `tool` messages, and the model is
called again — up to 6 rounds — until it produces a message for the candidate.

**The tools the LLM can call:**

| Tool | What it does |
|---|---|
| `save_fields` | Validates, strips unchanged values, writes the rest. Returns failures in plain language |
| `set_outcome` | Records `collection_status` + notes. Refuses `completed` while anything is outstanding |
| `refresh_candidate` | Re-reads the record, e.g. after a 409 |
| `flag_for_human` | Escalates: disputes, contradictions, an upset candidate |
| `end_conversation` | Closes the chat |

**Where state lives.** `agent/state.py` holds one `ConversationState` per run: the latest record
and its version, what was written, the transcript and every tool call. It is in memory only —
the database is the source of truth, and the run is dumped to `runs/` on exit. Nothing is cached
between runs, so a fresh run always reflects what is actually stored.

**Why the model is not trusted with validation.** The API stores whatever it is sent, so
`save_fields` validates first and hands failures *back to the model* as readable text —
`"CGPA 85 is above 10 … ask the candidate to confirm their CGPA out of 10 - do not convert it
yourself"`. The model then resolves it in conversation. That feedback loop, rather than a
fixed script, is what makes the agent recover.

---

## Decisions

**Working out what is needed.** Not just null checks. `suspect_fields()` re-validates every value
already on file, so `graduation_year: 2031` (c008) and `email: kabir.shah@gmail` (c005) are
treated as work to do even though the fields are populated. A record only counts as complete when
nothing is missing *and* nothing is invalid — which is why c003 and c006 start no conversation at
all and write nothing.

**Nothing is saved that the candidate did not say.** The single most important guard here, and it
exists because the model broke both ways during testing:

- Shown a broken email (`kabir.shah@gmail`), it "repaired" it to `kabir.shah@gmail.com` and saved
  it without ever asking.
- Asked to collect three fields from c002, it asked two questions, then invented
  `available_from: 2026-08-15` out of nothing — and wrote `"reason": "Candidate provided start
  date as 2026-08-15"` to justify it. The candidate had only ever said `yes`, `2027` and `3`.

The second is the dangerous one: a plausible value, a confident audit trail, and a record marked
`completed`. No prompt wording fixes it reliably, so `save_fields` now checks **every** value —
fills as well as overwrites — against what the candidate actually typed, and refuses anything it
cannot trace. The check is per field type, because the agent is *supposed* to convert formats:
"june 2027" → `2027-06-01` passes on the month, "comp engg" → `Computer Engineering` passes on a
token prefix, "bangalore" → `Bengaluru` passes on the alias. `tests/test_guards.py` pins down 20
cases of this, in both directions.

**Not overwriting needlessly.** `changed_only()` drops any field whose value already matches the
record, so a PATCH is only sent for things that genuinely changed. The write log stays clean.

**Concurrency and idempotency.** Every write sends `If-Match: "<version>"` and a fresh
`Idempotency-Key`. On a 409 the record is re-read, the change re-diffed against the newer version
and retried once; if it still fails the agent stops rather than clobbering. A client-side limiter
holds requests to 110/min, and 429/5xx are retried with backoff.

**Stops replying.** One nudge after the idle timeout, then the chat closes. If they gave nothing,
`no_answer`; if they gave something first, `in_progress` with what is still needed.

**Asks to continue later.** `in_progress`, with any time they suggested written into `notes`.
Whatever was collected is already saved, because `save_fields` is called as answers arrive rather
than batched at the end.

**Refusals.** The agent asks once why the field is needed, then accepts and moves on to the rest.
`refused` with the declined fields named in `notes`.

**Invalid or ambiguous answers.** Validation errors go back to the model, which re-asks in
conversation. Vague answers ("next summer") get one clarifying question. The agent converts
formats itself ("8.4 out of 10" → `8.4`) but never converts across scales — a CGPA of 85 is
queried, not divided by 10.

**Contradictions, disputes, upset candidates.** `flag_for_human`, no arguing, no guessing.

**`attempt_count`.** Incremented once per conversation that was actually opened, not only on
`no_answer`. It reads as "how many times have we tried to reach this person", which is what a
human picking the record up next would want. A record needing no conversation is not touched
at all, so it never increments.

**Outcome is always written.** If the model ends without calling `set_outcome`, `finalise()`
infers one from what happened. `completed` is additionally impossible to set while any field is
missing or invalid — the tool rejects it.

---

## Status

**What works** — verified against the live API, end to end:

- Full conversation on c001 (two missing fields) → `completed`, both fields written, tracking
  fields correct.
- c005: broken email + missing location. Agent queried the email, was blocked from inventing a
  fix, and completed once the candidate stated it.
- c002 (three missing fields): all three asked for and collected, "1 june 2027" stored as
  `2027-06-01`, `completed`. The fabricated-date bug above is gone — re-running the exact
  sequence that triggered it now produces a question instead of an invented value.
- c004: pulled `college` and `degree` out of a single "MPSTME, BTech IT" reply, confirmed
  `branch` separately, `completed`.
- `tests/test_guards.py`: 48 checks covering validation, the evidence guard and message
  cleanup, all passing.
- `runs/` holds five transcripts, each with the chat, the tool calls and every raw API call:
  a clean completion (c001), dirty data corrected by the candidate (c008), a candidate asking
  to continue later (c002), an invalid email rejected and re-asked until it was right (c007),
  and a candidate who could not answer, where the agent changed tack rather than repeating
  itself and ended `needs_human` (c004).
- c008: spotted `graduation_year: 2031` unprompted, refused to accept "my CGPA is 85" as a
  10-point value, and on `/quit` recorded `in_progress` with accurate notes.
- c003: no conversation started, nothing written.
- Both UIs drive the same core. The Streamlit sidebar shows each field as valid / missing /
  suspect and updates live as the conversation fills it.
- Every run writes `runs/<id>_<timestamp>.json` (transcript + tool calls + raw API calls) and a
  readable `.txt`.

**What does not work, or has not been proven:**

- **The 409 path has never actually fired.** The recovery code is written and reasoned through,
  but nothing has raced this workspace, so it is untested. The 422 path is likewise only
  defended against, not observed.
- **The evidence check is a heuristic.** It matches tokens from the value against what the
  candidate typed. A candidate who answers "yes, that's right" to a value the agent read out
  loud will be blocked, because the words never came from them. It fails in the safe direction —
  an extra question rather than an invented value — but it will sometimes be annoying.
- **One nudge only.** Real follow-up over hours or days is out of scope by design; the agent
  records state and stops.
- **Tests cover the guards only.** `validators.py` and the evidence check are pinned down. The
  agent loop, the database client and the UIs have no automated tests and were verified by
  running them.
- **Model dependence, and it is not subtle.** Built against `openai/gpt-oss-120b` on Groq. That
  model does not reliably stop at the end of one chat message. It stacks several questions into
  one bubble, invents the candidate's reply and answers it ("Sorry, I missed that — could you
  tell me…" before they have said anything), and loops a sentence until it runs out of tokens.
  No amount of prompt wording stopped it, so `tidy()` in `core.py` enforces it structurally:
  one question per message, narration cut, loops truncated. Four stop sequences and a
  `max_tokens` cap help. All of it is a workaround for a weak model, not a fix — a stronger
  model would need none of it, and `tidy()` is the first thing to delete when one is available.
- The agent writes `notes` as a single string, overwriting any previous note rather than
  appending to a history.

**What I would do next, in order:**

1. A fake DB server that returns 409s and 422s on demand, so the recovery paths are actually
   exercised.
2. Strengthen the evidence check with an explicit confirmation path: when the agent reads a value
   back and the candidate simply agrees, record *that exchange* as the evidence, so the one
   legitimate case the heuristic blocks today is handled properly.
3. Append to `notes` with a timestamp instead of replacing, so repeat conversations build a
   history.
4. Try a stronger model and delete `tidy()` if it turns out to be unnecessary.
