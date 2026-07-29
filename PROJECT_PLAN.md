# PROJECT_PLAN.md — TDS P1 Task 2, phased to the deadline

37.5 marks. 0.1 auto-awarded for registering identifiers; the rest is graded
from the live bot and the repo after the deadline. Phases are ordered so that
**you own marks early** and every later phase only adds to them.

---

## Phase 0 — Claim the free marks (30 minutes)

| # | Task | Done when |
|---|---|---|
| 0.1 | `@BotFather` → `/newbot`. Username **must end in `bot`**. Save the token. | You have a token |
| 0.2 | Create a **public** GitHub repo | URL loads logged out |
| 0.3 | Paste `repo_url, bot_username` into the assignment field, hit Check | Validation passes (+0.1) |

Do this before writing a line of code. Registration is worth marks on its own
and the field may close before the code does.

---

## Phase 1 — A bot that can never score zero (2–3 hours)

The goal is *not* correctness yet. It is a bot that always replies, always
replies with exactly one JSON object, and always publishes a log. A shaped
wrong answer is worth more than a crash.

| # | Task | Acceptance test |
|---|---|---|
| 1.1 | FastAPI + python-telegram-bot webhook skeleton | `GET /health` returns 200 |
| 1.2 | `answer_format.py` — extract the message's JSON template, mirror it | `pytest tests/test_answer_format.py` green (16 tests) |
| 1.3 | `run_logger.py` + `GET /run.jsonl` | `wget -qO- $URL/run.jsonl` works from a clean machine |
| 1.4 | Hardcoded stub answer end-to-end | Message the bot from your own account; get one JSON object back |
| 1.5 | Deploy to Render, register webhook | Bot answers within 10s of a message |

**Exit criteria:** you message the bot with the worked example from the
assignment and get back `{"answer": {...}, "log_url": "https://..."}` — one
line, no fences, and the log URL downloads.

---

## Phase 2 — The real agent (4–6 hours)

| # | Task | Acceptance test |
|---|---|---|
| 2.1 | `sandbox.py` — subprocess Python, rlimits, hard timeout | `pytest tests/test_sandbox.py` green |
| 2.2 | `tools.py` — `run_python`, `fetch_url`, `final_answer` | Tool schemas accepted by the API |
| 2.3 | `agent.py` — tool-use loop, temperature 0 | Solves an inline-data question end-to-end |
| 2.4 | Deadline guard + `_force_answer` + `_skeleton` | Set `TURN_DEADLINE_SECONDS=15`; still get a shaped reply |
| 2.5 | Multi-turn context in `conversation.py` | Two-message exchange: second answer uses the first message's data |

**Exit criteria:** an inline-data question (e.g. "forecast these numbers ×1.02,
round to 2dp") is answered *correctly*, computed — not guessed.

---

## Phase 3 — Real datasets (3–5 hours)

This is where the marks actually differentiate. MOSPI-class questions need the
agent to find and parse real data.

| # | Task | Notes |
|---|---|---|
| 3.1 | Harden `fetch_url` (redirects, non-UTF8, content-type) | MOSPI pages are messy |
| 3.2 | Teach the prompt to download inside `run_python`, not eyeball HTML | pandas `read_html`, `read_csv`, `read_excel` |
| 3.3 | Add a "verify before answering" instruction — recompute, sanity-check magnitude | Catches unit errors |
| 3.4 | Normalise Indian state/UT spellings to the source dataset's spelling | Exact match is unforgiving |
| 3.5 | Retry once on empty/failed fetch with a different source | |

**Exit criteria:** the maternal-mortality worked example returns a real state
name derived from a real fetch, visible in `run.jsonl`.

---

## Phase 4 — Adversarial self-grading (2–3 hours)

| # | Task |
|---|---|
| 4.1 | Clone the grader repo, run the full `generate → collect → grade` loop against your live bot |
| 4.2 | Write ≥10 of your own questions into `evals/questions.json`: inline data, list answers, numeric rounding, multi-turn, a URL dataset, a deliberately ambiguous one |
| 4.3 | Fix every `format_error` first, then every wrong answer |
| 4.4 | Chaos tests: kill the LLM key mid-run; send a 4000-char message; send two messages 1s apart; send an emoji-only message. Bot must reply with one JSON object every time |
| 4.5 | Cold-start test: let the service sleep, then message it. Measure time-to-reply |

**Exit criteria:** `grade.json` shows 100% on your own question set, and zero
`format_error` across every chaos test.

---

## Phase 5 — Grading-window hardening (1 hour)

| # | Task |
|---|---|
| 5.1 | Move off Render free → `starter`, or add an UptimeRobot/cron-job.org ping to `/health` every 10 min |
| 5.2 | Confirm the disk mount persists `logs/run.jsonl` across a restart |
| 5.3 | Top up LLM credits; verify the fallback model works by breaking the primary |
| 5.4 | Re-verify `wget` on `log_url` from a machine that has never seen the service |
| 5.5 | Freeze the repo. Final commit. Re-check the registered identifiers |

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Bot sends 2 messages | Grader reads the ack as the answer | Exactly one `reply_text` per handler; enforced by code review + CLAUDE.md |
| Prose leaks into the reply | `format_error`, 0 for that question | `enforce_single_json_object()` gate + 16 tests |
| Agent exceeds 300s exchange budget | `timeout`, 0 | 150s per-turn deadline, forced answer at 20s left |
| Host asleep at grading time | `timeout` across the board | Paid tier or keep-alive ping |
| LLM out of credits / rate-limited | Every question fails | Fallback model, retries, top-up before deadline |
| Log URL 404s or needs auth | Lost log marks | File created at boot; verified with a clean `wget` |
| Wrong answer shape (units, strings vs numbers) | Exact-match failure | Template mirroring + explicit prompt rules |

## Time budget

| Phase | Hours |
|---|---|
| 0 Registration | 0.5 |
| 1 Never-zero bot | 3 |
| 2 Real agent | 5 |
| 3 Real datasets | 4 |
| 4 Self-grading | 3 |
| 5 Hardening | 1 |
| **Total** | **~16.5** |

Front-load Phases 0–1. If you run out of time entirely, a Phase-1 bot with a
guessing agent still scores format and log marks; a half-finished Phase-3 bot
that crashes scores nothing.
