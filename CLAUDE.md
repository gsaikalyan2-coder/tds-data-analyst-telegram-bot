# CLAUDE.md — single source of truth

Read this before changing anything. The rules below are derived from the
*actual grading code*, not from guesses.

## What this is

A Telegram bot that answers one data-analysis question per message and replies
with **exactly one JSON object and nothing else**. TDS Project 1, Task 2, 37.5
marks. Graded by `github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot`.

## Non-negotiable rules (violating any of these costs marks)

1. **One reply per incoming message. Exactly one.**
   `collect.py` does `conv.send_message(msg)` then `await conv.get_response()`
   — it takes the **first** message the bot sends back. A "thinking…" ack would
   be captured as the answer and everything after it discarded. Never send a
   progress message, a typing chunk, or a second message.

2. **Never stay silent.** No reply inside `timeout_seconds` (default 300) is
   recorded as `timeout` — zero marks. Every code path must end in a reply,
   including exceptions. `on_error` in `telegram_bot.py` is the backstop.

3. **The reply is exactly one JSON object.** No markdown fences, no leading
   "Here is", no trailing newline commentary. Prose around it is a
   `format_error`. `enforce_single_json_object()` is the last gate before send.

4. **Mirror the shape the message asked for.** The message always spells out
   its own template. Two forms exist in the wild:
   - `{"answer": {...}, "log_url": "..."}` — the 2026 assignment contract
   - `{"state": "..."}` — the bare shape in the public `evals/questions.json`
   `extract_response_template()` reads the literal template out of the message
   and `build_reply()` mirrors it. **Do not hardcode either form.**

5. **Grading is exact match.** Same keys, same spelling, same capitalisation,
   same list order, same rounding. Numbers as JSON numbers unless the template
   shows strings. No units, no thousands separators.

6. **Multi-turn: answer every message as if it were the last.**
   The bot cannot know which message is final. Earlier turns are kept as
   context (`conversation.py`); each message gets a full agent run and one JSON
   reply. The last reply is therefore the answer to the last message — which is
   exactly what's graded.

7. **`log_url` must be public and `wget`-able**, serving JSONL, one JSON object
   per line. Served by this same service at `GET /run.jsonl`. It is created at
   boot so it is never a 404.

## Time budget

`timeout_seconds` (300s default) covers the **whole multi-turn exchange**, not
one message. So a single turn must finish fast:

| knob | default | why |
|---|---|---|
| `TURN_DEADLINE_SECONDS` | 150 | hard wall-clock for one agent run |
| `MAX_TOOL_CALLS` | 12 | loop bound |
| `SANDBOX_TIMEOUT_SECONDS` | 45 | one `run_python` call |

At 20s remaining the agent stops analysing and forces a shaped best-effort
answer (`_force_answer`). At absolute worst it emits a key-correct skeleton
(`_skeleton`). A wrong-but-shaped answer beats a malformed one.

## Module map

| file | responsibility | change with care |
|---|---|---|
| `app/answer_format.py` | template extraction + reply shaping | **highest risk file** — every change needs tests |
| `app/agent.py` | LLM tool-use loop, deadline, fallbacks | |
| `app/tools.py` | tool specs + dispatch | |
| `app/sandbox.py` | subprocess Python execution | |
| `app/conversation.py` | per-chat multi-turn state | |
| `app/telegram_bot.py` | handlers; exactly one `reply_text` | **only one send per message** |
| `app/run_logger.py` | JSONL logging | |
| `app/main.py` | FastAPI, webhook, `/run.jsonl` | |

## Conventions

- Python 3.12, no framework beyond FastAPI + python-telegram-bot + openai.
- The agent is sync; it runs in `asyncio.to_thread` so the event loop stays free.
- Every module-level failure path logs to JSONL before it returns.
- `pytest tests/ -q` must be green before any push. `scripts/selftest.py` must
  print `all good`.

## Things that will bite you

- Render's **free** plan sleeps after 15 min idle. Cold start can eat 50s+ of
  the grading budget. Use `starter`, or keep a 10-minute cron ping on `/health`.
- Render's filesystem is ephemeral without a disk — the `disk:` block in
  `render.yaml` is what makes `run.jsonl` survive restarts.
- Bots cannot message bots. The grader logs in as a **user account** via
  Telethon, so your bot must accept messages from ordinary users (it does by
  default — do not add a whitelist).
- Set the webhook only after `application.start()`, and never run polling and
  webhook at the same time.
