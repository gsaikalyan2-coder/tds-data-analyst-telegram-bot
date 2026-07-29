# TDS P1 — Data-Analyst Telegram Bot

An LLM agent on Telegram that answers a data-analysis question by *actually
computing* the answer (writing and running Python against inline data or public
datasets), then replies with **exactly one JSON object**:

```json
{"answer": {"state": "Assam"}, "log_url": "https://your-service.onrender.com/run.jsonl"}
```

## Architecture

```
Telegram user (grader account, via Telethon)
        │  plain-text question
        ▼
POST /telegram/<secret>        FastAPI + python-telegram-bot (webhook)
        │
        ├─ conversation.py     per-chat turn history; one lock per chat
        ├─ answer_format.py    extract the JSON template the message asked for
        │
        ▼
    agent.py  ── tool-use loop (OpenAI-compatible, AIPipe by default)
        │        ├─ run_python  → sandbox.py (subprocess, rlimits, timeout)
        │        ├─ fetch_url   → requests
        │        └─ final_answer
        │        deadline guard → forced answer → key-correct skeleton
        ▼
    run_logger.py  append JSONL ─────────► logs/run.jsonl  &  logs/runs/<id>.jsonl
        │                                        │
        ▼                                        ▼
  build_reply() + enforce_single_json_object()   GET /run.jsonl   ← log_url
        │
        ▼
  exactly one reply_text()  ──────────────► Telegram
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in TELEGRAM_BOT_TOKEN and LLM_API_KEY
python3 scripts/selftest.py   # offline format check -> "all good"
pytest tests/ -q              # 16 tests

# run in polling mode (no public URL needed)
USE_WEBHOOK=false uvicorn app.main:app --port 8000
```

Then message your bot from your own Telegram account:

```
Which state has the highest maternal mortality rate based on MOSPI data? Reply with ONLY this JSON object and nothing else: {"answer": {"state": "<state name>"}, "log_url": "<public wget-able URL to your agent's JSONL log>"}
```

## Deploy to Render

1. Push this repo to a **public** GitHub repo.
2. Render → New → Blueprint → point at the repo (`render.yaml` is picked up).
3. Set the two secret env vars: `TELEGRAM_BOT_TOKEN`, `LLM_API_KEY`.
4. Set `PUBLIC_BASE_URL` to the service URL Render assigns.
5. Deploy. The webhook registers itself on boot.

Verify:

```bash
curl -s https://your-service.onrender.com/health
wget -qO- https://your-service.onrender.com/run.jsonl | head -3
```

Both must work from a machine with no cookies or auth.

## Test against the real grader

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot
pip install -r requirements.txt
cp .env.example .env          # TELEGRAM_API_ID / API_HASH from my.telegram.org
python3 login.py              # paste the printed session string into .env
echo "email,github_url,telegram_bot_username" > students.csv
echo "you@example.com,https://github.com/you/repo,your_data_bot" >> students.csv
# add your own questions to evals/questions.json
python3 generate.py --students students.csv
python3 collect.py  --students students.csv
python3 grade.py    --students students.csv
cat data/*/grade.json
```

## Environment variables

See `.env.example`. The only two with no sensible default are
`TELEGRAM_BOT_TOKEN` and `LLM_API_KEY`.

## Rules that decide the score

See [CLAUDE.md](./CLAUDE.md). Short version: one reply per message, never
silent, exactly one JSON object, mirror the requested shape, public `run.jsonl`.
