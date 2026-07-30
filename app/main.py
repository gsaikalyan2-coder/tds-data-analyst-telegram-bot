"""FastAPI entrypoint.

Serves three things from one process:
  POST /telegram/<secret>   Telegram webhook
  GET  /run.jsonl           the public, wget-able cumulative run log  <-- log_url
  GET  /logs/<run_id>.jsonl a single run's log
  GET  /health              keep-alive / uptime probe

Run:  uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from telegram import Update

from .config import load_settings
from .run_logger import ensure_log_files
from .telegram_bot import build_application

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("tds.main")

settings = load_settings()
ensure_log_files(settings.log_dir)
LOG_DIR = Path(settings.log_dir)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    application = build_application(settings)
    app.state.application = application
    await application.initialize()
    await application.start()

    if settings.use_webhook and settings.public_base_url:
        # webhook_token, not webhook_secret: the raw value may be base64 and
        # contain '/', which would split the path and 404 every update.
        url = f"{settings.public_base_url}/telegram/{settings.webhook_token}"
        await application.bot.set_webhook(
            url=url,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            max_connections=40,
            secret_token=settings.webhook_token,
        )
        log.info("webhook set: %s", url)
    else:
        await application.bot.delete_webhook(drop_pending_updates=True)
        await application.updater.start_polling(drop_pending_updates=True)
        log.info("polling started (no PUBLIC_BASE_URL or USE_WEBHOOK=false)")

    try:
        yield
    finally:
        # Updates are now processed in background tasks, so a shutdown mid-run
        # would drop a reply the grader is waiting on. Give them a bounded
        # chance to finish -- silence scores zero, a late reply may not.
        if _BACKGROUND_TASKS:
            log.info("draining %d in-flight update(s)", len(_BACKGROUND_TASKS))
            with contextlib.suppress(Exception):
                await asyncio.wait(set(_BACKGROUND_TASKS), timeout=30)
        with contextlib.suppress(Exception):
            if application.updater and application.updater.running:
                await application.updater.stop()
        with contextlib.suppress(Exception):
            await application.stop()
        with contextlib.suppress(Exception):
            await application.shutdown()


app = FastAPI(title="TDS P1 Data-Analyst Telegram Bot", lifespan=lifespan)


# --- duplicate-delivery protection ----------------------------------------
# Telegram re-delivers an update if the webhook does not return 200 quickly
# (its patience is on the order of a minute). An agent run can take longer than
# that, so a slow question was being delivered TWICE and answered TWICE.
#
# collect.py takes the FIRST message the bot sends back after each send, so on a
# multi-turn question a stray duplicate of reply 1 gets read as the answer to
# message 2 -- silently scoring the wrong thing.
#
# Two independent defences:
#   1. ACK IMMEDIATELY, process in the background -> Telegram never retries.
#   2. Deduplicate on update_id -> even a retry that slips through is dropped.
_SEEN_UPDATES: OrderedDict[int, float] = OrderedDict()
_SEEN_MAX = 1000
_BACKGROUND_TASKS: set = set()


def _already_seen(update_id: int | None) -> bool:
    if update_id is None:
        return False
    if update_id in _SEEN_UPDATES:
        return True
    _SEEN_UPDATES[update_id] = time.time()
    while len(_SEEN_UPDATES) > _SEEN_MAX:
        _SEEN_UPDATES.popitem(last=False)
    return False


@app.post("/telegram/{secret}")
async def telegram_webhook(secret: str, request: Request) -> Response:
    # Accept the derived token, and the raw secret too, so an already-registered
    # webhook from a previous deploy keeps working until it re-registers.
    if secret not in (settings.webhook_token, settings.webhook_secret):
        raise HTTPException(status_code=403, detail="bad secret")
    header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if header is not None and header != settings.webhook_token:
        raise HTTPException(status_code=403, detail="bad secret token header")

    data = await request.json()
    update = Update.de_json(data, request.app.state.application.bot)

    if _already_seen(getattr(update, "update_id", None)):
        log.warning("duplicate update_id %s dropped", update.update_id)
        return Response(status_code=200)

    # Fire and forget. Returning 200 now is what stops the retry that caused
    # the double reply. Keep a strong reference so the task is not garbage
    # collected mid-run.
    task = asyncio.create_task(request.app.state.application.process_update(update))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    return Response(status_code=200)


@app.get("/run.jsonl")
def run_log() -> Response:
    path = LOG_DIR / "run.jsonl"
    body = path.read_bytes() if path.exists() else b""
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": 'inline; filename="run.jsonl"',
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/logs/{run_id}.jsonl")
def single_run_log(run_id: str) -> Response:
    if "/" in run_id or ".." in run_id:
        raise HTTPException(status_code=400, detail="bad run id")
    path = LOG_DIR / "runs" / f"{run_id}.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no such run")
    return Response(
        content=path.read_bytes(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "webhook": settings.use_webhook,
        "base_url": settings.public_base_url,
        "log_url": f"{settings.public_base_url}/run.jsonl",
        "model": settings.model,
        # The path Telegram must POST to. If getWebhookInfo shows a different
        # one, the registration is stale -- redeploy to re-register.
        "webhook_path": f"/telegram/{settings.webhook_token}",
    }


@app.get("/debug/llm")
def debug_llm() -> dict:
    """Run one LLM request from INSIDE the container and report the raw result.

    /health only reports configuration -- it never touches the LLM, so it stays
    green with a dead key or an unreachable endpoint. This actually calls out,
    and returns the exact exception type and message plus a plain-socket
    reachability probe, so a connection failure can be told apart from an auth
    failure, a bad model, or a quota problem.

    Diagnostic only. Safe to leave: it exposes no secrets, only lengths and
    prefixes.
    """
    import socket
    import ssl
    from urllib.parse import urlparse

    out: dict = {
        "base_url": settings.llm_base_url,
        "model": settings.model,
        "api_key_len": len(settings.llm_api_key),
        "api_key_prefix": settings.llm_api_key[:6],
        "api_key_has_whitespace": settings.llm_api_key != settings.llm_api_key.strip(),
    }

    # 1. Can we open a TLS socket to the host at all?
    host = urlparse(settings.llm_base_url).hostname or ""
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=15) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                out["tcp_tls"] = f"ok (TLS {ssock.version()})"
    except Exception as exc:  # noqa: BLE001
        out["tcp_tls"] = f"FAILED {type(exc).__name__}: {exc}"

    # 2. Raw HTTP POST, bypassing the OpenAI SDK entirely.
    try:
        import requests as _rq
        r = _rq.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}",
                     "Content-Type": "application/json"},
            json={"model": settings.model,
                  "messages": [{"role": "user", "content": "say pong"}],
                  "max_tokens": 5},
            timeout=30,
        )
        out["raw_http_status"] = r.status_code
        out["raw_http_body"] = (r.text or "")[:600]
    except Exception as exc:  # noqa: BLE001
        out["raw_http_status"] = None
        out["raw_http_body"] = f"{type(exc).__name__}: {exc}"

    # 3. The same call through the SDK the agent actually uses.
    try:
        agent = app.state.application.bot_data["agent"]
        resp = agent.client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": "say pong"}],
            max_tokens=5,
        )
        out["sdk"] = f"ok: {resp.choices[0].message.content!r}"
    except Exception as exc:  # noqa: BLE001
        out["sdk"] = f"{type(exc).__name__}: {exc}"

    return out


@app.get("/")
def root() -> dict:
    return {"service": "tds-p1-data-analyst-telegram-bot",
            "log_url": f"{settings.public_base_url}/run.jsonl"}
