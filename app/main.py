"""FastAPI entrypoint.

Serves three things from one process:
  POST /telegram/<secret>   Telegram webhook
  GET  /run.jsonl           the public, wget-able cumulative run log  <-- log_url
  GET  /logs/<run_id>.jsonl a single run's log
  GET  /health              keep-alive / uptime probe

Run:  uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import contextlib
import logging
import os
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
        with contextlib.suppress(Exception):
            if application.updater and application.updater.running:
                await application.updater.stop()
        with contextlib.suppress(Exception):
            await application.stop()
        with contextlib.suppress(Exception):
            await application.shutdown()


app = FastAPI(title="TDS P1 Data-Analyst Telegram Bot", lifespan=lifespan)


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
    await request.app.state.application.process_update(update)
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


@app.get("/")
def root() -> dict:
    return {"service": "tds-p1-data-analyst-telegram-bot",
            "log_url": f"{settings.public_base_url}/run.jsonl"}
