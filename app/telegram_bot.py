"""Telegram wiring. One incoming message -> exactly one outgoing JSON object."""
from __future__ import annotations

import asyncio
import json
import logging

from telegram import Update
from telegram.constants import ParseMode  # noqa: F401 (documented: we never use it)
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters

from .agent import DataAnalystAgent
from .answer_format import (build_reply, enforce_single_json_object,
                            extract_response_template)
from .config import Settings
from .conversation import ConversationStore
from .run_logger import RunLogger

log = logging.getLogger("tds.bot")


def build_application(settings: Settings) -> Application:
    app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(True)
        .build()
    )
    app.bot_data["settings"] = settings
    app.bot_data["agent"] = DataAnalystAgent(settings)
    app.bot_data["store"] = ConversationStore()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_message))
    app.add_handler(MessageHandler(filters.COMMAND, on_command))
    app.add_error_handler(on_error)
    return app


async def on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start, /reset etc. Still answered with a single JSON object so a stray
    command from the grader can never produce prose."""
    settings: Settings = context.bot_data["settings"]
    store: ConversationStore = context.bot_data["store"]
    text = (update.effective_message.text or "").strip().lower()
    if text.startswith("/reset"):
        store.reset(update.effective_chat.id)
    await update.effective_message.reply_text(
        json.dumps({"status": "ready", "log_url": f"{settings.public_base_url}/run.jsonl"}),
        disable_web_page_preview=True,
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    agent: DataAnalystAgent = context.bot_data["agent"]
    store: ConversationStore = context.bot_data["store"]

    message = update.effective_message
    chat_id = update.effective_chat.id
    text = message.text or ""

    # Serialise turns within a chat so a fast second message can't race the
    # first one's answer (the grader is strictly sequential anyway).
    async with store.lock(chat_id):
        conversation = store.add_user(chat_id, text)
        logger = RunLogger(settings.log_dir, settings.public_base_url, chat_id)
        logger.message_received(text, store.turn_index(chat_id))

        template = extract_response_template(text)
        logger.event("template_extracted", template=template,
                     found=template is not None)

        try:
            answer = await asyncio.to_thread(agent.solve, conversation, template, logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("agent", f"{type(exc).__name__}: {exc}")
            answer = {}

        reply = build_reply(answer, logger.log_url, template)

        # Final gate: whatever happens, what leaves this process is exactly one
        # JSON object.
        try:
            reply = enforce_single_json_object(reply)
        except ValueError:
            reply = json.dumps({"answer": {}, "log_url": logger.log_url})

        logger.final(answer, reply)
        store.add_assistant(chat_id, reply)

    # NOTE: exactly one send_message per incoming message. Never send a
    # "thinking..." ack -- collect.py takes the FIRST reply as the answer.
    await message.reply_text(reply, disable_web_page_preview=True)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("unhandled telegram error", exc_info=context.error)
    try:
        settings: Settings = context.bot_data["settings"]
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                json.dumps({"answer": {},
                            "log_url": f"{settings.public_base_url}/run.jsonl"}),
                disable_web_page_preview=True,
            )
    except Exception:  # noqa: BLE001
        pass
