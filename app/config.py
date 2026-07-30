"""Central configuration. Everything comes from environment variables so the
same image runs locally, on Render, and on Railway with no code changes."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the directory above app/). Does nothing if
# the file is absent, which is exactly what we want in production, where
# Render/Railway inject real environment variables instead. override=False so
# a real environment variable always wins over a stale .env line.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    # ALWAYS strip. A trailing newline pasted into a hosting dashboard is
    # invisible there but fatal here: an HTTP header cannot contain a newline,
    # so requests raises ValueError and the OpenAI SDK reports it as
    # "APIConnectionError: Connection error" -- which reads like a network
    # outage and sends you hunting in entirely the wrong place. Cost an hour.
    # No config value in this app legitimately has surrounding whitespace.
    if isinstance(value, str):
        value = value.strip()
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Expected it in {PROJECT_ROOT / '.env'} (copy .env.example to .env "
            f"and fill it in), or set it in the shell / your host's dashboard."
        )
    return value or ""


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except ValueError:
        return default


def url_safe_token(raw: str) -> str:
    """Derive a token that is safe both as a URL path segment and as Telegram's
    `secret_token` header value.

    Render's `generateValue: true` emits base64, which contains '/', '+' and
    '='. A '/' inside the secret splits the `/telegram/{secret}` route into two
    path segments, the route stops matching, and every update comes back 404 --
    which looks exactly like a dead service. Telegram's secret_token field is
    also restricted to [A-Za-z0-9_-], so the raw value is invalid there too.

    Strip to the safe alphabet; if too little survives, hash instead. Either way
    the result is deterministic, so the URL we register and the URL we serve
    always agree.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", raw or "")
    if len(cleaned) >= 12:
        return cleaned[:128]
    return hashlib.sha256((raw or "tds-p1").encode()).hexdigest()[:32]


@dataclass(frozen=True)
class Settings:
    # --- Telegram ---
    telegram_bot_token: str
    webhook_secret: str        # raw, as configured
    webhook_token: str         # URL-safe derivative actually used in the path
    use_webhook: bool

    # --- Public hosting ---
    public_base_url: str          # e.g. https://tds-bot.onrender.com (no trailing slash)
    port: int

    # --- LLM (OpenAI-compatible; AIPipe by default) ---
    llm_base_url: str
    llm_api_key: str
    model: str
    fallback_model: str

    # --- Agent budget ---
    # The grader's timeout_seconds (default 300) covers the WHOLE multi-turn
    # exchange, so a single turn must finish well inside it.
    turn_deadline_seconds: int
    max_tool_calls: int
    sandbox_timeout_seconds: int
    max_history_turns: int

    # --- Logging ---
    log_dir: str


def load_settings() -> Settings:
    base_url = _env("PUBLIC_BASE_URL", "").rstrip("/")
    # Render / Railway inject their own hostname vars; use them as a fallback.
    if not base_url:
        render_host = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
        railway_host = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
        if render_host:
            base_url = render_host
        elif railway_host:
            base_url = f"https://{railway_host}"

    raw_secret = _env("WEBHOOK_SECRET", "tds-p1-hook")

    return Settings(
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN", required=True),
        webhook_secret=raw_secret,
        webhook_token=url_safe_token(raw_secret),
        use_webhook=_env("USE_WEBHOOK", "true").lower() in {"1", "true", "yes"},
        public_base_url=base_url,
        port=_int_env("PORT", 8000),
        llm_base_url=_env("LLM_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/"),
        llm_api_key=_env("LLM_API_KEY", required=True),
        model=_env("LLM_MODEL", "gpt-4.1-mini"),
        fallback_model=_env("LLM_FALLBACK_MODEL", "gpt-4.1-mini"),
        turn_deadline_seconds=_int_env("TURN_DEADLINE_SECONDS", 150),
        max_tool_calls=_int_env("MAX_TOOL_CALLS", 18),
        sandbox_timeout_seconds=_int_env("SANDBOX_TIMEOUT_SECONDS", 45),
        max_history_turns=_int_env("MAX_HISTORY_TURNS", 8),
        log_dir=_resolve_log_dir(_env("LOG_DIR", "logs")),
    )


def _resolve_log_dir(value: str) -> str:
    """A relative LOG_DIR is resolved against the project root, not the current
    working directory -- so `uvicorn app.main:app` writes to the same place no
    matter which directory you launched it from."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
