"""THE most important module in this repo.

The grader does an exact comparison against a JSON object parsed out of the
bot's reply, and any prose around it is a `format_error`. So the reply we send
to Telegram must be:

  * exactly one JSON object,
  * nothing else -- no markdown fences, no "Here is the answer:", no trailing
    newline commentary,
  * shaped exactly the way the incoming message asked for.

The incoming message always spells out its own shape, e.g.

    ... Reply with ONLY this JSON object and nothing else:
    {"answer": {"state": "<state name>"}, "log_url": "<public URL>"}

or, in the older/simpler eval format,

    ... Reply with ONLY a JSON object like {"state": "<state name>"}

So we do not hardcode either shape. We *extract the requested template from
the message* and mirror it. That is strictly safer than assuming.
"""
from __future__ import annotations

import json
import re
from typing import Any

ANSWER_KEY = "answer"
LOG_KEY = "log_url"


# --------------------------------------------------------------------------
# 1. Pull candidate JSON-ish object literals out of free text
# --------------------------------------------------------------------------
def iter_brace_blocks(text: str):
    """Yield every balanced {...} block in `text`, outermost first.

    Brace-counting (not regex) so nested objects survive. Braces inside string
    literals are ignored.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[i:j + 1]
                    break
            j += 1
        i = j + 1 if j > i else i + 1


# A placeholder may or may not already be inside quotes:
#     {"state": "<state name>"}   <- quoted; must replace INCLUDING the quotes
#     {"total": <number>}         <- bare;   must replace and ADD quotes
# Substituting a quoted replacement into an already-quoted slot yields
# ""__PLACEHOLDER__"", which is invalid JSON -- so the quoted form is matched
# first, with its surrounding quotes consumed.
_QUOTED_PLACEHOLDER = re.compile(r'"\s*<[^<>"]*>\s*"')
_PLACEHOLDER = re.compile(r'<[^<>{}"]*>')


def _relax_to_json(block: str) -> str:
    """Turn a human-written template into parseable JSON.

    `{"answer": {"state": "<state name>"}, "log_url": "<public URL>"}` is
    already valid JSON. But templates like `{"values": [<numbers>]}` or
    `{"state": <state name>}` are not -- the angle-bracket placeholders sit
    where a value should be. Replace bare placeholders with null, and quoted
    ones with a marker string.
    """
    # Quoted placeholders first, consuming their own quotes, then bare ones.
    # Order matters: doing it the other way round produces ""__PLACEHOLDER__"".
    out = _QUOTED_PLACEHOLDER.sub('"__PLACEHOLDER__"', block)
    out = _PLACEHOLDER.sub('"__PLACEHOLDER__"', out)
    out = re.sub(r",\s*([}\]])", r"\1", out)          # trailing commas
    out = out.replace("'", '"')                        # single-quoted keys
    return out


def parse_template(block: str) -> dict | None:
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_relax_to_json(block))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def extract_response_template(message: str) -> dict | None:
    """Return the JSON skeleton the message is asking us to reply with.

    Preference order:
      1. a block containing BOTH "answer" and "log_url"  (the 2026 contract)
      2. the LAST parseable object block in the message  (older bare-shape evals)
    """
    candidates = []
    for block in iter_brace_blocks(message):
        parsed = parse_template(block)
        if parsed is not None:
            candidates.append(parsed)
    if not candidates:
        return None
    for cand in candidates:
        if ANSWER_KEY in cand and LOG_KEY in cand:
            return cand
    return candidates[-1]


# --------------------------------------------------------------------------
# 2. Describe the required ANSWER shape to the model
# --------------------------------------------------------------------------
def answer_schema_hint(template: dict | None) -> str:
    """A short natural-language description of what `answer` must look like."""
    if template is None:
        return (
            "The message did not include an explicit JSON template. Produce the "
            "most literal, minimal answer object that the question implies."
        )
    if ANSWER_KEY in template and LOG_KEY in template:
        inner = template[ANSWER_KEY]
        return (
            "Your answer MUST have exactly this shape (same keys, same types, "
            "same ordering of any lists):\n" + json.dumps(inner, ensure_ascii=False)
        )
    return (
        "Your answer MUST have exactly this shape (same keys, same types, "
        "same ordering of any lists):\n" + json.dumps(template, ensure_ascii=False)
    )


def wants_wrapper(template: dict | None) -> bool:
    """True when the message asked for {"answer": ..., "log_url": ...}."""
    if template is None:
        return True  # assignment default contract
    return ANSWER_KEY in template and LOG_KEY in template


# --------------------------------------------------------------------------
# 3. Build the outgoing reply -- exactly one JSON object, nothing else
# --------------------------------------------------------------------------
def build_reply(answer: Any, log_url: str, template: dict | None) -> str:
    """Serialise the final Telegram reply text."""
    if wants_wrapper(template):
        payload: Any = {ANSWER_KEY: answer, LOG_KEY: log_url}
    else:
        # Message asked for a bare shape (e.g. {"state": "..."}). Mirror it
        # exactly -- adding extra keys would fail an exact-match grader.
        payload = answer if isinstance(answer, dict) else {ANSWER_KEY: answer}
    return json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))


def coerce_answer(raw: Any, template: dict | None) -> Any:
    """Normalise whatever the model produced into the requested answer shape.

    Handles the common failure modes: the model wrapping its answer in
    {"answer": ...} when it shouldn't, returning a JSON string instead of an
    object, or returning the full outer envelope.
    """
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                pass

    if isinstance(raw, dict):
        # Model returned the whole envelope -> unwrap it.
        if ANSWER_KEY in raw and LOG_KEY in raw:
            raw = raw[ANSWER_KEY]

    inner_template = None
    if template is not None:
        inner_template = template.get(ANSWER_KEY) if wants_wrapper(template) else template

    # Model wrapped a single-key answer one level too deep.
    if (
        isinstance(raw, dict)
        and isinstance(inner_template, dict)
        and set(raw.keys()) == {ANSWER_KEY}
        and set(inner_template.keys()) != {ANSWER_KEY}
    ):
        raw = raw[ANSWER_KEY]

    # Same mistake, but with no template to compare against (the message did
    # not spell out a shape). A lone "answer" key is far more likely to be the
    # model echoing the envelope than a genuine answer field, and build_reply
    # is about to wrap it again -- so unwrap it here.
    if (
        inner_template is None
        and isinstance(raw, dict)
        and set(raw.keys()) == {ANSWER_KEY}
    ):
        raw = raw[ANSWER_KEY]

    return raw


# --------------------------------------------------------------------------
# 4. Last line of defence -- never let non-JSON reach Telegram
# --------------------------------------------------------------------------
def enforce_single_json_object(text: str) -> str:
    """Guarantee the outgoing string is exactly one JSON object.

    If `text` is already a single valid JSON object, it is re-serialised
    canonically. Otherwise the first balanced {...} block that parses is used.
    Raises ValueError if nothing usable is found -- callers must then fall back
    to a hardcoded safe envelope.
    """
    stripped = text.strip()
    # strip markdown fences if any snuck in
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
    except json.JSONDecodeError:
        pass
    for block in iter_brace_blocks(stripped):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
    raise ValueError("no single JSON object found in text")
