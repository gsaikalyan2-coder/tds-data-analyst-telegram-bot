"""The data-analyst agent loop.

One `solve()` call == one incoming Telegram message == one JSONL run log.

Contract with the rest of the app:
  * solve() ALWAYS returns an answer object. It never raises. A wrong answer
    still scores the format marks; an exception scores nothing.
  * solve() respects a hard wall-clock deadline so the grader's per-exchange
    timeout is never hit.
"""
from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from .answer_format import answer_schema_hint, coerce_answer, wants_wrapper
from .config import Settings
from .run_logger import RunLogger
from .tools import TOOL_SPECS, dispatch, parse_tool_args

SYSTEM_PROMPT = """\
You are a rigorous data analyst. You answer one data-analysis question by
actually computing the answer, then you submit it.

HOW YOU WORK
1. Read the question and identify precisely what value(s) are being asked for
   and in what unit/format.
2. If the data is embedded in the message, use it verbatim -- do not invent,
   round, reorder or "clean" it unless asked. Go straight to run_python; do not
   search.
3. If the question names a public dataset (MOSPI, SRS, NFHS, data.gov.in, RBI,
   Census, World Bank, ...) and does not embed the data, your FIRST action is
   web_search. Never guess a URL from memory -- guessed URLs 404 and waste your
   clock. Search, read the results, then fetch_url the most authoritative hit
   (prefer the primary government source over a news article about it).
   Write search queries that pin down the country and the actual publication:
   "MOSPI maternal mortality by state" returns US CDC pages. "SRS special
   bulletin maternal mortality ratio India statewise" returns the real one.
   Always name the country. Prefer the underlying statistical series (SRS,
   NFHS, Census) over the ministry portal, because the portal is usually a
   landing page and the series is the actual data.
   If a source is a PDF, fetch the bytes in run_python and parse with
   pdfplumber. If it is .xls/.xlsx/.csv, parse with pandas.
4. Write small, self-checking Python. Print intermediate values. Each
   run_python call is a FRESH process -- no variables carry over.
5. When a computation disagrees with your prior belief, trust the computation.
6. Call final_answer exactly once when you have the value.

CHOOSING THE RIGHT SOURCE -- DO NOT JUST TAKE SEARCH HIT #1
* Prefer the primary statistical release over anything that summarises it: the
  SRS Special Bulletin over a yearbook chapter, the NFHS factsheet over a news
  article, censusindia.gov.in / mospi.gov.in over an aggregator.
* Prefer the most recent edition unless the question names a year.
* Before you extract anything, confirm the document actually contains the
  indicator. If it does not, that is not a puzzle to solve by looking harder at
  the numbers that ARE there -- it is the wrong document. Search again.
* A general publication ("Women & Men in India", a yearbook chapter) usually
  does NOT carry the detailed state-wise table. The dedicated bulletin does.

READING PDFs
Use the read_pdf tool, never page-by-page probing in run_python. One read_pdf
call searches the entire document and returns the matching pages with their
tables. Walking `pdf.pages[0]`, `[1]`, `[2]`... costs one tool call each and
will exhaust your budget before you reach the table.

VERIFY THE COLUMN BEFORE YOU TRUST A NUMBER -- THIS IS WHERE ANSWERS GO WRONG
Statistical tables place unrelated indicators side by side. Crude Birth Rate,
Crude Death Rate, Infant Mortality Rate and Maternal Mortality Ratio commonly
sit in adjacent columns of the same table. Reading the wrong column produces a
confident, plausible, completely wrong answer.
So, before extracting:
  1. Print the header row and the row labels.
  2. Confirm in writing which column header matches the quantity asked for.
  3. Confirm the reference year/period.
  4. Sanity-check the magnitude. Indian MMR is per 100,000 live births and runs
     roughly 20-200. IMR is per 1,000 and runs roughly 5-60. If your numbers
     are in the wrong band for the indicator, you read the wrong column.
Then report that header verbatim in final_answer's `indicator` field. If you
cannot name the exact header you read, you do not have a verified answer.

NEVER FABRICATE INPUT DATA -- THIS IS ABSOLUTE
Every number you compute over must come from the message itself or from a
document you successfully fetched in THIS run. You are forbidden to:
  * write a "sample", "example", "simulated", "representative", "approximate"
    or "known" dataset into run_python and compute over it;
  * hardcode figures recalled from memory and present them as retrieved;
  * cite a URL you did not successfully fetch.
If a fetch fails, that is information: try a different search query or a
different source. Two failed fetches do not license you to make the numbers up.
A computation over invented data is worse than no computation at all, because
it looks correct and is not.

IF YOU CANNOT RETRIEVE THE DATA
Do not stall and do not fabricate. Give your single best answer from general
knowledge, in the exact requested shape, and set the SEPARATE `source`
parameter to "NO SOURCE - could not retrieve data" so the run log records that
it was unverified.

`answer_json` and `source` are two different parameters and must never be
confused. `answer_json` ALWAYS contains a real, plausible answer of the type
the question asked for -- a state name, a number, a list. It must NEVER contain
an error message, a status string, an explanation, null, or the words
"NO SOURCE". If the question asks which state, `answer_json` is
{"state": "<an actual state name>"} even when you are unsure. Put your caveat
in `source` and `confidence`, never in the answer.

RULES THAT DECIDE YOUR SCORE
* The answer is compared by exact match. Match the requested shape EXACTLY:
  same keys, same spelling, same capitalisation, same list order, same numeric
  rounding. If the question says "round to 2 decimals", round to 2 decimals.
* Return numbers as JSON numbers, not strings, unless the template shows a
  string.
* Never include units, currency symbols, commas in numbers, or explanatory
  text inside the answer values unless the template shows them.
* For Indian state/UT names use the standard official spelling as it appears
  in the source dataset.
* If you genuinely cannot compute it, still call final_answer with your best
  supported estimate in the correct shape. An empty or malformed answer scores
  zero; a shaped best-effort answer might not.

TIME
You are on a hard clock. Do not browse speculatively. Typical good runs:
  inline data : run_python -> final_answer
  HTML source : web_search -> fetch_url -> run_python -> final_answer
  PDF source  : web_search -> read_pdf -> run_python -> final_answer
Aim for 4-7 tool calls. Never spend more than two attempts on a single source
before trying a different one -- if a document is not yielding the indicator
after two looks, it is the wrong document.
If you are running out of time you will be told to answer immediately.
"""


_STATUS_MARKERS = (
    "no source", "could not retrieve", "could not find", "not available",
    "unable to", "unknown", "n/a", "error", "failed", "no data",
)


def _contains_status_string(answer: Any) -> bool:
    """True if the model leaked a status/error message into the answer value."""
    def walk(node: Any) -> bool:
        if isinstance(node, str):
            low = node.strip().lower()
            return any(m in low for m in _STATUS_MARKERS)
        if isinstance(node, dict):
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False
    return walk(answer)


class DataAnalystAgent:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=90.0,
            max_retries=2,
        )

    # ------------------------------------------------------------------ api
    def solve(
        self,
        conversation: list[dict],
        template: dict | None,
        logger: RunLogger,
    ) -> Any:
        """`conversation` is the full turn history, oldest first:
        [{"role": "user", "text": ...}, {"role": "assistant", "text": ...}, ...]
        The LAST user message is the one being answered."""
        deadline = time.time() + self.s.turn_deadline_seconds
        messages = self._build_messages(conversation, template)
        logger.event("agent_start", model=self.s.model,
                     template=template, deadline_s=self.s.turn_deadline_seconds)

        answer: Any = None
        for step in range(self.s.max_tool_calls):
            remaining = deadline - time.time()
            if remaining <= 20:
                logger.event("deadline_pressure", remaining_s=round(remaining, 1))
                answer = self._force_answer(messages, template, logger)
                break

            try:
                resp = self.client.chat.completions.create(
                    model=self.s.model,
                    messages=messages,
                    tools=TOOL_SPECS,
                    tool_choice="auto",
                    temperature=0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("llm_call", f"{type(exc).__name__}: {exc}")
                if not self._swap_to_fallback(logger):
                    break
                continue

            choice = resp.choices[0].message
            if choice.content:
                logger.plan(choice.content)

            tool_calls = choice.tool_calls or []
            messages.append({
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ] or None,
            })
            if messages[-1]["tool_calls"] is None:
                messages[-1].pop("tool_calls")

            if not tool_calls:
                # Model answered in prose. Salvage any JSON it emitted.
                answer = self._salvage(choice.content or "", template)
                if answer is not None:
                    logger.event("answer_salvaged_from_prose")
                    break
                messages.append({
                    "role": "user",
                    "content": "You must use the tools. Call final_answer now with the answer JSON.",
                })
                continue

            finished = False
            for tc in tool_calls:
                name = tc.function.name
                args = parse_tool_args(tc.function.arguments)

                if name == "final_answer":
                    source = (args.get("source") or "").strip()
                    unsourced = (
                        not source
                        or source.upper().startswith("NO SOURCE")
                    )
                    logger.event(
                        "final_answer_tool",
                        arguments=args,
                        source=source or "(none declared)",
                        indicator=args.get("indicator") or "(none declared)",
                        reference_period=args.get("reference_period") or "(none declared)",
                        unsourced=unsourced,
                        confidence=args.get("confidence"),
                    )
                    if unsourced:
                        # Not fatal -- a shaped unverified answer still scores
                        # better than silence -- but it must be visible in the
                        # log so a bad run is diagnosable at a glance.
                        logger.event("warning_unsourced_answer",
                                     detail="answer submitted without a retrieved source")
                    answer = self._parse_final(args, template)
                    if _contains_status_string(answer):
                        # The model leaked a status/error string into the answer
                        # instead of the `source` field. Reject it and make it
                        # answer properly -- a sentinel string would be graded
                        # as a wrong answer AND look broken in the log.
                        logger.event("rejected_status_string_answer", answer=answer)
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": json.dumps({
                                "ok": False,
                                "error": (
                                    "answer_json contained a status/error string. "
                                    "answer_json must contain ONLY a real answer value "
                                    "of the requested type. Put any caveat in `source`. "
                                    "Call final_answer again with a real best-effort value."
                                ),
                            }),
                        })
                        answer = None
                        continue
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": '{"ok": true}'})
                    finished = True
                    break

                logger.tool_call(name, args)
                result = dispatch(name, args, self.s.sandbox_timeout_seconds)
                logger.tool_result(name, bool(result.get("ok")), result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str)[:20000],
                })

            if finished:
                break

        if answer is None:
            logger.event("no_answer_after_loop")
            answer = self._force_answer(messages, template, logger)
        if answer is None:
            answer = self._skeleton(template)
            logger.event("answer_skeleton_fallback", answer=answer)

        return coerce_answer(answer, template)

    # -------------------------------------------------------------- helpers
    def _build_messages(self, conversation: list[dict], template: dict | None) -> list[dict]:
        history_note = ""
        prior = conversation[:-1]
        if prior:
            lines = []
            for turn in prior[-self.s.max_history_turns:]:
                who = "USER" if turn["role"] == "user" else "YOU"
                lines.append(f"{who}: {turn['text']}")
            history_note = (
                "Earlier messages in this same conversation (context for the "
                "question you must answer now):\n" + "\n".join(lines) + "\n\n"
            )

        current = conversation[-1]["text"] if conversation else ""
        user_block = (
            f"{history_note}QUESTION TO ANSWER NOW:\n{current}\n\n"
            f"REQUIRED ANSWER SHAPE:\n{answer_schema_hint(template)}\n\n"
            "Submit via the final_answer tool. `answer_json` must contain ONLY "
            "the answer object in that shape"
            + (" (do NOT include log_url -- the server adds it)."
               if wants_wrapper(template) else ".")
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ]

    def _swap_to_fallback(self, logger: RunLogger) -> bool:
        if self.s.model == self.s.fallback_model:
            return False
        logger.event("model_fallback", to=self.s.fallback_model)
        object.__setattr__(self.s, "model", self.s.fallback_model)
        return True

    # The model occasionally invents a plausible parameter name instead of the
    # declared one -- "answer", "result", "value". Missing it costs a whole
    # extra round trip through _force_answer, which near the deadline is the
    # difference between answering and timing out. So accept the aliases.
    _ANSWER_KEYS = ("answer_json", "answer", "result", "value", "output")

    def _parse_final(self, args: dict, template: dict | None) -> Any:
        raw = None
        for key in self._ANSWER_KEYS:
            if args.get(key) is not None:
                raw = args[key]
                break
        if raw is None:
            return None
        if isinstance(raw, (dict, list, int, float, bool)):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            salvaged = self._salvage(str(raw), template)
            if salvaged is not None:
                return salvaged
            # Not JSON and nothing to salvage -- but a bare scalar like
            # "Assam" or "hello" is a legitimate answer for a scalar-shaped
            # question. Returning it beats discarding it and paying for a
            # whole extra _force_answer round trip.
            text = str(raw).strip()
            return text or None

    def _salvage(self, text: str, template: dict | None) -> Any:
        from .answer_format import enforce_single_json_object
        try:
            return json.loads(enforce_single_json_object(text))
        except (ValueError, json.JSONDecodeError):
            return None

    def _force_answer(self, messages: list[dict], template: dict | None,
                      logger: RunLogger) -> Any:
        """Last chance: one non-tool call asking for the JSON answer only."""
        prompt = (
            "STOP analysing. Time is up. Reply with ONLY the answer JSON object "
            "in exactly this shape, and nothing else:\n"
            + answer_schema_hint(template)
            + "\nUse your best supported estimate from everything computed so far."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.s.model,
                messages=messages + [{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            logger.event("forced_answer_raw", content=content)
            return self._salvage(content, template)
        except Exception as exc:  # noqa: BLE001
            logger.error("force_answer", f"{type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _skeleton(template: dict | None) -> Any:
        """Absolute last resort: an object with the right keys and null values.
        Wrong, but well-formed -- keeps the format contract intact."""
        from .answer_format import ANSWER_KEY, LOG_KEY
        if not template:
            return {}
        inner = template.get(ANSWER_KEY) if (ANSWER_KEY in template and LOG_KEY in template) else template
        if isinstance(inner, dict):
            return {k: None for k in inner}
        return inner
