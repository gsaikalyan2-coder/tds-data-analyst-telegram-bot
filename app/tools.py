"""Tool definitions exposed to the LLM, plus their dispatch."""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse, parse_qs

import requests

from .sandbox import run_python

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
# Rotated for the keyless backends -- a single fixed UA hitting a scraper
# endpoint three times in five seconds is what got us blocked.
_USER_AGENTS = [
    USER_AGENT,
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

MAX_FETCH_CHARS = 20000
MAX_SEARCH_RESULTS = 8

# --- self-throttle -------------------------------------------------------
# The agent can fire several searches back to back. Keyless endpoints treat
# that as abuse and start returning nothing, which is exactly how a run ends
# up unsourced. Enforce a floor on the gap between outbound searches.
_SEARCH_LOCK = threading.Lock()
_LAST_SEARCH_AT = 0.0
_MIN_SEARCH_INTERVAL = 1.5  # seconds; only applied to keyless backends

# --- tiny result cache ---------------------------------------------------
# Agents re-issue near-identical queries when a fetch disappoints them. Serving
# a repeat from cache costs nothing and avoids burning the rate limit.
_SEARCH_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 600.0
_NEGATIVE_CACHE_TTL = 45.0


def _throttle() -> None:
    global _LAST_SEARCH_AT
    with _SEARCH_LOCK:
        gap = time.time() - _LAST_SEARCH_AT
        if gap < _MIN_SEARCH_INTERVAL:
            time.sleep(_MIN_SEARCH_INTERVAL - gap)
        _LAST_SEARCH_AT = time.time()


def _headers() -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://duckduckgo.com/",
    }

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for pages, datasets and official statistics. "
                "Use this FIRST whenever the question names a dataset or a "
                "statistic you do not already have in hand. Never guess a URL "
                "-- search for it, then fetch_url the result you want. Returns "
                "a list of {title, url, snippet}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A specific search query built from words in the QUESTION: "
                            "the indicator as the question words it, the geography, "
                            "the period, and the publication name if known. Always "
                            "name the country."
                        ),
                    },
                    "max_results": {"type": "integer", "description": "Default 8."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python 3 in a fresh sandbox and return stdout/stderr. "
                "The sandbox HAS network access. pandas, numpy, scipy, "
                "statsmodels, scikit-learn, requests, beautifulsoup4, lxml, "
                "openpyxl, xlrd, pdfplumber and pyarrow are installed -- so you "
                "can download and parse .csv/.xls/.xlsx/.pdf/.json directly here. "
                "State does NOT persist between calls -- each call is a fresh "
                "process, so re-declare what you need. You MUST print() every "
                "value you want to see; nothing is returned implicitly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python source to run."},
                    "why": {"type": "string", "description": "One line: what this step establishes."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "HTTP GET a URL and return its text (HTML/CSV/JSON), truncated. "
                "Use for quickly inspecting a dataset page before writing parsing "
                "code. For anything you need to compute over, prefer downloading "
                "it inside run_python instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "description": "Default 20000."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf",
            "description": (
                "Download a PDF and search every page at once. ALWAYS use this "
                "instead of paging through a PDF with run_python -- one call "
                "replaces ten. Returns the page count, plus the full text of "
                "every page whose text matches `find` (case-insensitive), with "
                "page numbers and any tables detected on those pages. "
                "If `find` matches nothing, you get the document's headings "
                "back so you can pick a better term -- that is your signal the "
                "document may not contain the indicator at all, in which case "
                "go back to web_search rather than mining it anyway."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Direct URL to the .pdf"},
                    "find": {
                        "type": "string",
                        "description": (
                            "The exact indicator name to look for, taken from the "
                            "question. Use the full official wording, not an "
                            "acronym -- documents spell indicator names out."
                        ),
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Max matching pages to return. Default 6.",
                    },
                },
                "required": ["url", "find"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Submit the final answer. Call this exactly once, at the end. "
                "The `answer` argument must already be in the exact shape the "
                "question asked for -- no extra keys, no prose, no units unless "
                "the question asked for them. You must declare where the "
                "underlying numbers came from."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer_json": {
                        "type": "string",
                        "description": (
                            "The answer serialised as JSON. E.g. "
                            '{"state": "Assam"} or {"values": [1.02, 2.04]}.'
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "REQUIRED. Where the input numbers came from. Must be one of: "
                            "(a) the exact URL you fetched, "
                            "(b) the literal string 'inline data from the message' if the "
                            "question embedded the data, or "
                            "(c) the literal string 'NO SOURCE - could not retrieve data' "
                            "if every retrieval attempt failed. "
                            "Never cite a URL you did not successfully fetch, and never "
                            "cite a source for numbers you wrote yourself."
                        ),
                    },
                    "indicator": {
                        "type": "string",
                        "description": (
                            "REQUIRED when the answer came from a document. The exact "
                            "column/row header you read the numbers from, copied "
                            "verbatim from the table, including any unit shown. "
                            "If the header you actually read does not name the "
                            "quantity the question asked for, you have the wrong column: "
                            "go back and find the right one instead of submitting. "
                            "Use 'inline' for data embedded in the message."
                        ),
                    },
                    "reference_period": {
                        "type": "string",
                        "description": (
                            "REQUIRED when the answer came from a document. The year or "
                            "period the figures cover, as printed in the source, e.g. "
                            "'2019-21'. Use 'n/a' for inline data."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Two sentences on how the answer was derived. For the log only.",
                    },
                },
                "required": ["answer_json", "source"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# web_search: provider-agnostic. Tries the best backend that has credentials,
# falls back to a keyless one so the bot still works with zero configuration.
# ---------------------------------------------------------------------------
def _search_tavily(query: str, n: int) -> list[dict] | None:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return None
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": n,
                  "search_depth": "basic", "include_answer": False},
            timeout=25,
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": (r.get("content") or "")[:400]}
            for r in resp.json().get("results", [])
        ][:n]
    except Exception:  # noqa: BLE001
        return None


def _search_brave(query: str, n: int) -> list[dict] | None:
    key = os.getenv("BRAVE_API_KEY", "").strip()
    if not key:
        return None
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": n},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=25,
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": re.sub(r"<[^>]+>", "", r.get("description", ""))[:400]}
            for r in resp.json().get("web", {}).get("results", [])
        ][:n]
    except Exception:  # noqa: BLE001
        return None


def _search_duckduckgo(query: str, n: int) -> list[dict] | None:
    """Keyless fallback. Scrapes the DuckDuckGo lite endpoint. Flakier than a
    real API, but needs no signup, so the bot is never search-blind."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    # (endpoint, http method). POST is the documented form for the lite/html
    # endpoints, but when DDG starts throttling POSTs the GET form often still
    # answers, so both are attempted before giving up on a host.
    attempts = [
        ("https://lite.duckduckgo.com/lite/", "post"),
        ("https://html.duckduckgo.com/html/", "post"),
        ("https://lite.duckduckgo.com/lite/", "get"),
        ("https://html.duckduckgo.com/html/", "get"),
    ]
    for index, (endpoint, method) in enumerate(attempts):
        try:
            _throttle()
            if index:
                # Exponential-ish backoff with jitter between retries. Cheap
                # insurance: a blocked scraper usually unblocks within seconds.
                time.sleep(min(2 ** index * 0.4, 4.0) + random.uniform(0, 0.6))
            headers = _headers()
            if method == "post":
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                resp = requests.post(endpoint, data={"q": query},
                                     headers=headers, timeout=25)
            else:
                resp = requests.get(endpoint, params={"q": query},
                                    headers=headers, timeout=25)
            if not resp.ok:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            out: list[dict] = []
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                if "uddg=" in href:  # DDG redirect wrapper
                    qs = parse_qs(urlparse(href).query)
                    href = unquote(qs.get("uddg", [""])[0])
                if not href.startswith("http") or "duckduckgo.com" in href:
                    continue
                title = a.get_text(" ", strip=True)
                if not title or len(title) < 3:
                    continue
                if any(r["url"] == href for r in out):
                    continue

                # Pull the snippet that follows the result link. Without it the
                # model is choosing between bare URLs and re-searches blindly.
                snippet = ""
                for sel in ("td.result-snippet", ".result__snippet"):
                    node = a.find_parent("tr")
                    node = node.find_next_sibling("tr") if node else None
                    found = node.select_one(sel) if node else None
                    if found:
                        snippet = found.get_text(" ", strip=True)
                        break
                if not snippet:
                    parent = a.find_parent(["div", "td"])
                    sib = parent.find_next(class_=re.compile("snippet")) if parent else None
                    if sib:
                        snippet = sib.get_text(" ", strip=True)

                out.append({"title": title[:200], "url": href,
                            "snippet": snippet[:400]})
                if len(out) >= n:
                    break
            if out:
                return out
        except Exception:  # noqa: BLE001
            continue
    return None


def _search_mojeek(query: str, n: int) -> list[dict] | None:
    """Second keyless backend. Mojeek runs its own index and is markedly more
    tolerant of automated requests than DuckDuckGo, so it usually answers when
    DDG has started stonewalling us."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    try:
        _throttle()
        resp = requests.get("https://www.mojeek.com/search",
                            params={"q": query}, headers=_headers(), timeout=25)
        if not resp.ok:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        out: list[dict] = []
        for li in soup.select("ul.results-standard li, li.result"):
            link = li.select_one("a.title") or li.select_one("h2 a") or li.select_one("a[href^=http]")
            if not link:
                continue
            href = link.get("href", "")
            if not href.startswith("http"):
                continue
            desc = li.select_one("p.s") or li.select_one("p")
            out.append({
                "title": link.get_text(" ", strip=True)[:200],
                "url": href,
                "snippet": desc.get_text(" ", strip=True)[:400] if desc else "",
            })
            if len(out) >= n:
                break
        return out or None
    except Exception:  # noqa: BLE001
        return None


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> dict:
    n = max(1, min(int(max_results or MAX_SEARCH_RESULTS), 15))
    query = (query or "").strip()
    if not query:
        return {"ok": False, "query": query, "results": [], "error": "empty query"}

    cache_key = f"{query.lower()}|{n}"
    hit = _SEARCH_CACHE.get(cache_key)
    if hit:
        age = time.time() - hit[0]
        # Successes are cached for the full TTL; failures for much less, so a
        # transient block doesn't poison the run, but the agent also can't burn
        # 8s of its deadline re-running the same doomed query three times.
        ttl = _CACHE_TTL if hit[1].get("ok") else _NEGATIVE_CACHE_TTL
        if age < ttl:
            return {**hit[1], "cached": True}

    tried: list[str] = []
    for provider, fn in (("tavily", _search_tavily),
                         ("brave", _search_brave),
                         ("duckduckgo", _search_duckduckgo),
                         ("mojeek", _search_mojeek)):
        results = fn(query, n)
        tried.append(provider)
        if results:
            payload = {"ok": True, "provider": provider, "query": query,
                       "count": len(results), "results": results}
            _SEARCH_CACHE[cache_key] = (time.time(), payload)
            return payload

    failure = {
        "ok": False,
        "query": query,
        "results": [],
        "providers_tried": tried,
        "error": (
            "All search providers failed or returned nothing. Try ONE shorter, "
            "differently-worded query (drop acronyms, use plain words). If that "
            "also fails, try fetch_url directly on the official statistical "
            "agency domain for the country in the question. Do NOT invent "
            "data -- if you truly "
            "cannot retrieve a source, answer from general knowledge and set "
            "source to 'NO SOURCE - could not retrieve data'."
        ),
    }
    _SEARCH_CACHE[cache_key] = (time.time(), failure)
    return failure


MAX_PDF_BYTES = 40 * 1024 * 1024
_PDF_CACHE: dict[str, tuple[float, bytes]] = {}


def _get_with_ssl_fallback(url: str, **kwargs):
    """GET, retrying once without certificate verification on a cert error.

    Several Indian government statistics hosts (censusindia.gov.in among them)
    serve an incomplete certificate chain, so requests raises
    SSLCertVerificationError: unable to get local issuer certificate. That
    killed two tool calls on the primary source in a live run.

    We are downloading public statistics, not sending credentials, so falling
    back to an unverified fetch trades a MITM risk we do not care about for
    access to the authoritative document. The fallback is logged in the result
    so it is visible in the run log rather than silent.
    """
    try:
        return requests.get(url, **kwargs), False
    except requests.exceptions.SSLError:
        insecure = dict(kwargs)
        insecure["verify"] = False
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass
        return requests.get(url, **insecure), True


def _pdf_bytes(url: str) -> bytes:
    """Download once, reuse across calls in the same run. Government PDFs are
    often 5-20 MB and re-downloading them per page walk is what makes PDF
    questions blow the turn deadline."""
    hit = _PDF_CACHE.get(url)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    resp, _insecure = _get_with_ssl_fallback(
        url, timeout=60, headers={"User-Agent": USER_AGENT},
        allow_redirects=True, stream=True)
    resp.raise_for_status()
    data = resp.raw.read(MAX_PDF_BYTES + 1, decode_content=True) or resp.content
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"PDF larger than {MAX_PDF_BYTES // 1024 // 1024} MB; refuse to load")
    _PDF_CACHE[url] = (time.time(), data)
    return data


def read_pdf(url: str, find: str, max_pages: int = 6) -> dict:
    """Search a whole PDF in one tool call.

    Replaces the page-by-page probing pattern that otherwise eats the entire
    tool-call budget: `pdf.pages[20]` -> IndexError -> `len(pdf.pages)` ->
    `pages[2]` -> `pages[3]` -> ... Each of those was a full round trip.
    """
    try:
        import pdfplumber
    except ImportError:
        return {"ok": False, "error": "pdfplumber is not installed in the sandbox."}

    from io import BytesIO
    try:
        data = _pdf_bytes(url)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "error": f"download failed: {type(exc).__name__}: {exc}"}

    needle = (find or "").strip().lower()
    limit = max(1, min(int(max_pages or 6), 12))
    matches: list[dict] = []
    headings: list[str] = []

    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:  # noqa: BLE001
                    continue
                # Collect short lines as candidate headings for the miss case.
                if len(headings) < 60:
                    for line in text.splitlines():
                        s = line.strip()
                        if 6 < len(s) < 90 and not s[0].isdigit():
                            headings.append(f"p{i}: {s}")
                            break
                if needle and needle in text.lower() and len(matches) < limit:
                    entry = {"page": i, "text": text[:6000]}
                    try:
                        tables = page.extract_tables() or []
                        if tables:
                            # First table only; enough to expose the header row,
                            # which is what the model must verify before it
                            # extracts any numbers.
                            entry["first_table_rows"] = [
                                [("" if c is None else str(c))[:60] for c in row]
                                for row in tables[0][:25]
                            ]
                            entry["table_count"] = len(tables)
                    except Exception:  # noqa: BLE001
                        pass
                    matches.append(entry)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "error": f"parse failed: {type(exc).__name__}: {exc}"}

    if not matches:
        return {
            "ok": True,
            "url": url,
            "pages": total,
            "find": find,
            "matched_pages": 0,
            "headings_sample": headings[:40],
            "note": (
                f"'{find}' does not appear anywhere in this {total}-page PDF. "
                "This document probably does NOT contain the indicator you need. "
                "Do not mine it for adjacent-looking numbers. Either retry "
                "read_pdf with the indicator's exact official wording, or go "
                "back to web_search for the correct publication."
            ),
        }

    return {
        "ok": True,
        "url": url,
        "pages": total,
        "find": find,
        "matched_pages": len(matches),
        "results": matches,
        "note": (
            "Before extracting numbers: confirm the column header actually says "
            f"'{find}', confirm the reference year, and confirm the row labels "
            "are the geography you were asked about. Neighbouring columns in "
            "these tables are different indicators entirely."
        ),
    }


_BINARY_HINTS = (
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats",
    "application/pdf",
    "application/zip",
    "application/octet-stream",
    "application/x-parquet",
    "image/",
)


def fetch_url(url: str, max_chars: int = MAX_FETCH_CHARS) -> dict:
    try:
        resp, insecure = _get_with_ssl_fallback(
            url,
            timeout=30,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            allow_redirects=True,
        )
        ctype = resp.headers.get("Content-Type", "")
        # Recorded in the payload so the run log shows when a certificate chain
        # was not verified, rather than that being invisible.
        ssl_note = {"tls_verified": not insecure} if insecure else {}

        # Never inline binary payloads. Decoding a .xls/.pdf as text produces
        # pages of mojibake that burn the context window and tell the model
        # nothing. Point it at run_python instead, which can parse the bytes.
        if any(h in ctype.lower() for h in _BINARY_HINTS):
            return {
                "ok": True,
                "status_code": resp.status_code,
                "content_type": ctype,
                "bytes": len(resp.content),
                "binary": True,
                "text": None,
                "note": (
                    f"This is a binary file ({ctype}, {len(resp.content)} bytes), not text. "
                    "Do NOT try to read it here. Download and parse it inside run_python, "
                    "e.g.:\n"
                    "  import requests, pandas as pd, io\n"
                    f"  r = requests.get({url!r}, timeout=60, headers={{'User-Agent': 'Mozilla/5.0'}})\n"
                    "  df = pd.read_excel(io.BytesIO(r.content))   # or pd.read_csv / pdfplumber\n"
                    "  print(df.head(20)); print(df.columns.tolist())"
                ),
                **ssl_note,
            }

        text = resp.text or ""
        return {
            "ok": resp.ok,
            "status_code": resp.status_code,
            "content_type": ctype,
            "length": len(text),
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
            **ssl_note,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def dispatch(name: str, args: dict[str, Any], sandbox_timeout: int) -> dict:
    """Run a tool call and return a JSON-serialisable result."""
    if name == "run_python":
        return run_python(args.get("code", ""), timeout=sandbox_timeout)
    if name == "fetch_url":
        return fetch_url(args.get("url", ""), int(args.get("max_chars") or MAX_FETCH_CHARS))
    if name == "web_search":
        return web_search(args.get("query", ""), int(args.get("max_results") or MAX_SEARCH_RESULTS))
    if name == "read_pdf":
        return read_pdf(args.get("url", ""), args.get("find", ""),
                        int(args.get("max_pages") or 6))
    return {"ok": False, "error": f"unknown tool: {name}"}


def parse_tool_args(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_unparsed": raw}
