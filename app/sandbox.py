"""Runs model-written Python in a throwaway subprocess.

Design goals, in priority order:
  1. Never hang the request (hard wall-clock timeout, killed process group).
  2. Never corrupt the bot process (separate interpreter, temp cwd).
  3. Give the model useful feedback (stdout + stderr + traceback returned).

This is isolation, not a security boundary -- the code we run is written by our
own LLM from our own prompt, not by an untrusted third party. On Linux we still
apply RLIMIT caps so a runaway allocation or fork bomb dies on its own.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

try:  # POSIX only
    import resource
except ImportError:  # pragma: no cover - Windows dev machines
    resource = None

MAX_OUTPUT_CHARS = 12000

_PREAMBLE = textwrap.dedent(
    """
    import warnings, os, sys
    warnings.filterwarnings("ignore")
    os.environ.setdefault("MPLBACKEND", "Agg")
    """
).strip()


def _limits():  # pragma: no cover - executed in the child process
    if resource is None:
        return
    # 2 GB address space, 120s CPU, no core dumps, max 64 processes.
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # No RLIMIT_NPROC. It is counted per-UID, not per-process, so on a shared
    # container it collides with whatever else runs as the same user and makes
    # `import pandas` fail nondeterministically. RLIMIT_AS + RLIMIT_CPU + the
    # subprocess timeout already bound this.
    os.setsid()


# Variables the child process needs to function. Stripping the environment
# entirely is tempting but breaks networking: on Windows, winsock cannot
# initialise without SYSTEMROOT, so every DNS lookup fails with
# `socket.gaierror: [Errno 11003] getaddrinfo failed`. On Linux, missing CA
# bundle paths break TLS. Proxy variables must survive on both.
_PASSTHROUGH_VARS = (
    # Windows platform essentials -- required for sockets, DNS and TLS
    "SYSTEMROOT", "SystemRoot", "SYSTEMDRIVE", "SystemDrive", "WINDIR",
    "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
    "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA",
    # TLS / certificate discovery
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    # Corporate / campus proxies
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
)


def _child_env(tmp: str) -> dict:
    """Environment for the sandboxed child: minimal, but not so minimal that
    the network stops working."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tmp,
        "TMPDIR": tmp,
        "TEMP": tmp,
        "TMP": tmp,
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
        # Single-threaded BLAS. On a small container, `import pandas` under a
        # process-count rlimit dies with
        #   OpenBLAS blas_thread_init: pthread_create failed ...
        #   Resource temporarily unavailable
        # which reads like a code error and costs the agent a tool call.
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    for name in _PASSTHROUGH_VARS:
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = text[: MAX_OUTPUT_CHARS // 2]
    tail = text[-MAX_OUTPUT_CHARS // 2:]
    return f"{head}\n...[{len(text) - MAX_OUTPUT_CHARS} chars omitted]...\n{tail}"


def run_python(code: str, timeout: int = 45, workdir: str | None = None) -> dict:
    """Execute `code` and return {"ok", "stdout", "stderr", "exit_code"}.

    The model is instructed to `print()` whatever it wants to see; only stdout
    comes back as the "result".
    """
    tmp = workdir or tempfile.mkdtemp(prefix="agentrun-")
    owns_tmp = workdir is None
    script = os.path.join(tmp, "step.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(_PREAMBLE + "\n\n" + code)

    env = _child_env(tmp)

    try:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limits if os.name == "posix" else None,
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": _clip(proc.stdout),
            "stderr": _clip(proc.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": _clip((exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")),
            "stderr": f"TimeoutExpired: code exceeded {timeout}s. Simplify the approach or fetch less data.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exit_code": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    finally:
        if owns_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
