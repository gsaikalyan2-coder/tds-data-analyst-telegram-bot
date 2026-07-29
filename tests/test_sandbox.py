import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.sandbox import run_python


def test_basic_stdout():
    r = run_python("print(sum([1,2,3]))", timeout=20)
    assert r["ok"] and r["stdout"].strip() == "6"


def test_error_is_captured_not_raised():
    r = run_python("1/0", timeout=20)
    assert not r["ok"] and "ZeroDivisionError" in r["stderr"]


def test_timeout_is_bounded():
    r = run_python("import time; time.sleep(30)", timeout=3)
    assert not r["ok"] and "Timeout" in r["stderr"]
