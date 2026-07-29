"""The contract tests. If these pass, the format marks are safe."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.answer_format import (build_reply, coerce_answer,
                               enforce_single_json_object,
                               extract_response_template, wants_wrapper)

WRAPPED = (
    'Which state has the highest maternal mortality rate based on MOSPI data? '
    'Reply with ONLY this JSON object and nothing else: '
    '{"answer": {"state": "<state name>"}, "log_url": "<public wget-able URL to your agent\'s JSONL log>"}'
)
BARE = (
    'Which state has the highest maternal mortality rate based on MOSPI data? '
    'Reply with ONLY a JSON object like {"state": "<state name>"}'
)
LIST_SHAPE = (
    'Forecast flow rate for these inputs: [12, 40, 87]. '
    'Reply with ONLY {"values": [<numbers>]}.'
)
NO_TEMPLATE = "Build a model to forecast monthly rainfall from the data above."


def test_extracts_wrapped_template():
    t = extract_response_template(WRAPPED)
    assert t is not None and "answer" in t and "log_url" in t
    # placeholders are preserved when the template is already valid JSON
    assert list(t["answer"].keys()) == ["state"]
    assert wants_wrapper(t) is True


def test_extracts_bare_template():
    t = extract_response_template(BARE)
    assert list(t.keys()) == ["state"]
    assert wants_wrapper(t) is False


def test_extracts_unquoted_placeholder_list():
    t = extract_response_template(LIST_SHAPE)
    assert list(t.keys()) == ["values"] and isinstance(t["values"], list)


def test_no_template_defaults_to_wrapper():
    assert extract_response_template(NO_TEMPLATE) is None
    assert wants_wrapper(None) is True


def test_build_reply_wrapped_is_exactly_two_keys():
    t = extract_response_template(WRAPPED)
    out = build_reply({"state": "Assam"}, "https://h/run.jsonl", t)
    parsed = json.loads(out)
    assert set(parsed) == {"answer", "log_url"}
    assert parsed["answer"] == {"state": "Assam"}
    assert out.strip() == out and out.startswith("{") and out.endswith("}")


def test_build_reply_bare_mirrors_shape():
    t = extract_response_template(BARE)
    out = build_reply({"state": "Assam"}, "https://h/run.jsonl", t)
    assert json.loads(out) == {"state": "Assam"}


def test_coerce_unwraps_double_wrapped_model_output():
    t = extract_response_template(WRAPPED)
    raw = {"answer": {"state": "Assam"}, "log_url": "http://model-made-this"}
    assert coerce_answer(raw, t) == {"state": "Assam"}


def test_coerce_parses_json_string():
    t = extract_response_template(BARE)
    assert coerce_answer('{"state": "Assam"}', t) == {"state": "Assam"}


def test_coerce_unwraps_one_level_too_deep():
    t = extract_response_template(WRAPPED)
    assert coerce_answer({"answer": {"state": "Assam"}}, t) == {"state": "Assam"}


def test_enforce_strips_markdown_fence():
    text = '```json\n{"state": "Assam"}\n```'
    assert json.loads(enforce_single_json_object(text)) == {"state": "Assam"}


def test_enforce_strips_surrounding_prose():
    text = 'Sure! Here is the answer:\n{"state": "Assam"}\nHope that helps.'
    assert json.loads(enforce_single_json_object(text)) == {"state": "Assam"}


def test_enforce_handles_nested_objects():
    text = 'noise {"answer": {"a": {"b": 1}}, "log_url": "x"} noise'
    assert json.loads(enforce_single_json_object(text)) == {
        "answer": {"a": {"b": 1}}, "log_url": "x"}


def test_reply_is_single_line_no_extras():
    t = extract_response_template(WRAPPED)
    out = build_reply({"state": "Assam"}, "https://h/run.jsonl", t)
    assert "\n" not in out
    assert not out.startswith("`")
