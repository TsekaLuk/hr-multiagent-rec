"""Test the JSON-extraction helper used by chat_json. No network, no mock LLM."""
from __future__ import annotations

import pytest

from hr_rec.agents.llm import _safe_parse_json

pytestmark = pytest.mark.unit


def test_plain_json() -> None:
    assert _safe_parse_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_markdown_fenced_json() -> None:
    s = '```json\n{"core": ["Python"]}\n```'
    assert _safe_parse_json(s) == {"core": ["Python"]}


def test_fenced_without_lang() -> None:
    s = '```\n{"a": 1}\n```'
    assert _safe_parse_json(s) == {"a": 1}


def test_with_leading_prose() -> None:
    s = 'Sure, here you go:\n{"k": 42}\nThanks!'
    assert _safe_parse_json(s) == {"k": 42}


def test_nested_json_extracted() -> None:
    s = 'noise {"outer": {"inner": [1, 2]}} more noise'
    assert _safe_parse_json(s) == {"outer": {"inner": [1, 2]}}


def test_invalid_returns_none() -> None:
    assert _safe_parse_json("nope, just prose") is None


def test_non_dict_returns_none() -> None:
    """A JSON array is valid JSON but not a dict — we reject it."""
    assert _safe_parse_json("[1, 2, 3]") is None


def test_empty_returns_none() -> None:
    assert _safe_parse_json("") is None


def test_unicode_preserved() -> None:
    s = '{"职责": "后端开发", "薪资": "15-25K"}'
    obj = _safe_parse_json(s)
    assert obj is not None
    assert obj["职责"] == "后端开发"
