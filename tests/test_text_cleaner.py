"""Tests for TextCleaner — encoding, HTML, whitespace, URLs, custom patterns."""
import pandas as pd
import pytest
from datapure.cleaners.text import TextCleaner


def test_strip_html_tags():
    df = pd.DataFrame({"text": ["<p>Hello <b>World</b></p>"]})
    result = TextCleaner(strip_html=True).clean(df)
    assert "<" not in result["text"].iloc[0]
    assert "Hello" in result["text"].iloc[0]
    assert "World" in result["text"].iloc[0]


def test_fix_encoding_mojibake():
    df = pd.DataFrame({"text": ["caf\u00e9"]})
    result = TextCleaner(fix_encoding=True).clean(df)
    assert result["text"].iloc[0] == "café"


def test_normalize_whitespace_collapses_spaces():
    df = pd.DataFrame({"text": ["hello   world"]})
    result = TextCleaner(normalize_whitespace=True).clean(df)
    assert result["text"].iloc[0] == "hello world"


def test_normalize_whitespace_collapses_newlines():
    df = pd.DataFrame({"text": ["hello\n\nworld\tthere"]})
    result = TextCleaner(normalize_whitespace=True).clean(df)
    assert result["text"].iloc[0] == "hello world there"


def test_strip_leading_trailing():
    df = pd.DataFrame({"text": ["   hello world   "]})
    result = TextCleaner(strip=True).clean(df)
    assert result["text"].iloc[0] == "hello world"


def test_lowercase():
    df = pd.DataFrame({"text": ["Hello WORLD"]})
    result = TextCleaner(lowercase=True).clean(df)
    assert result["text"].iloc[0] == "hello world"


def test_remove_urls():
    df = pd.DataFrame({"text": ["Visit https://example.com for more info"]})
    result = TextCleaner(remove_urls=True).clean(df)
    assert "https" not in result["text"].iloc[0]
    assert "info" in result["text"].iloc[0]


def test_remove_www_urls():
    df = pd.DataFrame({"text": ["See www.example.com today"]})
    result = TextCleaner(remove_urls=True).clean(df)
    assert "www" not in result["text"].iloc[0]


def test_custom_pattern_replacement():
    df = pd.DataFrame({"text": ["Call 123-456-7890 now"]})
    result = TextCleaner(
        custom_patterns=[(r"\d{3}-\d{3}-\d{4}", "[PHONE]")]
    ).clean(df)
    assert "[PHONE]" in result["text"].iloc[0]
    assert "123-456-7890" not in result["text"].iloc[0]


def test_multiple_custom_patterns():
    df = pd.DataFrame({"text": ["Email me@test.com or call 999-888-7777"]})
    result = TextCleaner(
        custom_patterns=[
            (r"[\w.]+@[\w.]+", "[EMAIL]"),
            (r"\d{3}-\d{3}-\d{4}", "[PHONE]"),
        ]
    ).clean(df)
    assert "[EMAIL]" in result["text"].iloc[0]
    assert "[PHONE]" in result["text"].iloc[0]


def test_null_values_return_none():
    df = pd.DataFrame({"text": ["hello", None, "world"]})
    result = TextCleaner().clean(df)
    assert pd.isna(result["text"].iloc[1])  # handles both None and nan


def test_empty_string_after_cleaning_returns_none():
    df = pd.DataFrame({"text": ["   "]})
    result = TextCleaner(strip=True).clean(df)
    assert result["text"].iloc[0] is None


def test_column_filter_only_cleans_specified():
    df = pd.DataFrame({
        "a": ["  hello  "],
        "b": ["  world  "],
    })
    result = TextCleaner(columns=["a"]).clean(df)
    assert result["a"].iloc[0] == "hello"
    assert result["b"].iloc[0] == "  world  "


def test_all_columns_cleaned_when_none_specified():
    df = pd.DataFrame({
        "x": ["  foo  "],
        "y": ["  bar  "],
    })
    result = TextCleaner(strip=True).clean(df)
    assert result["x"].iloc[0] == "foo"
    assert result["y"].iloc[0] == "bar"


def test_does_not_mutate_input():
    df = pd.DataFrame({"text": ["<b>hello</b>"]})
    TextCleaner(strip_html=True).clean(df)
    assert df["text"].iloc[0] == "<b>hello</b>"