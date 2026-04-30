"""
Tests for AISuggester — mocks the API call so no real key needed.
Tests JSON parsing, CleaningPlan building, pipeline construction.
"""
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from datapure.ai.suggester import (
    AISuggester,
    CleaningPlan,
    CleaningSuggestion,
)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def df():
    return pd.DataFrame({
        "age":    [25, np.nan, 35, np.nan, 45],
        "salary": [50000.0, 60000.0, np.nan, 80000.0, 1000000.0],
        "city":   ["Delhi", None, "Mumbai", "Delhi", None],
    })


MOCK_API_RESPONSE = json.dumps({
    "summary": "Dataset has missing values in age and city, and an outlier in salary.",
    "overall_quality": "fair",
    "suggestions": [
        {
            "column": "age",
            "issue": "40% missing values",
            "cleaner": "MissingValueCleaner",
            "strategy": "median",
            "confidence": 0.95,
            "reason": "Median is robust to the salary outlier.",
            "priority": "high",
        },
        {
            "column": "salary",
            "issue": "Outlier at 1000000",
            "cleaner": "OutlierCleaner",
            "strategy": "iqr",
            "confidence": 0.88,
            "reason": "IQR method will cap the extreme value.",
            "priority": "high",
        },
        {
            "column": "city",
            "issue": "40% missing values, categorical",
            "cleaner": "MissingValueCleaner",
            "strategy": "mode",
            "confidence": 0.80,
            "reason": "Mode fills with most frequent city.",
            "priority": "medium",
        },
        {
            "column": "__all__",
            "issue": "Duplicate rows possible",
            "cleaner": "DuplicateCleaner",
            "strategy": "exact",
            "confidence": 0.70,
            "reason": "Remove exact duplicates as a precaution.",
            "priority": "low",
        },
    ],
    "estimated_rows_at_risk": 4,
    "recommended_pipeline_order": [
        "MissingValueCleaner",
        "OutlierCleaner",
        "DuplicateCleaner",
    ],
})


@pytest.fixture
def mock_suggester(monkeypatch):
    """AISuggester with API call mocked out."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
    suggester = AISuggester()
    suggester._call_api = MagicMock(return_value=MOCK_API_RESPONSE)
    return suggester


# ── Tests: CleaningSuggestion ─────────────────────────────────────

def test_suggestion_str():
    s = CleaningSuggestion(
        column="age", issue="40% nulls",
        cleaner="MissingValueCleaner", strategy="median",
        confidence=0.95, reason="Safe default.", priority="high",
    )
    assert "MissingValueCleaner" in str(s)
    assert "age" in str(s)


# ── Tests: CleaningPlan ───────────────────────────────────────────

@pytest.fixture
def plan(mock_suggester, df):
    return mock_suggester.suggest(df)


def test_suggest_returns_cleaning_plan(plan):
    assert isinstance(plan, CleaningPlan)


def test_plan_has_summary(plan):
    assert len(plan.summary) > 0


def test_plan_has_quality(plan):
    assert plan.overall_quality in ("poor", "fair", "good", "excellent")


def test_plan_has_suggestions(plan):
    assert len(plan.suggestions) == 4


def test_plan_suggestions_are_correct_type(plan):
    for s in plan.suggestions:
        assert isinstance(s, CleaningSuggestion)


def test_plan_high_priority_filter(plan):
    high = plan.high_priority()
    assert all(s.priority == "high" for s in high)
    assert len(high) == 2


def test_plan_for_column_filter(plan):
    age_suggestions = plan.for_column("age")
    assert all(s.column == "age" for s in age_suggestions)
    assert len(age_suggestions) == 1


def test_plan_estimated_rows_at_risk(plan):
    assert plan.estimated_rows_at_risk == 4


def test_plan_recommended_order(plan):
    assert "MissingValueCleaner" in plan.recommended_pipeline_order


def test_plan_print_no_crash(plan, capsys):
    plan.print_plan()
    captured = capsys.readouterr()
    assert "AI CLEANING PLAN" in captured.out


# ── Tests: build_pipeline ─────────────────────────────────────────

def test_build_pipeline_returns_pipeline(mock_suggester, plan):
    from datapure.core.pipeline import Pipeline
    pipeline = mock_suggester.build_pipeline(plan, min_confidence=0.5)
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_has_cleaners(mock_suggester, plan):
    pipeline = mock_suggester.build_pipeline(plan, min_confidence=0.5)
    assert len(pipeline._cleaners) > 0


def test_build_pipeline_respects_min_confidence(mock_suggester, plan):
    # Only suggestions with confidence >= 0.9 (just age's MissingValueCleaner)
    pipeline = mock_suggester.build_pipeline(plan, min_confidence=0.90)
    assert len(pipeline._cleaners) == 1


def test_build_pipeline_no_duplicate_cleaners(mock_suggester, plan):
    pipeline = mock_suggester.build_pipeline(plan, min_confidence=0.5)
    cleaner_types = [type(c).__name__ for c in pipeline._cleaners]
    # No cleaner type should appear twice
    assert len(cleaner_types) == len(set(cleaner_types))


def test_pipeline_runs_on_df(mock_suggester, plan, df):
    pipeline = mock_suggester.build_pipeline(plan, min_confidence=0.5)
    result = pipeline.run(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


# ── Tests: error handling ─────────────────────────────────────────

def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        AISuggester().suggest(pd.DataFrame({"a": [1]}))


def test_invalid_api_key_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-valid-key")
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        AISuggester().suggest(pd.DataFrame({"a": [1]}))


def test_invalid_json_response_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
    suggester = AISuggester()
    suggester._call_api = MagicMock(return_value="not valid json {{{")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        suggester.suggest(pd.DataFrame({"a": [1]}))


def test_markdown_fences_stripped(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
    suggester = AISuggester()
    fenced = f"```json\n{MOCK_API_RESPONSE}\n```"
    suggester._call_api = MagicMock(return_value=fenced)
    plan = suggester.suggest(pd.DataFrame({"a": [1]}))
    assert isinstance(plan, CleaningPlan)


def test_parse_empty_suggestions(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
    empty_response = json.dumps({
        "summary": "Data looks clean.",
        "overall_quality": "excellent",
        "suggestions": [],
        "estimated_rows_at_risk": 0,
        "recommended_pipeline_order": [],
    })
    suggester = AISuggester()
    suggester._call_api = MagicMock(return_value=empty_response)
    plan = suggester.suggest(pd.DataFrame({"a": [1, 2, 3]}))
    assert plan.overall_quality == "excellent"
    assert len(plan.suggestions) == 0