"""Tests for DataProfiler."""
import numpy as np
import pandas as pd
import pytest
from datapure.core.profiler import DataProfiler, DataProfile


@pytest.fixture
def df():
    return pd.DataFrame({
        "age":    [25, np.nan, 35, np.nan, 45],
        "salary": [50000, 60000, np.nan, 80000, np.nan],
        "city":   ["Delhi", None, "Mumbai", "Delhi", None],
        "junk":   [np.nan] * 5,
    })


@pytest.fixture
def profile(df):
    return DataProfiler().run(df)


def test_profile_returns_dataprofile(profile):
    assert isinstance(profile, DataProfile)


def test_profile_shape(df, profile):
    assert profile.n_rows == len(df)
    assert profile.n_cols == len(df.columns)


def test_profile_all_columns_present(df, profile):
    assert set(profile.columns.keys()) == set(df.columns)


def test_null_count_correct(profile):
    assert profile.columns["age"].null_count == 2
    assert profile.columns["salary"].null_count == 2
    assert profile.columns["city"].null_count == 2
    assert profile.columns["junk"].null_count == 5


def test_null_pct_correct(profile):
    assert profile.columns["age"].null_pct == 40.0
    assert profile.columns["junk"].null_pct == 100.0


def test_numeric_stats_computed(profile):
    age = profile.columns["age"]
    assert age.mean is not None
    assert age.median is not None
    assert age.min_val == 25.0
    assert age.max_val == 45.0


def test_string_stats_computed(profile):
    city = profile.columns["city"]
    assert city.avg_str_len is not None


def test_fully_null_col_flagged(profile):
    issues = profile.columns["junk"].issues
    assert any("entirely null" in i for i in issues)


def test_high_null_col_flagged(profile):
    issues = profile.columns["age"].issues
    assert any("null" in i.lower() for i in issues)


def test_duplicate_rows_counted():
    df2 = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    profile = DataProfiler().run(df2)
    assert profile.duplicate_rows == 1


def test_get_issues_only_returns_problem_cols(profile):
    issues = profile.get_issues()
    assert "junk" in issues
    assert all(len(v) > 0 for v in issues.values())


def test_has_issues_true(profile):
    assert profile.has_issues() is True


def test_has_issues_false():
    df_clean = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    profile = DataProfiler().run(df_clean)
    assert profile.has_issues() is False


def test_memory_mb_positive(profile):
    assert profile.memory_mb > 0


def test_print_summary_no_crash(profile):
    # Should not raise
    DataProfiler().print_summary(profile)