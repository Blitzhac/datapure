"""Tests for MissingValueCleaner — all 9 strategies."""
import numpy as np
import pandas as pd
import pytest
from datapure.cleaners.missing import MissingValueCleaner


@pytest.fixture
def df():
    return pd.DataFrame({
        "age":    [25, np.nan, 35, np.nan, 45],
        "salary": [50000, 60000, np.nan, 80000, np.nan],
        "city":   ["Delhi", None, "Mumbai", "Delhi", None],
        "score":  [np.nan] * 5,
    })


def test_drop_rows(df):
    result = MissingValueCleaner(strategy="drop_rows").clean(df)
    assert result.isnull().sum().sum() == 0
    assert len(result) < len(df)


def test_drop_cols_removes_fully_null_col(df):
    result = MissingValueCleaner(strategy="drop_cols", threshold=0.5).clean(df)
    assert "score" not in result.columns


def test_drop_cols_keeps_partial_null_col(df):
    result = MissingValueCleaner(strategy="drop_cols", threshold=0.9).clean(df)
    assert "age" in result.columns


def test_mean_fills_numeric(df):
    result = MissingValueCleaner(strategy="mean").clean(df)
    assert result["age"].isnull().sum() == 0
    assert result["salary"].isnull().sum() == 0


def test_mean_skips_non_numeric(df):
    result = MissingValueCleaner(strategy="mean").clean(df)
    # city is non-numeric — mean can't fill it, nulls remain
    assert result["city"].isnull().sum() > 0


def test_median_robust_to_outliers():
    df2 = pd.DataFrame({"x": [1, 2, np.nan, 100]})
    result = MissingValueCleaner(strategy="median").clean(df2)
    # median of [1,2,100] = 2.0, not mean=34.3
    assert result["x"].iloc[2] == 2.0


def test_mode_fills_categorical(df):
    result = MissingValueCleaner(strategy="mode").clean(df)
    assert result["city"].isnull().sum() == 0
    assert result["city"].iloc[1] == "Delhi"


def test_ffill_propagates_forward():
    df2 = pd.DataFrame({"t": [20.0, np.nan, np.nan, 22.0]})
    result = MissingValueCleaner(strategy="ffill").clean(df2)
    assert result["t"].tolist() == [20.0, 20.0, 20.0, 22.0]


def test_bfill_propagates_backward():
    df2 = pd.DataFrame({"t": [np.nan, np.nan, 22.0, 25.0]})
    result = MissingValueCleaner(strategy="bfill").clean(df2)
    assert result["t"].tolist() == [22.0, 22.0, 22.0, 25.0]


def test_constant_fill_numeric(df):
    result = MissingValueCleaner(strategy="constant", fill_value=-1).clean(df)
    assert (result["age"] == -1).any()


def test_constant_fill_string():
    df2 = pd.DataFrame({"city": ["Delhi", None, "Mumbai"]})
    result = MissingValueCleaner(
        strategy="constant", fill_value="Unknown"
    ).clean(df2)
    assert result["city"].iloc[1] == "Unknown"


def test_knn_imputation():
    df2 = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": [10.0, 20.0, 30.0, 40.0],
    })
    result = MissingValueCleaner(strategy="knn", knn_k=2).clean(df2)
    assert result["a"].isnull().sum() == 0
    assert 2.0 <= result["a"].iloc[2] <= 4.0


def test_per_column_strategy_override():
    df2 = pd.DataFrame({
        "age":  [25, np.nan, 35],
        "city": ["A", None, "A"],
    })
    result = MissingValueCleaner(
        strategy="mean",
        col_strategy={"city": "mode"},
    ).clean(df2)
    assert result.isnull().sum().sum() == 0


def test_does_not_mutate_input(df):
    nulls_before = df.isnull().sum().sum()
    MissingValueCleaner(strategy="mean").clean(df)
    assert df.isnull().sum().sum() == nulls_before


def test_repr_contains_strategy():
    c = MissingValueCleaner(strategy="knn", knn_k=3)
    assert "knn" in repr(c)
    assert "3" in repr(c)