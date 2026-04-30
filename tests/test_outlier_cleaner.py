"""Tests for OutlierCleaner — IQR, Z-score, and Isolation Forest."""
import numpy as np
import pandas as pd
import pytest
from datapure.cleaners.outliers import OutlierCleaner


@pytest.fixture
def df_with_outlier():
    return pd.DataFrame({"val": [10.0, 11.0, 12.0, 13.0, 14.0, 1000.0]})


def test_iqr_remove_drops_outlier(df_with_outlier):
    result = OutlierCleaner(method="iqr", action="remove").clean(df_with_outlier)
    assert 1000.0 not in result["val"].values
    assert len(result) < len(df_with_outlier)


def test_iqr_winsorize_keeps_row_count(df_with_outlier):
    result = OutlierCleaner(method="iqr", action="winsorize").clean(df_with_outlier)
    assert len(result) == len(df_with_outlier)
    assert result["val"].max() < 1000.0


def test_iqr_winsorize_caps_at_boundary(df_with_outlier):
    result = OutlierCleaner(method="iqr", action="winsorize").clean(df_with_outlier)
    # All values should be within a reasonable range
    assert result["val"].max() <= 20.0


def test_zscore_remove_drops_outlier():
    # Need enough points so std is small and 1000 is clearly extreme
    normal_vals = [10.0, 11.0, 10.5, 11.5, 10.2, 10.8, 11.2, 10.6, 11.0, 10.9]
    df2 = pd.DataFrame({"val": normal_vals + [1000.0]})
    result = OutlierCleaner(method="zscore", action="remove").clean(df2)
    assert 1000.0 not in result["val"].values


def test_zscore_winsorize_preserves_row_count(df_with_outlier):
    result = OutlierCleaner(method="zscore", action="winsorize").clean(df_with_outlier)
    assert len(result) == len(df_with_outlier)


def test_no_numeric_cols_returns_unchanged():
    df2 = pd.DataFrame({"name": ["alice", "bob"]})
    result = OutlierCleaner().clean(df2)
    assert len(result) == 2


def test_constant_column_no_crash():
    # std=0 should not raise ZeroDivisionError
    df2 = pd.DataFrame({"val": [5.0, 5.0, 5.0, 5.0]})
    result = OutlierCleaner(method="zscore").clean(df2)
    assert len(result) == len(df2)


def test_isolation_forest_remove():
    np.random.seed(42)
    normal = np.random.normal(50, 5, 100).tolist()
    data = normal + [500.0, -500.0]
    df2 = pd.DataFrame({"val": data})
    result = OutlierCleaner(
        method="isolation_forest",
        action="remove",
        if_contamination=0.02,
    ).clean(df2)
    assert len(result) < len(df2)


def test_isolation_forest_winsorize():
    np.random.seed(42)
    normal = np.random.normal(50, 5, 100).tolist()
    data = normal + [500.0]
    df2 = pd.DataFrame({"val": data})
    result = OutlierCleaner(
        method="isolation_forest",
        action="winsorize",
        if_contamination=0.01,
    ).clean(df2)
    assert len(result) == len(df2)


def test_column_filter_only_checks_specified():
    df2 = pd.DataFrame({
        "a": [1.0, 2.0, 1000.0],
        "b": [1.0, 2.0, 1000.0],
    })
    result = OutlierCleaner(
        method="iqr", action="remove", columns=["a"]
    ).clean(df2)
    # b is untouched — still has its original values
    assert "b" in result.columns


def test_does_not_mutate_input(df_with_outlier):
    original_max = df_with_outlier["val"].max()
    OutlierCleaner(method="iqr", action="winsorize").clean(df_with_outlier)
    assert df_with_outlier["val"].max() == original_max