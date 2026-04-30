"""Tests for PolarsNativeMissingCleaner — native Polars operations."""
import polars as pl
import pytest
from datapure.cleaners.polars_missing import PolarsNativeMissingCleaner


@pytest.fixture
def lf():
    return pl.DataFrame({
        "age":    [25.0, None, 35.0, None, 45.0],
        "salary": [50000.0, 60000.0, None, 80000.0, None],
        "city":   ["Delhi", None, "Mumbai", "Delhi", None],
    }).lazy()


def test_returns_lazyframe(lf):
    result = PolarsNativeMissingCleaner(strategy="mean").clean(lf)
    assert isinstance(result, pl.LazyFrame)


def test_mean_fills_numeric_nulls(lf):
    result = PolarsNativeMissingCleaner(strategy="mean").clean(lf).collect()
    assert result["age"].null_count() == 0
    assert result["salary"].null_count() == 0


def test_median_fills_numeric_nulls(lf):
    result = PolarsNativeMissingCleaner(strategy="median").clean(lf).collect()
    assert result["age"].null_count() == 0


def test_median_value_correct():
    lf2 = pl.DataFrame({"x": [1.0, 2.0, None, 100.0]}).lazy()
    result = PolarsNativeMissingCleaner(strategy="median").clean(lf2).collect()
    # median of [1,2,100] = 2.0
    assert result["x"][2] == 2.0


def test_drop_rows_removes_nulls(lf):
    result = PolarsNativeMissingCleaner(strategy="drop_rows").clean(lf).collect()
    assert result.null_count().sum_horizontal()[0] == 0
    assert len(result) < 5


def test_drop_cols_removes_high_null_col():
    lf2 = pl.DataFrame({
        "a": [1.0, None, 3.0],
        "b": [None, None, None],
    }).lazy()
    result = PolarsNativeMissingCleaner(
        strategy="drop_cols", threshold=0.5
    ).clean(lf2).collect()
    assert "b" not in result.columns
    assert "a" in result.columns


def test_constant_fill(lf):
    result = PolarsNativeMissingCleaner(
        strategy="constant", fill_value=0
    ).clean(lf).collect()
    assert result["age"].null_count() == 0
    assert result["salary"].null_count() == 0


def test_mode_fills_string_col(lf):
    result = PolarsNativeMissingCleaner(strategy="mode").clean(lf).collect()
    assert result["city"].null_count() == 0
    assert result["city"][1] == "Delhi"


def test_mean_skips_string_cols(lf):
    result = PolarsNativeMissingCleaner(strategy="mean").clean(lf).collect()
    # city is string — mean can't fill it, nulls remain
    assert result["city"].null_count() > 0


def test_empty_dataframe_no_crash():
    lf2 = pl.DataFrame({"a": [], "b": []}).lazy()
    result = PolarsNativeMissingCleaner(strategy="median").clean(lf2).collect()
    assert len(result) == 0