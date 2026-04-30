"""Tests for DataSampler — payload building and column profiling."""
import numpy as np
import pandas as pd
import pytest
from datapure.ai.sampler import DataSampler


@pytest.fixture
def df():
    return pd.DataFrame({
        "age":    [25, np.nan, 35, np.nan, 45],
        "salary": [50000.0, 60000.0, np.nan, 80000.0, 1000000.0],
        "city":   ["Delhi", None, "Mumbai", "Delhi", None],
    })


@pytest.fixture
def payload(df):
    return DataSampler().build(df)


def test_payload_has_shape(payload, df):
    assert payload["shape"]["rows"] == len(df)
    assert payload["shape"]["cols"] == len(df.columns)


def test_payload_has_all_columns(payload, df):
    assert set(payload["columns"].keys()) == set(df.columns)


def test_payload_has_sample_rows(payload):
    assert "sample_rows" in payload
    assert len(payload["sample_rows"]) <= 10


def test_sample_rows_are_json_safe(payload):
    import json
    # Should not raise
    json.dumps(payload)


def test_null_count_correct(payload):
    assert payload["columns"]["age"]["null_count"] == 2
    assert payload["columns"]["city"]["null_count"] == 2


def test_null_pct_correct(payload):
    assert payload["columns"]["age"]["null_pct"] == 40.0


def test_numeric_col_has_stats(payload):
    age = payload["columns"]["age"]
    assert "mean" in age
    assert "median" in age
    assert "min" in age
    assert "max" in age


def test_numeric_col_outlier_count(payload):
    salary = payload["columns"]["salary"]
    # 1000000 is a clear outlier
    assert salary["outlier_count_iqr"] >= 1


def test_string_col_has_top_values(payload):
    city = payload["columns"]["city"]
    assert "top_values" in city
    assert "Delhi" in city["top_values"]


def test_string_col_has_avg_len(payload):
    city = payload["columns"]["city"]
    assert "avg_str_len" in city
    assert city["avg_str_len"] > 0


def test_duplicate_rows_counted():
    df2 = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    payload = DataSampler().build(df2)
    assert payload["duplicate_rows"] == 1


def test_total_null_pct_correct(payload):
    assert payload["total_null_pct"] > 0


def test_empty_dataframe_no_crash():
    df2 = pd.DataFrame({"a": [], "b": []})
    payload = DataSampler().build(df2)
    assert payload["shape"]["rows"] == 0


def test_sample_capped_at_10_rows():
    df2 = pd.DataFrame({"x": range(100)})
    payload = DataSampler().build(df2)
    assert len(payload["sample_rows"]) <= 10


def test_nan_replaced_with_none_in_sample():
    df2 = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    payload = DataSampler().build(df2)
    sample_vals = [r["a"] for r in payload["sample_rows"]]
    assert None in sample_vals
    assert not any(
        str(v) == "nan" for v in sample_vals if v is not None
    )