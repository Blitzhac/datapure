"""Tests for DataWriter — all output formats and audit log."""
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from datapure.io.writer import DataWriter


@pytest.fixture
def df():
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol"],
        "age":    [25, 30, 35],
        "salary": [50000.0, 60000.0, 70000.0],
    })


@pytest.fixture
def audit_log():
    return [
        {
            "step": 1,
            "cleaner": "MissingValueCleaner",
            "config": {"strategy": "median"},
            "rows_before": 4,
            "rows_after": 4,
            "rows_removed": 0,
            "nulls_before": 2,
            "nulls_after": 0,
            "nulls_fixed": 2,
            "duration_ms": 1.5,
        }
    ]


def test_write_csv(df, tmp_path):
    path = tmp_path / "out.csv"
    result = DataWriter().write(df, path)
    assert result.exists()
    loaded = pd.read_csv(result)
    assert len(loaded) == len(df)


def test_write_parquet(df, tmp_path):
    path = tmp_path / "out.parquet"
    result = DataWriter().write(df, path)
    assert result.exists()
    loaded = pd.read_parquet(result)
    assert len(loaded) == len(df)


def test_write_json(df, tmp_path):
    path = tmp_path / "out.json"
    result = DataWriter().write(df, path)
    assert result.exists()
    loaded = pd.read_json(result)
    assert len(loaded) == len(df)


def test_write_excel(df, tmp_path):
    path = tmp_path / "out.xlsx"
    result = DataWriter().write(df, path)
    assert result.exists()
    loaded = pd.read_excel(result)
    assert len(loaded) == len(df)


def test_write_creates_parent_dirs(df, tmp_path):
    path = tmp_path / "nested" / "deep" / "out.csv"
    DataWriter().write(df, path)
    assert path.exists()


def test_write_with_audit_log_creates_log_file(df, tmp_path, audit_log):
    path = tmp_path / "out.csv"
    DataWriter().write(df, path, audit_log=audit_log)
    log_path = tmp_path / "out_cleaning_log.json"
    assert log_path.exists()


def test_audit_log_content_correct(df, tmp_path, audit_log):
    path = tmp_path / "out.csv"
    DataWriter().write(df, path, audit_log=audit_log)
    log_path = tmp_path / "out_cleaning_log.json"
    content = json.loads(log_path.read_text())
    assert content["output_rows"] == len(df)
    assert "pipeline_steps" in content
    assert len(content["pipeline_steps"]) == 1


def test_write_polars_parquet(tmp_path):
    lf = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).lazy()
    path = tmp_path / "out.parquet"
    DataWriter().write_polars(lf, path)
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 3


def test_write_polars_csv(tmp_path):
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    path = tmp_path / "out.csv"
    DataWriter().write_polars(lf, path)
    assert path.exists()


def test_unsupported_format_raises(df, tmp_path):
    path = tmp_path / "out.xyz"
    with pytest.raises(ValueError):
        DataWriter().write(df, path)


def test_written_csv_preserves_data(df, tmp_path):
    path = tmp_path / "out.csv"
    DataWriter().write(df, path)
    loaded = pd.read_csv(path)
    assert loaded["name"].tolist() == df["name"].tolist()
    assert loaded["age"].tolist() == df["age"].tolist()


def test_written_parquet_preserves_dtypes(df, tmp_path):
    path = tmp_path / "out.parquet"
    DataWriter().write(df, path)
    loaded = pd.read_parquet(path)
    assert loaded["age"].dtype == df["age"].dtype