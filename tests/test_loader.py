"""Tests for DataLoader — format detection, encoding, pandas/polars switching."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import polars as pl

from datapure.io.loader import DataLoader


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol"],
        "age":    [25, 30, 35],
        "salary": [50000.0, 60000.0, 70000.0],
    })


@pytest.fixture
def csv_file(sample_df, tmp_path):
    path = tmp_path / "test.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def parquet_file(sample_df, tmp_path):
    path = tmp_path / "test.parquet"
    sample_df.to_parquet(path, index=False)
    return path


@pytest.fixture
def json_file(sample_df, tmp_path):
    path = tmp_path / "test.json"
    sample_df.to_json(path, orient="records", indent=2)
    return path


@pytest.fixture
def excel_file(sample_df, tmp_path):
    path = tmp_path / "test.xlsx"
    sample_df.to_excel(path, index=False)
    return path


def test_load_csv_returns_dataframe(csv_file):
    result = DataLoader().load(csv_file, force_pandas=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_load_csv_columns_correct(csv_file, sample_df):
    result = DataLoader().load(csv_file, force_pandas=True)
    assert list(result.columns) == list(sample_df.columns)


def test_load_parquet_returns_dataframe(parquet_file):
    result = DataLoader().load(parquet_file, force_pandas=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_load_json_returns_dataframe(json_file):
    result = DataLoader().load(json_file, force_pandas=True)
    assert isinstance(result, pd.DataFrame)


def test_load_excel_returns_dataframe(excel_file):
    result = DataLoader().load(excel_file, force_pandas=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_force_polars_returns_lazyframe(csv_file):
    result = DataLoader().load(csv_file, force_polars=True)
    assert isinstance(result, pl.LazyFrame)


def test_force_polars_parquet(parquet_file):
    result = DataLoader().load(parquet_file, force_polars=True)
    assert isinstance(result, pl.LazyFrame)


def test_force_pandas_overrides_threshold(csv_file):
    # Even with threshold=0, force_pandas wins
    loader = DataLoader(size_threshold_mb=0)
    result = loader.load(csv_file, force_pandas=True)
    assert isinstance(result, pd.DataFrame)


def test_small_file_uses_pandas_by_default(csv_file):
    # Default threshold is 100 MB — small file should use pandas
    loader = DataLoader(size_threshold_mb=100)
    result = loader.load(csv_file)
    assert isinstance(result, pd.DataFrame)


def test_low_threshold_switches_to_polars(csv_file):
    # Set threshold to 0 — any file triggers Polars
    loader = DataLoader(size_threshold_mb=0)
    result = loader.load(csv_file)
    assert isinstance(result, pl.LazyFrame)


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        DataLoader().load("nonexistent_file.csv")


def test_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("data")
    with pytest.raises(ValueError, match="Unsupported"):
        DataLoader().load(bad_file)


def test_detect_encoding_utf8(csv_file):
    loader = DataLoader()
    encoding = loader._detect_encoding(csv_file)
    assert encoding.lower().replace("-", "") in ("utf8", "ascii")


def test_get_preview_returns_n_rows(csv_file):
    result = DataLoader().get_preview(csv_file, n_rows=2)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_get_preview_parquet(parquet_file):
    result = DataLoader().get_preview(parquet_file, n_rows=2)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_tsv_file_loads_correctly(sample_df, tmp_path):
    path = tmp_path / "test.tsv"
    sample_df.to_csv(path, sep="\t", index=False)
    result = DataLoader().load(path, force_pandas=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3