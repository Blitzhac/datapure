"""Tests for CLI commands using Click's test runner."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from datapure.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({
        "age":    [25, np.nan, 35, np.nan, 45, 25],
        "salary": [50000.0, 60000.0, np.nan, 80000.0, 1000000.0, 50000.0],
        "city":   ["Delhi", None, "Mumbai", "Delhi", None, "Delhi"],
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def parquet_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path / "test.parquet"
    df.to_parquet(path, index=False)
    return path


# ── version ──────────────────────────────────────────────────────

def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "clean" in result.output
    assert "profile" in result.output
    assert "suggest" in result.output


# ── clean command ─────────────────────────────────────────────────

def test_clean_basic(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--no-schema",
    ])
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_clean_output_has_fewer_nulls(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--missing", "median",
        "--no-schema",
    ])
    df_in = pd.read_csv(csv_file)
    df_out = pd.read_csv(output)
    assert df_out.isnull().sum().sum() <= df_in.isnull().sum().sum()


def test_clean_missing_mean(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--missing", "mean",
        "--no-schema",
    ])
    assert result.exit_code == 0


def test_clean_missing_drop_rows(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--missing", "drop_rows",
        "--no-schema",
    ])
    df_in = pd.read_csv(csv_file)
    df_out = pd.read_csv(output)
    assert len(df_out) <= len(df_in)


def test_clean_outliers_none(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--outliers", "none",
        "--no-schema",
    ])
    assert result.exit_code == 0


def test_clean_outliers_remove(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--outliers", "iqr",
        "--outlier-action", "remove",
        "--no-schema",
    ])
    assert result.exit_code == 0


def test_clean_no_duplicates_flag(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--no-duplicates",
        "--no-schema",
    ])
    assert result.exit_code == 0


def test_clean_with_text_flag(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--text",
        "--no-schema",
    ])
    assert result.exit_code == 0


def test_clean_output_parquet(runner, csv_file, tmp_path):
    output = tmp_path / "out.parquet"
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--no-schema",
    ])
    assert result.exit_code == 0
    assert output.exists()
    df = pd.read_parquet(output)
    assert len(df) > 0


def test_clean_generates_audit_log(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--no-schema",
    ])
    log_path = tmp_path / "out_cleaning_log.json"
    assert log_path.exists()
    content = json.loads(log_path.read_text())
    assert "pipeline_steps" in content


def test_clean_generates_html_report(runner, csv_file, tmp_path):
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--output", str(output),
        "--report",
        "--no-schema",
    ])
    assert result.exit_code == 0
    report_path = tmp_path / "out_report.html"
    assert report_path.exists()
    content = report_path.read_text()
    assert "DataPure" in content


def test_clean_default_output_path(runner, csv_file):
    """Without --output, creates input_clean.csv in same dir."""
    result = runner.invoke(cli, [
        "clean", str(csv_file),
        "--no-schema",
    ])
    assert result.exit_code == 0
    expected = csv_file.parent / "test_clean.csv"
    assert expected.exists()
    expected.unlink()


def test_clean_nonexistent_file(runner):
    result = runner.invoke(cli, ["clean", "nonexistent.csv"])
    assert result.exit_code != 0


# ── profile command ───────────────────────────────────────────────

def test_profile_runs(runner, csv_file):
    result = runner.invoke(cli, ["profile", str(csv_file)])
    assert result.exit_code == 0


def test_profile_shows_column_names(runner, csv_file):
    result = runner.invoke(cli, ["profile", str(csv_file)])
    assert "age" in result.output
    assert "salary" in result.output
    assert "city" in result.output


def test_profile_json_output(runner, csv_file, tmp_path):
    json_out = tmp_path / "profile.json"
    result = runner.invoke(cli, [
        "profile", str(csv_file),
        "--json-output", str(json_out),
    ])
    assert result.exit_code == 0
    assert json_out.exists()
    content = json.loads(json_out.read_text())
    assert "shape" in content
    assert "columns" in content


def test_profile_shows_null_info(runner, csv_file):
    result = runner.invoke(cli, ["profile", str(csv_file)])
    # Rich table output should mention null percentages
    assert "%" in result.output


def test_profile_parquet(runner, parquet_file):
    result = runner.invoke(cli, ["profile", str(parquet_file)])
    assert result.exit_code == 0


# ── suggest command ───────────────────────────────────────────────

def test_suggest_no_api_key(runner, csv_file, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(cli, ["suggest", str(csv_file)])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


# ── help texts ────────────────────────────────────────────────────

def test_clean_help(runner):
    result = runner.invoke(cli, ["clean", "--help"])
    assert result.exit_code == 0
    assert "--missing" in result.output
    assert "--outliers" in result.output
    assert "--report" in result.output


def test_profile_help(runner):
    result = runner.invoke(cli, ["profile", "--help"])
    assert result.exit_code == 0


def test_suggest_help(runner):
    result = runner.invoke(cli, ["suggest", "--help"])
    assert result.exit_code == 0