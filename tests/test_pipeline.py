"""Tests for Pipeline — chaining, audit log, auto-build, error handling."""
import numpy as np
import pandas as pd
import pytest
from datapure.core.pipeline import Pipeline
from datapure.core.profiler import DataProfiler
from datapure.cleaners.missing import MissingValueCleaner
from datapure.cleaners.outliers import OutlierCleaner
from datapure.cleaners.duplicates import DuplicateCleaner
from datapure.cleaners.schema import SchemaCleaner


@pytest.fixture
def df():
    return pd.DataFrame({
        "age":    [25, np.nan, 35, np.nan, 45, 25],
        "salary": [50000, 60000, np.nan, 80000, np.nan, 50000],
        "city":   ["Delhi", None, "Mumbai", "Delhi", None, "Delhi"],
    })


def test_empty_pipeline_returns_copy(df):
    result = Pipeline().run(df)
    assert result.shape == df.shape


def test_add_returns_self_for_chaining():
    p = Pipeline()
    returned = p.add(MissingValueCleaner())
    assert returned is p


def test_add_wrong_type_raises():
    with pytest.raises(TypeError):
        Pipeline().add("not a cleaner")


def test_single_cleaner_runs(df):
    result = Pipeline().add(MissingValueCleaner(strategy="median")).run(df)
    assert result["age"].isnull().sum() == 0
    assert result["salary"].isnull().sum() == 0


def test_chained_cleaners_run_in_order(df):
    result = (
        Pipeline()
        .add(MissingValueCleaner(strategy="median", col_strategy={"city": "mode"}))
        .add(DuplicateCleaner(mode="exact"))
        .run(df)
    )
    assert result.isnull().sum().sum() == 0
    assert len(result) < len(df)  # duplicates removed


def test_pipeline_does_not_mutate_input(df):
    original_nulls = df.isnull().sum().sum()
    Pipeline().add(MissingValueCleaner(strategy="mean")).run(df)
    assert df.isnull().sum().sum() == original_nulls


def test_audit_log_length_matches_steps(df):
    p = (
        Pipeline()
        .add(MissingValueCleaner(strategy="median"))
        .add(DuplicateCleaner())
    )
    p.run(df)
    assert len(p.get_audit_log()) == 2


def test_audit_log_structure(df):
    p = Pipeline().add(MissingValueCleaner(strategy="mean"))
    p.run(df)
    log = p.get_audit_log()
    assert log[0]["cleaner"] == "MissingValueCleaner"
    assert "rows_before" in log[0]
    assert "nulls_fixed" in log[0]
    assert "duration_ms" in log[0]


def test_audit_log_nulls_fixed_correct(df):
    p = Pipeline().add(
        MissingValueCleaner(strategy="median", col_strategy={"city": "mode"})
    )
    p.run(df)
    log = p.get_audit_log()
    assert log[0]["nulls_fixed"] > 0


def test_print_summary_no_crash(df):
    p = Pipeline().add(MissingValueCleaner(strategy="median"))
    p.run(df)
    p.print_summary()


def test_print_summary_before_run_no_crash():
    Pipeline().print_summary()


def test_from_profile_builds_pipeline(df):
    profile = DataProfiler().run(df)
    pipeline = Pipeline.from_profile(profile)
    assert len(pipeline._cleaners) > 0


def test_from_profile_runs_successfully(df):
    profile = DataProfiler().run(df)
    pipeline = Pipeline.from_profile(profile)
    result = pipeline.run(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_from_profile_reduces_nulls(df):
    profile = DataProfiler().run(df)
    pipeline = Pipeline.from_profile(profile)
    result = pipeline.run(df)
    assert result.isnull().sum().sum() <= df.isnull().sum().sum()


def test_pipeline_name_in_repr():
    p = Pipeline(name="My Pipeline")
    assert p.name == "My Pipeline"