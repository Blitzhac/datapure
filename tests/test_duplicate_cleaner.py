"""Tests for DuplicateCleaner — exact, subset, and fuzzy modes."""
import pandas as pd
import pytest
from datapure.cleaners.duplicates import DuplicateCleaner


@pytest.fixture
def df():
    return pd.DataFrame({
        "id":   [1, 2, 2, 3, 4],
        "name": ["Alice", "Bob", "Bob", "Carol", "Dave"],
        "city": ["NY", "LA", "LA", "NY", "SF"],
    })


def test_exact_removes_duplicate_rows(df):
    result = DuplicateCleaner(mode="exact").clean(df)
    assert len(result) == 4


def test_exact_no_duplicates_unchanged():
    df2 = pd.DataFrame({"a": [1, 2, 3]})
    result = DuplicateCleaner(mode="exact").clean(df2)
    assert len(result) == 3


def test_subset_dedup_on_name(df):
    result = DuplicateCleaner(mode="subset", subset=["name"]).clean(df)
    assert len(result) == 4
    assert result["name"].tolist() == ["Alice", "Bob", "Carol", "Dave"]


def test_subset_keep_last():
    df2 = pd.DataFrame({
        "name": ["Bob", "Bob"],
        "score": [10, 20],
    })
    result = DuplicateCleaner(mode="subset", subset=["name"], keep="last").clean(df2)
    assert result["score"].iloc[0] == 20


def test_subset_missing_param_raises():
    with pytest.raises(ValueError, match="subset"):
        DuplicateCleaner(mode="subset").clean(pd.DataFrame({"a": [1, 1]}))


def test_fuzzy_removes_near_duplicates():
    df2 = pd.DataFrame({
        "name": ["John Smith", "Jon Smith", "Jane Doe"]
    })
    result = DuplicateCleaner(
        mode="fuzzy", fuzzy_col="name", fuzzy_thresh=85
    ).clean(df2)
    assert len(result) == 2
    assert "John Smith" in result["name"].values


def test_fuzzy_keeps_dissimilar_names():
    df2 = pd.DataFrame({
        "name": ["Alice Johnson", "Bob Williams", "Alyce Johnson"]
    })
    result = DuplicateCleaner(
        mode="fuzzy", fuzzy_col="name", fuzzy_thresh=95
    ).clean(df2)
    # High threshold — only near-exact matches removed
    assert len(result) >= 2


def test_fuzzy_missing_col_raises():
    with pytest.raises(ValueError, match="fuzzy_col"):
        DuplicateCleaner(mode="fuzzy").clean(pd.DataFrame({"a": ["x"]}))


def test_fuzzy_nonexistent_col_raises():
    with pytest.raises(KeyError):
        DuplicateCleaner(
            mode="fuzzy", fuzzy_col="nonexistent"
        ).clean(pd.DataFrame({"a": ["x"]}))


def test_index_reset_after_dedup(df):
    result = DuplicateCleaner(mode="exact").clean(df)
    assert result.index.tolist() == list(range(len(result)))


def test_does_not_mutate_input(df):
    original_len = len(df)
    DuplicateCleaner().clean(df)
    assert len(df) == original_len