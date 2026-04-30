"""
Tests for BaseCleaner abstract interface.
Verifies that the ABC contract is correctly enforced.
"""
import pytest
import pandas as pd
import polars as pl
from datapure.cleaners.base import BaseCleaner


# ── Concrete implementation for testing ───────────────────────

class PassthroughCleaner(BaseCleaner):
    """Minimal cleaner that does nothing — used to test the base."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


class ConfiguredCleaner(BaseCleaner):
    """Cleaner with config params — tests get_config() and repr."""

    def __init__(self, threshold: float = 0.5, mode: str = "auto"):
        self.threshold = threshold
        self.mode = mode

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


class BrokenCleaner(BaseCleaner):
    """Forgot to implement clean() — ABC must reject this."""
    pass


# ── Tests ────────────────────────────────────────────────────

def test_cannot_instantiate_abstract_class():
    """BaseCleaner itself must not be instantiable."""
    with pytest.raises(TypeError):
        BaseCleaner()


def test_missing_clean_raises_typeerror():
    """Subclass that forgets clean() must raise TypeError."""
    with pytest.raises(TypeError):
        BrokenCleaner()


def test_concrete_cleaner_instantiates():
    c = PassthroughCleaner()
    assert c is not None


def test_clean_returns_dataframe():
    c = PassthroughCleaner()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = c.clean(df)
    assert isinstance(result, pd.DataFrame)


def test_clean_preserves_shape():
    c = PassthroughCleaner()
    df = pd.DataFrame({"x": range(10)})
    assert c.clean(df).shape == df.shape


def test_polars_bridge_works():
    """Default clean_polars() must work via pandas bridge."""
    c = PassthroughCleaner()
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    result = c.clean_polars(lf)
    assert isinstance(result, pl.LazyFrame)
    assert result.collect().shape == (3, 1)


def test_get_config_returns_dict():
    c = ConfiguredCleaner(threshold=0.8, mode="strict")
    cfg = c.get_config()
    assert cfg == {"threshold": 0.8, "mode": "strict"}


def test_repr_contains_class_name():
    c = ConfiguredCleaner()
    assert "ConfiguredCleaner" in repr(c)


def test_repr_contains_config_values():
    c = ConfiguredCleaner(threshold=0.9)
    assert "0.9" in repr(c)