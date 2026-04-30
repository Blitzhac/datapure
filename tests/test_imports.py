"""
Phase 1 smoke tests — verify the package structure is correct.
These should all pass immediately after pip install -e .[dev]
"""
import sys


def test_python_version():
    """datapure requires Python 3.10+ for match/case syntax."""
    assert sys.version_info >= (3, 10), (
        f"Need Python 3.10+, got {sys.version}"
    )


def test_main_package_imports():
    """Top-level package must import cleanly."""
    import datapure
    assert datapure.__version__ == "0.1.0"


def test_all_subpackages_import():
    """Every subpackage must be importable."""
    from datapure import cleaners, core, ai, io, cli
    assert all([cleaners, core, ai, io, cli])


def test_base_cleaner_importable():
    """BaseCleaner must be importable from cleaners package."""
    from datapure.cleaners import BaseCleaner
    assert BaseCleaner is not None


def test_key_dependencies_available():
    """All critical third-party libs must be installed."""
    import pandas, polars, numpy, sklearn, rich, click
    assert all([pandas, polars, numpy, sklearn, rich, click])


def test_pandas_version():
    """Need pandas 2.x for modern API."""
    import pandas as pd
    major = int(pd.__version__.split(".")[0])
    assert major >= 2, f"Need pandas 2+, got {pd.__version__}"