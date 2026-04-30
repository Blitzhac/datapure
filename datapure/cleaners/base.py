"""
BaseCleaner — the contract every cleaner must fulfill.

All cleaners inherit from this class. This guarantees:
- A consistent clean(df) interface across all modules
- Automatic Polars support via pandas bridge
- Serializable config for pipeline logging
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import polars as pl


class BaseCleaner(ABC):
    """
    Abstract base class for all datapure cleaners.

    Subclasses MUST implement clean(df).
    Subclasses MAY override clean_polars(lf) for native
    Polars performance on large files.
    """

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a pandas DataFrame.

        Rules:
          - Never modify df in-place — always return a new DataFrame
          - Always return a DataFrame even if nothing changed
          - Log what was changed using the logging module
        """
        ...

    def clean_polars(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Clean a Polars LazyFrame.

        Default implementation bridges through pandas.
        Override in subclasses for native Polars performance
        (important for files larger than ~500 MB).
        """
        df = lf.collect().to_pandas()
        df_clean = self.clean(df)
        return pl.from_pandas(df_clean).lazy()

    def get_config(self) -> dict[str, Any]:
        """
        Return current config as a serializable dict.
        Used by Pipeline to log what settings each cleaner used.
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def __repr__(self) -> str:
        cfg = ", ".join(
            f"{k}={v!r}" for k, v in self.get_config().items()
        )
        return f"{self.__class__.__name__}({cfg})"
