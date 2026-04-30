"""
PolarsNativeMissingCleaner — fills nulls using native Polars expressions.
No pandas conversion — operates directly on LazyFrames.
Use this for files larger than ~500 MB where the pandas bridge is too slow.
"""
from __future__ import annotations

import logging
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

Strategy = Literal["mean", "median", "mode", "drop_rows", "drop_cols", "constant"]


class PolarsNativeMissingCleaner:
    """
    Native Polars missing value cleaner.

    Does NOT inherit from BaseCleaner because it operates on
    pl.LazyFrame not pd.DataFrame. Used by the IO layer when
    a file is loaded in Polars mode.

    Args:
        strategy:     Global fill strategy.
        threshold:    For drop_cols — drop cols with null% above this.
        fill_value:   For constant strategy.

    Example:
        lf = pl.scan_csv("big_file.csv")
        cleaner = PolarsNativeMissingCleaner(strategy="median")
        lf_clean = cleaner.clean(lf)
        df = lf_clean.collect()
    """

    def __init__(
        self,
        strategy: Strategy = "median",
        threshold: float = 0.5,
        fill_value: int | float | str = 0,
    ) -> None:
        self.strategy = strategy
        self.threshold = threshold
        self.fill_value = fill_value

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply strategy to LazyFrame. Returns LazyFrame."""
        match self.strategy:
            case "drop_rows":
                return lf.drop_nulls()

            case "drop_cols":
                return self._drop_null_cols(lf)

            case "constant":
                return lf.fill_null(self.fill_value)

            case "mean":
                return self._fill_with_stat(lf, "mean")

            case "median":
                return self._fill_with_stat(lf, "median")

            case "mode":
                return self._fill_with_mode(lf)

            case _:
                logger.warning(
                    "PolarsNativeMissingCleaner: unknown strategy '%s'",
                    self.strategy,
                )
                return lf

    def _fill_with_stat(
        self, lf: pl.LazyFrame, stat: Literal["mean", "median"]
    ) -> pl.LazyFrame:
        """
        Fill numeric nulls with mean or median using Polars expressions.
        Non-numeric columns are left unchanged.
        Operates fully lazily — no collect() needed.
        """
        schema = lf.collect_schema()
        exprs = []

        for col_name, dtype in schema.items():
            if dtype in (
                pl.Float32, pl.Float64,
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            ):
                if stat == "mean":
                    fill_expr = pl.col(col_name).fill_null(
                        pl.col(col_name).mean()
                    )
                else:
                    fill_expr = pl.col(col_name).fill_null(
                        pl.col(col_name).median()
                    )
                exprs.append(fill_expr)
            else:
                exprs.append(pl.col(col_name))

        return lf.select(exprs)

    def _fill_with_mode(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Fill nulls with the most frequent value per column.
        Requires a collect() to compute mode — unavoidable in Polars.
        """
        df = lf.collect()
        exprs = []

        for col_name in df.columns:
            series = df[col_name]
            non_null = series.drop_nulls()
            if len(non_null) == 0:
                exprs.append(pl.col(col_name))
                continue
            mode_val = non_null.mode()[0]
            exprs.append(
                pl.col(col_name).fill_null(pl.lit(mode_val))
            )

        return df.lazy().select(exprs)

    def _drop_null_cols(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Drop columns where null fraction exceeds self.threshold.
        Requires a collect() to calculate null fractions.
        """
        df = lf.collect()
        n_rows = len(df)
        if n_rows == 0:
            return lf

        cols_to_keep = [
            col for col in df.columns
            if df[col].null_count() / n_rows <= self.threshold
        ]

        removed = set(df.columns) - set(cols_to_keep)
        if removed:
            logger.info(
                "PolarsNativeMissingCleaner: dropping cols %s", removed
            )

        return df.select(cols_to_keep).lazy()
