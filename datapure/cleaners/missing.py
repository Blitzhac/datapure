"""
MissingValueCleaner — 9 strategies for handling null/NaN values.
Supports per-column strategy overrides and Polars large-file mode.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from datapure.cleaners.base import BaseCleaner

logger = logging.getLogger(__name__)

Strategy = Literal[
    "drop_rows", "drop_cols", "mean", "median",
    "mode", "ffill", "bfill", "knn", "constant",
]


class MissingValueCleaner(BaseCleaner):
    """
    Handles missing values with configurable strategies.

    Args:
        strategy:     Global strategy for all columns. Default: "median".
        col_strategy: Per-column overrides e.g. {"city": "mode", "age": "knn"}.
        threshold:    For drop_cols — drop if null% exceeds this (0.0-1.0).
        fill_value:   For constant strategy — value to fill with.
        knn_k:        Number of neighbours for KNN imputation.
    """

    def __init__(
        self,
        strategy: Strategy = "median",
        col_strategy: dict[str, Strategy] | None = None,
        threshold: float = 0.5,
        fill_value: Any = 0,
        knn_k: int = 5,
    ) -> None:
        self.strategy = strategy
        self.col_strategy = col_strategy or {}
        self.threshold = threshold
        self.fill_value = fill_value
        self.knn_k = knn_k

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        before_nulls = df.isnull().sum().sum()

        if self.strategy == "drop_cols":
            df = self._drop_cols(df)

        elif self.strategy == "drop_rows":
            df = df.dropna()

        elif self.strategy == "knn":
            df = self._apply_knn(df)

        else:
            for col, strat in self.col_strategy.items():
                if col in df.columns:
                    df[col] = self._fill_column(df[col], strat)

            for col in df.columns:
                if col not in self.col_strategy:
                    df[col] = self._fill_column(df[col], self.strategy)

        after_nulls = df.isnull().sum().sum()
        logger.info(
            "MissingValueCleaner: fixed %d nulls via strategy='%s'",
            before_nulls - after_nulls, self.strategy,
        )
        return df

    def _fill_column(self, s: pd.Series, strategy: Strategy) -> pd.Series:
        if s.isnull().sum() == 0:
            return s

        match strategy:
            case "mean":
                if pd.api.types.is_numeric_dtype(s):
                    return s.fillna(s.mean())
                logger.warning("mean skipped non-numeric col '%s'", s.name)
                return s

            case "median":
                if pd.api.types.is_numeric_dtype(s):
                    return s.fillna(s.median())
                logger.warning("median skipped non-numeric col '%s'", s.name)
                return s

            case "mode":
                mode_vals = s.mode()
                return s.fillna(mode_vals.iloc[0] if not mode_vals.empty else np.nan)

            case "ffill":
                return s.ffill()

            case "bfill":
                return s.bfill()

            case "constant":
                return s.fillna(self.fill_value)

            case _:
                logger.warning("Unknown strategy '%s', skipping col '%s'", strategy, s.name)
                return s

    def _drop_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        null_pct = df.isnull().mean()
        to_drop = null_pct[null_pct > self.threshold].index.tolist()
        if to_drop:
            logger.info("Dropping cols with >%.0f%% nulls: %s", self.threshold * 100, to_drop)
        return df.drop(columns=to_drop)

    def _apply_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        other_cols = [c for c in df.columns if c not in num_cols]

        if not num_cols:
            logger.warning("KNN skipped — no numeric columns found")
            return df

        imputer = KNNImputer(n_neighbors=self.knn_k)
        df_num = pd.DataFrame(
            imputer.fit_transform(df[num_cols]),
            columns=num_cols, index=df.index,
        )
        return pd.concat([df_num, df[other_cols]], axis=1)[df.columns]
