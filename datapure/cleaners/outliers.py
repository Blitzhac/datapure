"""
OutlierCleaner — detects and handles outliers in numeric columns.
Methods: IQR, Z-score, Isolation Forest. Actions: remove or winsorize.
"""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sklearn.ensemble import IsolationForest

from datapure.cleaners.base import BaseCleaner

logger = logging.getLogger(__name__)

Method = Literal["iqr", "zscore", "isolation_forest"]
Action = Literal["remove", "winsorize"]


class OutlierCleaner(BaseCleaner):
    """
    Args:
        method:           "iqr" | "zscore" | "isolation_forest".
        action:           "remove" | "winsorize".
        iqr_factor:       IQR multiplier. Default 1.5.
        zscore_thresh:    Z-score cutoff. Default 3.0.
        if_contamination: Isolation Forest outlier fraction. Default "auto".
        columns:          Specific columns to check. None = all numeric.
    """

    def __init__(
        self,
        method: Method = "iqr",
        action: Action = "winsorize",
        iqr_factor: float = 1.5,
        zscore_thresh: float = 3.0,
        if_contamination: float | str = "auto",
        columns: list[str] | None = None,
    ) -> None:
        self.method = method
        self.action = action
        self.iqr_factor = iqr_factor
        self.zscore_thresh = zscore_thresh
        self.if_contamination = if_contamination
        self.columns = columns

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        num_cols = (
            self.columns if self.columns
            else df.select_dtypes(include="number").columns.tolist()
        )

        if not num_cols:
            logger.warning("OutlierCleaner: no numeric columns found, skipping")
            return df

        if self.method == "isolation_forest":
            return self._apply_isolation_forest(df, num_cols)

        outlier_mask = pd.Series(False, index=df.index)
        for col in num_cols:
            outlier_mask |= self._detect_col(df[col])

        if self.action == "remove":
            logger.info("OutlierCleaner: removed %d outlier rows", outlier_mask.sum())
            return df[~outlier_mask].reset_index(drop=True)

        for col in num_cols:
            df[col] = self._winsorize_col(df[col])
        logger.info("OutlierCleaner: winsorized %d cols", len(num_cols))
        return df

    def _detect_col(self, s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if len(s_clean) == 0:
            return pd.Series(False, index=s.index)

        if self.method == "iqr":
            q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - self.iqr_factor * iqr, q3 + self.iqr_factor * iqr
            return (s < lower) | (s > upper)

        if self.method == "zscore":
            mean, std = s_clean.mean(), s_clean.std()
            if std == 0:
                return pd.Series(False, index=s.index)
            z = (s - mean) / std
            return z.abs() > self.zscore_thresh

        return pd.Series(False, index=s.index)

    def _winsorize_col(self, s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if len(s_clean) == 0:
            return s

        if self.method == "iqr":
            q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
        else:
            mean, std = s_clean.mean(), s_clean.std()
            lower = mean - self.zscore_thresh * std
            upper = mean + self.zscore_thresh * std

        return s.clip(lower=lower, upper=upper)

    def _apply_isolation_forest(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        subset = df[cols].dropna()
        clf = IsolationForest(contamination=self.if_contamination, random_state=42)
        preds = clf.fit_predict(subset)
        outlier_idx = subset.index[preds == -1]

        if self.action == "remove":
            logger.info("IsolationForest: removing %d outliers", len(outlier_idx))
            return df.drop(index=outlier_idx).reset_index(drop=True)

        df_clean = df.copy()
        for col in cols:
            df_clean.loc[outlier_idx, col] = df[col].median()
        return df_clean
