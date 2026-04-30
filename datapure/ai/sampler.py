"""
DataSampler — extracts a compact, representative sample from a DataFrame.
Sends column statistics + sample rows to Claude API safely.
Never sends the full dataset — only a statistical summary + small sample.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Max rows sent to API — keeps payload small and cost low
MAX_SAMPLE_ROWS = 10
# Max unique values shown per categorical column
MAX_UNIQUE_SHOWN = 5


class DataSampler:
    """
    Builds a compact JSON-serializable summary of a DataFrame.
    Safe to send to external APIs — no full data exposure.

    Usage:
        sampler = DataSampler()
        payload = sampler.build(df)
        # payload is a dict ready to json.dumps() and send to Claude
    """

    def build(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Build a compact profile payload from a DataFrame.

        Returns a dict containing:
        - shape: rows and columns
        - sample_rows: first N rows as records
        - columns: per-column stats (dtype, nulls, range, top values)
        """
        payload: dict[str, Any] = {
            "shape": {
                "rows": len(df),
                "cols": len(df.columns),
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "total_null_pct": round(
                df.isnull().sum().sum() / max(df.size, 1) * 100, 2
            ),
            "sample_rows": self._safe_sample(df),
            "columns": {},
        }

        for col in df.columns:
            payload["columns"][col] = self._profile_col(df[col], len(df))

        logger.info(
            "DataSampler: built payload for %d cols (%d sample rows)",
            len(df.columns), len(payload["sample_rows"]),
        )
        return payload

    def _safe_sample(self, df: pd.DataFrame) -> list[dict]:
        """Return up to MAX_SAMPLE_ROWS rows as JSON-safe records."""
        sample = df.head(MAX_SAMPLE_ROWS)
        records = sample.to_dict(orient="records")
        # Manually replace float nan with None — pandas where() misses numeric cols
        cleaned = []
        for row in records:
            cleaned.append({
                k: (None if isinstance(v, float) and v != v else v)
                for k, v in row.items()
            })
        return cleaned

    def _profile_col(self, s: pd.Series, n_rows: int) -> dict[str, Any]:
        """Build per-column stats dict."""
        null_count = int(s.isnull().sum())
        non_null = s.dropna()

        profile: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": round(null_count / max(n_rows, 1) * 100, 2),
            "unique_count": int(s.nunique()),
        }

        if pd.api.types.is_numeric_dtype(s) and len(non_null) > 0:
            profile.update({
                "mean": round(float(non_null.mean()), 4),
                "median": round(float(non_null.median()), 4),
                "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0,
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "outlier_count_iqr": self._count_iqr_outliers(non_null),
            })

        elif len(non_null) > 0:
            top_values = (
                non_null.astype(str)
                .value_counts()
                .head(MAX_UNIQUE_SHOWN)
                .to_dict()
            )
            profile.update({
                "top_values": top_values,
                "avg_str_len": round(
                    float(non_null.astype(str).str.len().mean()), 2
                ),
            })

        return profile

    def _count_iqr_outliers(self, s: pd.Series) -> int:
        """Count values outside IQR fence."""
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
