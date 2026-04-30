"""
SchemaCleaner — fixes dtypes, normalises formats, validates emails/phones.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from dateutil import parser as dateparser
import pandas as pd

from datapure.cleaners.base import BaseCleaner

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$")
_PHONE_RE = re.compile(r"[\d\s\-\(\)\+]{7,15}")


class SchemaCleaner(BaseCleaner):
    """
    Args:
        auto_coerce:       Try to coerce object columns to numeric.
        normalize_strings: Strip whitespace from all string columns.
        lowercase_strings: Also lowercase all string values.
        date_columns:      Columns to parse as datetime.
        bool_columns:      Columns to coerce to bool.
        email_columns:     Columns to validate as emails (adds _valid col).
        phone_columns:     Columns to validate as phones (adds _valid col).
    """

    def __init__(
        self,
        auto_coerce: bool = True,
        normalize_strings: bool = True,
        lowercase_strings: bool = False,
        date_columns: list[str] | None = None,
        bool_columns: list[str] | None = None,
        email_columns: list[str] | None = None,
        phone_columns: list[str] | None = None,
    ) -> None:
        self.auto_coerce = auto_coerce
        self.normalize_strings = normalize_strings
        self.lowercase_strings = lowercase_strings
        self.date_columns = date_columns or []
        self.bool_columns = bool_columns or []
        self.email_columns = email_columns or []
        self.phone_columns = phone_columns or []

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.normalize_strings:
            df = self._normalize_strings(df)

        for col in self.date_columns:
            if col in df.columns:
                df[col] = self._parse_dates(df[col])

        for col in self.bool_columns:
            if col in df.columns:
                df[col] = self._coerce_bool(df[col])

        for col in self.email_columns:
            if col in df.columns:
                df[f"{col}_valid"] = df[col].apply(self._validate_email)

        for col in self.phone_columns:
            if col in df.columns:
                df[f"{col}_valid"] = df[col].apply(self._validate_phone)

        if self.auto_coerce:
            df = self._auto_coerce(df)

        return df

    def _normalize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            df[col] = df[col].astype("str").str.strip()
            if self.lowercase_strings:
                df[col] = df[col].str.lower()
            df[col] = df[col].replace("nan", pd.NA)
        return df

    def _parse_dates(self, s: pd.Series) -> pd.Series:
        try:
            return pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
        except Exception:
            return s.apply(
                lambda v: dateparser.parse(str(v)) if pd.notna(v) else pd.NaT
            )

    def _coerce_bool(self, s: pd.Series) -> pd.Series:
        mapping: dict[Any, bool] = {
            "true": True, "1": True, "yes": True, "y": True,
            "false": False, "0": False, "no": False, "n": False,
            1: True, 0: False,
        }
        return s.map(lambda v: mapping.get(str(v).lower().strip(), pd.NA))

    def _validate_email(self, value: Any) -> bool:
        if pd.isna(value):
            return False
        return bool(_EMAIL_RE.match(str(value).strip()))

    def _validate_phone(self, value: Any) -> bool:
        if pd.isna(value):
            return False
        return bool(_PHONE_RE.search(str(value).strip()))

    def _auto_coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            numeric_attempt = pd.to_numeric(df[col], errors="coerce")
            if numeric_attempt.notna().mean() > 0.9:
                df[col] = numeric_attempt
                logger.info("Auto-coerced '%s' to numeric", col)
        return df
