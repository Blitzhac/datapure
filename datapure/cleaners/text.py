"""
TextCleaner — fixes encoding, strips HTML, normalises unicode and whitespace.
"""
from __future__ import annotations

import logging
import re
import unicodedata

import ftfy
import pandas as pd

from datapure.cleaners.base import BaseCleaner

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


class TextCleaner(BaseCleaner):
    """
    Args:
        columns:             Columns to clean. None = all object columns.
        fix_encoding:        Fix mojibake with ftfy.
        strip_html:          Remove HTML tags.
        normalize_unicode:   NFC unicode normalisation.
        remove_urls:         Remove http/www URLs.
        normalize_whitespace:Collapse spaces/newlines to single space.
        strip:               Strip leading/trailing whitespace.
        lowercase:           Convert to lowercase.
        custom_patterns:     List of (regex_pattern, replacement) tuples.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        fix_encoding: bool = True,
        strip_html: bool = True,
        normalize_unicode: bool = True,
        remove_urls: bool = False,
        normalize_whitespace: bool = True,
        strip: bool = True,
        lowercase: bool = False,
        custom_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        self.columns = columns
        self.fix_encoding = fix_encoding
        self.strip_html = strip_html
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.normalize_whitespace = normalize_whitespace
        self.strip = strip
        self.lowercase = lowercase
        self.custom_patterns = [
            (re.compile(p), r) for p, r in (custom_patterns or [])
        ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = (
            self.columns if self.columns
            else df.select_dtypes(include=["object", "string"]).columns.tolist()
        )
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_value)
                logger.info("TextCleaner: cleaned column '%s'", col)
        return df

    def _clean_value(self, value: object) -> str | None:
        if pd.isna(value):
            return None

        text = str(value)

        if self.fix_encoding:
            text = ftfy.fix_text(text)

        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        if self.strip_html:
            text = _HTML_TAG_RE.sub(" ", text)

        if self.remove_urls:
            text = _URL_RE.sub("", text)

        for pattern, replacement in self.custom_patterns:
            text = pattern.sub(replacement, text)

        if self.normalize_whitespace:
            text = _MULTI_SPACE_RE.sub(" ", text)

        if self.strip:
            text = text.strip()

        if self.lowercase:
            text = text.lower()

        return text if text else None
