"""
DuplicateCleaner — removes exact, subset, and fuzzy duplicate rows.
"""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from rapidfuzz import fuzz

from datapure.cleaners.base import BaseCleaner

logger = logging.getLogger(__name__)

Mode = Literal["exact", "subset", "fuzzy"]


class DuplicateCleaner(BaseCleaner):
    """
    Removes duplicate rows with three modes:
    - exact:  Completely identical rows.
    - subset: Rows identical on specified columns only.
    - fuzzy:  Near-duplicate rows in a text column.

    Args:
        mode:         "exact" | "subset" | "fuzzy". Default: "exact".
        subset:       Columns for subset mode.
        fuzzy_col:    Column to run fuzzy matching on.
        fuzzy_thresh: Similarity threshold 0-100. Default: 90.
        keep:         "first" | "last".
    """

    def __init__(
        self,
        mode: Mode = "exact",
        subset: list[str] | None = None,
        fuzzy_col: str | None = None,
        fuzzy_thresh: float = 90,
        keep: Literal["first", "last"] = "first",
    ) -> None:
        self.mode = mode
        self.subset = subset
        self.fuzzy_col = fuzzy_col
        self.fuzzy_thresh = fuzzy_thresh
        self.keep = keep

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        before = len(df)

        match self.mode:
            case "exact":
                df = df.drop_duplicates(keep=self.keep)

            case "subset":
                if not self.subset:
                    raise ValueError("subset mode requires the 'subset' parameter")
                df = df.drop_duplicates(subset=self.subset, keep=self.keep)

            case "fuzzy":
                if not self.fuzzy_col:
                    raise ValueError("fuzzy mode requires the 'fuzzy_col' parameter")
                if self.fuzzy_col not in df.columns:
                    raise KeyError(f"Column '{self.fuzzy_col}' not found in DataFrame")
                df = self._remove_fuzzy_dupes(df)

        removed = before - len(df)
        logger.info("DuplicateCleaner: removed %d duplicates (mode=%s)", removed, self.mode)
        return df.reset_index(drop=True)

    def _remove_fuzzy_dupes(self, df: pd.DataFrame) -> pd.DataFrame:
        values = df[self.fuzzy_col].astype("str").tolist()
        keep_mask = [True] * len(values)

        for i in range(len(values)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(values)):
                if not keep_mask[j]:
                    continue
                score = fuzz.ratio(values[i], values[j])
                if score >= self.fuzzy_thresh:
                    keep_mask[j] = False

        return df[keep_mask]
