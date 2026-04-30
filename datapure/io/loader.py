"""
DataLoader — smart file loader for CSV, Parquet, JSON, Excel.
Auto-detects encoding, file type, and switches to Polars
lazy mode for files larger than size_threshold_mb.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import chardet
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# File types we support
FileFormat = Literal["csv", "parquet", "json", "jsonl", "excel", "tsv"]

# Above this size → use Polars LazyFrame instead of pandas DataFrame
DEFAULT_THRESHOLD_MB = 100


class DataLoader:
    """
    Loads data files into pandas or Polars automatically.

    For files under size_threshold_mb  → returns pd.DataFrame
    For files over  size_threshold_mb  → returns pl.LazyFrame

    Usage:
        loader = DataLoader()

        # Auto mode — picks pandas or Polars based on file size
        data = loader.load("sales.csv")

        # Force Polars regardless of size
        lf = loader.load("sales.csv", force_polars=True)

        # Force pandas regardless of size
        df = loader.load("sales.csv", force_pandas=True)
    """

    def __init__(self, size_threshold_mb: float = DEFAULT_THRESHOLD_MB) -> None:
        self.size_threshold_mb = size_threshold_mb

    def load(
        self,
        path: str | Path,
        force_pandas: bool = False,
        force_polars: bool = False,
        **kwargs,
    ) -> pd.DataFrame | pl.LazyFrame:
        """
        Load a file. Returns pd.DataFrame or pl.LazyFrame.

        Args:
            path:          Path to the file.
            force_pandas:  Always return pd.DataFrame.
            force_polars:  Always return pl.LazyFrame.
            **kwargs:      Passed to the underlying reader.
        """
        path = Path(path)
        self._validate_path(path)

        fmt = self._detect_format(path)
        size_mb = path.stat().st_size / 1_048_576
        use_polars = self._should_use_polars(size_mb, force_pandas, force_polars)

        logger.info(
            "Loading '%s' | format=%s | size=%.2f MB | engine=%s",
            path.name, fmt, size_mb, "polars" if use_polars else "pandas",
        )

        if use_polars:
            return self._load_polars(path, fmt, **kwargs)
        return self._load_pandas(path, fmt, **kwargs)

    # ── private helpers ──────────────────────────────────────────

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

    def _detect_format(self, path: Path) -> FileFormat:
        """Infer format from file extension."""
        suffix = path.suffix.lower().lstrip(".")
        mapping = {
            "csv":     "csv",
            "tsv":     "tsv",
            "parquet": "parquet",
            "pq":      "parquet",
            "json":    "json",
            "jsonl":   "jsonl",
            "ndjson":  "jsonl",
            "xlsx":    "excel",
            "xls":     "excel",
            "xlsm":    "excel",
        }
        fmt = mapping.get(suffix)
        if fmt is None:
            raise ValueError(
                f"Unsupported file extension '.{suffix}'. "
                f"Supported: {list(mapping.keys())}"
            )
        return fmt

    def _should_use_polars(
        self, size_mb: float, force_pandas: bool, force_polars: bool
    ) -> bool:
        if force_pandas:
            return False
        if force_polars:
            return True
        return size_mb >= self.size_threshold_mb

    def _detect_encoding(self, path: Path, sample_bytes: int = 65_536) -> str:
        """Detect file encoding using chardet. Samples first 64KB."""
        with open(path, "rb") as f:
            raw = f.read(sample_bytes)
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        confidence = detected.get("confidence", 0)
        logger.info(
            "Detected encoding: %s (confidence: %.0f%%)",
            encoding, confidence * 100,
        )
        return encoding

    def _load_pandas(self, path: Path, fmt: FileFormat, **kwargs) -> pd.DataFrame:
        """Load file into pandas DataFrame."""
        match fmt:
            case "csv":
                encoding = self._detect_encoding(path)
                return pd.read_csv(path, encoding=encoding, **kwargs)

            case "tsv":
                encoding = self._detect_encoding(path)
                return pd.read_csv(path, sep="\t", encoding=encoding, **kwargs)

            case "parquet":
                return pd.read_parquet(path, **kwargs)

            case "json":
                return pd.read_json(path, **kwargs)

            case "jsonl":
                return pd.read_json(path, lines=True, **kwargs)

            case "excel":
                return pd.read_excel(path, **kwargs)

            case _:
                raise ValueError(f"Unknown format: {fmt}")

    def _load_polars(self, path: Path, fmt: FileFormat, **kwargs) -> pl.LazyFrame:
        """Load file into Polars LazyFrame — zero copy, lazy evaluation."""
        match fmt:
            case "csv" | "tsv":
                sep = "\t" if fmt == "tsv" else ","
                return pl.scan_csv(path, separator=sep, infer_schema_length=10_000)

            case "parquet":
                return pl.scan_parquet(path)

            case "json":
                # Polars has no lazy JSON — collect then lazy
                return pl.read_json(path).lazy()

            case "jsonl":
                return pl.scan_ndjson(path)

            case "excel":
                # No lazy Excel in Polars — fall back to pandas then convert
                logger.warning(
                    "Polars has no lazy Excel reader — loading via pandas bridge"
                )
                df_pd = pd.read_excel(path, **kwargs)
                return pl.from_pandas(df_pd).lazy()

            case _:
                raise ValueError(f"Unknown format: {fmt}")

    def get_preview(
        self,
        path: str | Path,
        n_rows: int = 5,
    ) -> pd.DataFrame:
        """
        Load only the first n_rows for a quick preview.
        Always returns pandas regardless of file size.
        Useful for profiling before committing to a full load.
        """
        path = Path(path)
        self._validate_path(path)
        fmt = self._detect_format(path)

        match fmt:
            case "csv" | "tsv":
                sep = "\t" if fmt == "tsv" else ","
                encoding = self._detect_encoding(path)
                return pd.read_csv(
                    path, sep=sep, encoding=encoding, nrows=n_rows
                )
            case "parquet":
                lf = pl.scan_parquet(path)
                return lf.head(n_rows).collect().to_pandas()
            case "json":
                return pd.read_json(path).head(n_rows)
            case "jsonl":
                return pd.read_json(path, lines=True, nrows=n_rows)
            case "excel":
                return pd.read_excel(path, nrows=n_rows)
            case _:
                raise ValueError(f"Unknown format: {fmt}")
