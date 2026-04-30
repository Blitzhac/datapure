"""
DataWriter — writes cleaned DataFrames to CSV, Parquet, JSON, or Excel.
Always saves a cleaning_log.json alongside the output.
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

OutputFormat = Literal["csv", "parquet", "json", "excel"]


class DataWriter:
    """
    Writes cleaned data to disk in multiple formats.

    Usage:
        writer = DataWriter()

        # Write pandas DataFrame
        writer.write(df_clean, "output/clean.parquet")

        # Write with audit log
        writer.write(
            df_clean,
            "output/clean.csv",
            audit_log=pipeline.get_audit_log(),
        )

        # Write Polars LazyFrame directly (no pandas conversion)
        writer.write_polars(lf_clean, "output/clean.parquet")
    """

    def write(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        fmt: OutputFormat | None = None,
        audit_log: list[dict] | None = None,
        **kwargs,
    ) -> Path:
        """
        Write DataFrame to file. Format inferred from extension if not given.

        Args:
            df:          Cleaned pandas DataFrame.
            output_path: Destination path including filename and extension.
            fmt:         Override format detection.
            audit_log:   If provided, saved as output_path + _log.json.
            **kwargs:    Passed to the underlying writer.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detected_fmt = fmt or self._detect_format(output_path)
        size_before = len(df)

        match detected_fmt:
            case "csv":
                df.to_csv(output_path, index=False, **kwargs)
            case "parquet":
                df.to_parquet(output_path, index=False, **kwargs)
            case "json":
                df.to_json(output_path, orient="records", indent=2, **kwargs)
            case "excel":
                df.to_excel(output_path, index=False, **kwargs)
            case _:
                raise ValueError(f"Unsupported format: {detected_fmt}")

        size_mb = output_path.stat().st_size / 1_048_576
        logger.info(
            "Written %d rows to '%s' (%.3f MB)",
            size_before, output_path, size_mb,
        )

        if audit_log is not None:
            self._write_log(output_path, audit_log, df)

        return output_path

    def write_polars(
        self,
        lf: pl.LazyFrame,
        output_path: str | Path,
        fmt: OutputFormat | None = None,
        audit_log: list[dict] | None = None,
    ) -> Path:
        """
        Write Polars LazyFrame directly — no pandas conversion.
        Significantly faster for large files.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detected_fmt = fmt or self._detect_format(output_path)

        match detected_fmt:
            case "parquet":
                # Streaming collect — handles files larger than RAM
                lf.sink_parquet(output_path)
            case "csv":
                lf.sink_csv(output_path)
            case "json":
                # No sink_json in Polars — collect then write
                lf.collect().write_ndjson(output_path)
            case _:
                raise ValueError(
                    f"Polars writer does not support format: {detected_fmt}. "
                    "Use write() for Excel output."
                )

        size_mb = output_path.stat().st_size / 1_048_576
        logger.info("Written Polars LazyFrame to '%s' (%.3f MB)", output_path, size_mb)

        if audit_log is not None:
            collected = lf.collect().to_pandas()
            self._write_log(output_path, audit_log, collected)

        return output_path

    def _detect_format(self, path: Path) -> OutputFormat:
        suffix = path.suffix.lower().lstrip(".")
        mapping = {
            "csv":     "csv",
            "tsv":     "csv",
            "parquet": "parquet",
            "pq":      "parquet",
            "json":    "json",
            "jsonl":   "json",
            "xlsx":    "excel",
            "xls":     "excel",
        }
        fmt = mapping.get(suffix)
        if fmt is None:
            raise ValueError(f"Cannot infer format from extension '.{suffix}'")
        return fmt

    def _write_log(
        self,
        data_path: Path,
        audit_log: list[dict],
        df: pd.DataFrame,
    ) -> Path:
        """Write cleaning_log.json alongside the output file."""
        log_path = data_path.parent / f"{data_path.stem}_cleaning_log.json"
        log_data = {
            "generated_at": datetime.now().isoformat(),
            "output_file": str(data_path),
            "output_rows": len(df),
            "output_cols": len(df.columns),
            "columns": list(df.columns),
            "pipeline_steps": audit_log,
        }
        log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
        logger.info("Audit log saved to '%s'", log_path)
        return log_path
