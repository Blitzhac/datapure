"""
DataProfiler — analyzes a DataFrame and surfaces quality issues.
Run this before cleaning to understand what problems exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ColumnProfile:
    """Profile statistics for a single column."""
    name: str
    dtype: str
    total_rows: int
    null_count: int
    null_pct: float
    unique_count: int
    cardinality_pct: float
    # numeric only
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    outlier_count_iqr: int = 0
    # string only
    avg_str_len: float | None = None
    sample_values: list[Any] = field(default_factory=list)
    # issues detected
    issues: list[str] = field(default_factory=list)


@dataclass
class DataProfile:
    """Full profile of a DataFrame."""
    n_rows: int
    n_cols: int
    memory_mb: float
    duplicate_rows: int
    duplicate_pct: float
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    total_nulls: int = 0
    total_null_pct: float = 0.0

    def get_issues(self) -> dict[str, list[str]]:
        """Return dict of column → list of detected issues."""
        return {
            col: prof.issues
            for col, prof in self.columns.items()
            if prof.issues
        }

    def has_issues(self) -> bool:
        return any(prof.issues for prof in self.columns.values())


class DataProfiler:
    """
    Profiles a DataFrame and surfaces data quality issues.

    Usage:
        profiler = DataProfiler()
        profile = profiler.run(df)
        profiler.print_summary(profile)
        issues = profile.get_issues()
    """

    # Thresholds for issue detection
    HIGH_NULL_THRESH = 0.3       # 30% nulls = warning
    CRITICAL_NULL_THRESH = 0.7   # 70% nulls = critical
    HIGH_OUTLIER_THRESH = 0.05   # 5% outliers = warning
    HIGH_CARDINALITY_THRESH = 0.95  # 95% unique = likely ID column

    def run(self, df: pd.DataFrame) -> DataProfile:
        """Run full profiling pipeline. Returns DataProfile."""
        n_rows, n_cols = df.shape
        memory_mb = round(df.memory_usage(deep=True).sum() / 1_048_576, 6)
        dup_count = int(df.duplicated().sum())
        dup_pct = round(dup_count / max(n_rows, 1) * 100, 2)
        total_nulls = int(df.isnull().sum().sum())
        total_null_pct = round(total_nulls / max(n_rows * n_cols, 1) * 100, 2)

        profile = DataProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            memory_mb=memory_mb,
            duplicate_rows=dup_count,
            duplicate_pct=dup_pct,
            total_nulls=total_nulls,
            total_null_pct=total_null_pct,
        )

        for col in df.columns:
            profile.columns[col] = self._profile_column(df[col], n_rows)

        logger.info(
            "DataProfiler: profiled %d rows × %d cols, found %d issues",
            n_rows, n_cols,
            sum(len(p.issues) for p in profile.columns.values()),
        )
        return profile

    def _profile_column(self, s: pd.Series, n_rows: int) -> ColumnProfile:
        null_count = int(s.isnull().sum())
        null_pct = round(null_count / max(n_rows, 1) * 100, 2)
        unique_count = int(s.nunique())
        cardinality_pct = round(unique_count / max(n_rows, 1) * 100, 2)
        non_null = s.dropna()

        # Sample up to 3 non-null values for display
        sample = non_null.head(3).tolist()

        col_profile = ColumnProfile(
            name=str(s.name),
            dtype=str(s.dtype),
            total_rows=n_rows,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            cardinality_pct=cardinality_pct,
            sample_values=sample,
        )

        # Numeric stats
        if pd.api.types.is_numeric_dtype(s) and len(non_null) > 0:
            q1 = float(non_null.quantile(0.25))
            q3 = float(non_null.quantile(0.75))
            iqr = q3 - q1
            outlier_mask = (non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)
            col_profile.mean = round(float(non_null.mean()), 4)
            col_profile.median = round(float(non_null.median()), 4)
            col_profile.std = round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0
            col_profile.min_val = float(non_null.min())
            col_profile.max_val = float(non_null.max())
            col_profile.outlier_count_iqr = int(outlier_mask.sum())

        # String stats
        elif not pd.api.types.is_numeric_dtype(s) and len(non_null) > 0:
            str_lengths = non_null.astype(str).str.len()
            col_profile.avg_str_len = round(float(str_lengths.mean()), 2)

        # Detect issues
        col_profile.issues = self._detect_issues(col_profile, n_rows)
        return col_profile

    def _detect_issues(self, p: ColumnProfile, n_rows: int) -> list[str]:
        issues = []
        null_frac = p.null_pct / 100

        if null_frac >= self.CRITICAL_NULL_THRESH:
            issues.append(f"CRITICAL: {p.null_pct}% nulls — consider dropping column")
        elif null_frac >= self.HIGH_NULL_THRESH:
            issues.append(f"WARNING: {p.null_pct}% nulls — imputation recommended")

        if p.outlier_count_iqr > 0:
            outlier_pct = p.outlier_count_iqr / max(n_rows, 1)
            if outlier_pct >= self.HIGH_OUTLIER_THRESH:
                issues.append(
                    f"WARNING: {p.outlier_count_iqr} outliers ({outlier_pct*100:.1f}%)"
                )

        if p.cardinality_pct >= self.HIGH_CARDINALITY_THRESH * 100 and n_rows > 10:
            issues.append("INFO: very high cardinality — possible ID column")

        if p.null_count == n_rows:
            issues.append("CRITICAL: column is entirely null")

        return issues

    def print_summary(self, profile: DataProfile) -> None:
        """Print a rich formatted summary table to the terminal."""
        # Header stats
        console.print(f"\n[bold]DataPure Profile[/bold] — "
                      f"{profile.n_rows} rows × {profile.n_cols} cols | "
                      f"{profile.memory_mb} MB | "
                      f"{profile.duplicate_rows} duplicate rows")

        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Column", style="bold")
        table.add_column("Type")
        table.add_column("Nulls %")
        table.add_column("Unique")
        table.add_column("Outliers")
        table.add_column("Issues")

        for col, p in profile.columns.items():
            null_color = (
                "red" if p.null_pct >= 70
                else "yellow" if p.null_pct >= 30
                else "green"
            )
            outliers = str(p.outlier_count_iqr) if p.outlier_count_iqr > 0 else "—"
            issue_text = "; ".join(p.issues) if p.issues else "[green]OK[/green]"

            table.add_row(
                col,
                p.dtype,
                f"[{null_color}]{p.null_pct}%[/{null_color}]",
                str(p.unique_count),
                outliers,
                issue_text,
            )

        console.print(table)
        console.print(
            f"[dim]Total nulls: {profile.total_nulls} ({profile.total_null_pct}%) | "
            f"Issues in {len(profile.get_issues())} column(s)[/dim]\n"
        )
