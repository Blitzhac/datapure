"""
Pipeline — chains multiple cleaners with logging, timing, and audit trail.
Supports manual chaining and auto-build from DataProfile.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from datapure.cleaners.base import BaseCleaner

if TYPE_CHECKING:
    from datapure.core.profiler import DataProfile

logger = logging.getLogger(__name__)
console = Console()


class PipelineStep:
    """Records what happened at each pipeline step."""

    def __init__(self, cleaner: BaseCleaner):
        self.cleaner = cleaner
        self.cleaner_name = cleaner.__class__.__name__
        self.rows_before: int = 0
        self.rows_after: int = 0
        self.nulls_before: int = 0
        self.nulls_after: int = 0
        self.duration_ms: float = 0.0

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def nulls_fixed(self) -> int:
        return self.nulls_before - self.nulls_after


class Pipeline:
    """
    Chains cleaners into a sequential cleaning pipeline.

    Usage (manual):
        pipeline = (
            Pipeline()
            .add(MissingValueCleaner(strategy="median"))
            .add(OutlierCleaner(method="iqr"))
            .add(DuplicateCleaner())
        )
        df_clean = pipeline.run(df)
        pipeline.print_summary()

    Usage (auto from profile):
        profile = DataProfiler().run(df)
        pipeline = Pipeline.from_profile(profile)
        df_clean = pipeline.run(df)
    """

    def __init__(self, name: str = "DataPure Pipeline") -> None:
        self.name = name
        self._cleaners: list[BaseCleaner] = []
        self._steps: list[PipelineStep] = []

    def add(self, cleaner: BaseCleaner) -> Pipeline:
        """Add a cleaner to the pipeline. Returns self for chaining."""
        if not isinstance(cleaner, BaseCleaner):
            raise TypeError(
                f"Expected a BaseCleaner subclass, got {type(cleaner).__name__}"
            )
        self._cleaners.append(cleaner)
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all cleaners in sequence.
        Returns cleaned DataFrame and stores audit trail in self._steps.
        """
        if not self._cleaners:
            logger.warning("Pipeline has no cleaners — returning original DataFrame")
            return df.copy()

        self._steps = []
        current_df = df.copy()

        console.print(f"\n[bold cyan]▶ {self.name}[/bold cyan] "
                      f"— {len(self._cleaners)} cleaner(s), "
                      f"{len(df)} rows × {len(df.columns)} cols\n")

        for cleaner in self._cleaners:
            step = PipelineStep(cleaner)
            step.rows_before = len(current_df)
            step.nulls_before = int(current_df.isnull().sum().sum())

            t0 = time.perf_counter()
            try:
                current_df = cleaner.clean(current_df)
            except Exception as e:
                logger.error(
                    "Cleaner %s failed: %s — skipping",
                    cleaner.__class__.__name__, e,
                )
                console.print(
                    f"  [red]✗ {cleaner.__class__.__name__} failed: {e}[/red]"
                )
                continue

            step.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            step.rows_after = len(current_df)
            step.nulls_after = int(current_df.isnull().sum().sum())
            self._steps.append(step)

            console.print(
                f"  [green]✓[/green] {step.cleaner_name:<30} "
                f"rows: {step.rows_before}→{step.rows_after} "
                f"({'-' if step.rows_removed == 0 else '-' + str(step.rows_removed)}) | "
                f"nulls fixed: {step.nulls_fixed} | "
                f"{step.duration_ms}ms"
            )

        logger.info(
            "Pipeline complete: %d→%d rows, %d nulls fixed",
            len(df), len(current_df),
            sum(s.nulls_fixed for s in self._steps),
        )
        return current_df

    def print_summary(self) -> None:
        """Print a rich table summary of all pipeline steps."""
        if not self._steps:
            console.print("[yellow]No steps recorded — run the pipeline first.[/yellow]")
            return

        table = Table(
            title=f"{self.name} — Summary",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Step", style="bold")
        table.add_column("Cleaner")
        table.add_column("Rows removed")
        table.add_column("Nulls fixed")
        table.add_column("Time (ms)")

        for i, step in enumerate(self._steps, 1):
            table.add_row(
                str(i),
                step.cleaner_name,
                str(step.rows_removed),
                str(step.nulls_fixed),
                str(step.duration_ms),
            )

        console.print(table)
        console.print(
            f"[dim]Total rows removed: "
            f"{sum(s.rows_removed for s in self._steps)} | "
            f"Total nulls fixed: "
            f"{sum(s.nulls_fixed for s in self._steps)}[/dim]\n"
        )

    def get_audit_log(self) -> list[dict]:
        """Return audit trail as a list of dicts — useful for saving to JSON."""
        return [
            {
                "step": i + 1,
                "cleaner": s.cleaner_name,
                "config": s.cleaner.get_config(),
                "rows_before": s.rows_before,
                "rows_after": s.rows_after,
                "rows_removed": s.rows_removed,
                "nulls_before": s.nulls_before,
                "nulls_after": s.nulls_after,
                "nulls_fixed": s.nulls_fixed,
                "duration_ms": s.duration_ms,
            }
            for i, s in enumerate(self._steps)
        ]

    @classmethod
    def from_profile(cls, profile: DataProfile) -> Pipeline:
        """
        Auto-build a pipeline from a DataProfile.
        Inspects issues found by DataProfiler and selects
        appropriate cleaners with sensible defaults.
        """
        from datapure.cleaners.duplicates import DuplicateCleaner
        from datapure.cleaners.missing import MissingValueCleaner
        from datapure.cleaners.outliers import OutlierCleaner
        from datapure.cleaners.schema import SchemaCleaner

        pipeline = cls(name="Auto Pipeline")
        col_strategies: dict[str, str] = {}
        needs_outlier = False
        needs_drop_cols = False

        for col, col_profile in profile.columns.items():
            null_frac = col_profile.null_pct / 100

            # Entirely null column — mark for dropping
            if null_frac >= 0.99:
                needs_drop_cols = True
                continue

            # High null — pick best fill strategy
            if null_frac >= 0.3:
                if col_profile.dtype == "object":
                    col_strategies[col] = "mode"
                else:
                    col_strategies[col] = "median"

            # Outlier issues
            if col_profile.outlier_count_iqr > 0:
                outlier_pct = col_profile.outlier_count_iqr / max(profile.n_rows, 1)
                if outlier_pct >= 0.05:
                    needs_outlier = True

        # Build pipeline in logical order
        if needs_drop_cols:
            pipeline.add(MissingValueCleaner(strategy="drop_cols", threshold=0.99))

        if col_strategies or any(
            p.null_count > 0 for p in profile.columns.values()
        ):
            pipeline.add(MissingValueCleaner(
                strategy="median",
                col_strategy=col_strategies,
            ))

        if needs_outlier:
            pipeline.add(OutlierCleaner(method="iqr", action="winsorize"))

        pipeline.add(DuplicateCleaner(mode="exact"))
        pipeline.add(SchemaCleaner(normalize_strings=True))

        console.print(
            f"[bold]Auto Pipeline[/bold] built "
            f"{len(pipeline._cleaners)} cleaner(s) from profile"
        )
        return pipeline
