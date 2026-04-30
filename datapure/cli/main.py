"""
datapure CLI — clean data files from the terminal.

Commands:
    datapure clean   input.csv --output clean.csv --missing median
    datapure profile input.csv
    datapure suggest input.csv
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import click
from rich import box
from rich.console import Console
from rich.panel import Panel

console = Console()


# ── Root group ───────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0", prog_name="datapure")
def cli():
    """
    \b
    ██████╗  █████╗ ████████╗ █████╗ ██████╗ ██╗   ██╗██████╗ ███████╗
    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██║   ██║██╔══██╗██╔════╝
    ██║  ██║███████║   ██║   ███████║██████╔╝██║   ██║██████╔╝█████╗
    ██║  ██║██╔══██║   ██║   ██╔══██║██╔═══╝ ██║   ██║██╔══██╗██╔══╝
    ██████╔╝██║  ██║   ██║   ██║  ██║██║     ╚██████╔╝██║  ██║███████╗
    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝

    AI-powered data cleaning for data scientists.
    """


# ── datapure clean ───────────────────────────────────────────────

@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    default=None,
    help="Output file path. Default: input_clean.csv",
)
@click.option(
    "--missing", "-m",
    type=click.Choice([
        "median", "mean", "mode", "ffill", "bfill",
        "drop_rows", "drop_cols", "constant", "knn",
    ]),
    default="median",
    show_default=True,
    help="Strategy for filling missing values.",
)
@click.option(
    "--outliers", "-l",
    type=click.Choice(["iqr", "zscore", "isolation_forest", "none"]),
    default="iqr",
    show_default=True,
    help="Outlier detection method. Use 'none' to skip.",
)
@click.option(
    "--outlier-action",
    type=click.Choice(["winsorize", "remove"]),
    default="winsorize",
    show_default=True,
    help="What to do with outliers.",
)
@click.option(
    "--duplicates/--no-duplicates",
    default=True,
    show_default=True,
    help="Remove exact duplicate rows.",
)
@click.option(
    "--schema/--no-schema",
    default=True,
    show_default=True,
    help="Normalize string columns and coerce types.",
)
@click.option(
    "--text/--no-text",
    default=False,
    show_default=True,
    help="Clean text columns (fix encoding, strip HTML).",
)
@click.option(
    "--ai",
    is_flag=True,
    default=False,
    help="Use Claude AI to suggest and apply cleaning steps.",
)
@click.option(
    "--ai-confidence",
    type=float,
    default=0.75,
    show_default=True,
    help="Minimum AI suggestion confidence to apply (0.0-1.0).",
)
@click.option(
    "--report/--no-report",
    default=False,
    help="Generate an HTML cleaning report.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["csv", "parquet", "json", "excel"]),
    default=None,
    help="Output format. Inferred from extension if not set.",
)
@click.option(
    "--threshold",
    type=float,
    default=100.0,
    show_default=True,
    help="File size (MB) above which Polars engine is used.",
)
def clean(
    input_file, output, missing, outliers, outlier_action,
    duplicates, schema, text, ai, ai_confidence,
    report, output_format, threshold,
):
    """
    Clean a data file and save the result.

    \b
    Examples:
        datapure clean sales.csv
        datapure clean sales.csv --output clean.csv --missing median
        datapure clean sales.csv --outliers iqr --report
        datapure clean sales.csv --ai --ai-confidence 0.8
        datapure clean big_file.csv --format parquet
    """
    from datapure.core.report import ReportGenerator
    from datapure.io.loader import DataLoader
    from datapure.io.writer import DataWriter

    input_path = Path(input_file)

    # Default output path
    if output is None:
        output = str(input_path.parent / f"{input_path.stem}_clean{input_path.suffix}")
    output_path = Path(output)

    console.print(Panel(
        f"[bold cyan]Input:[/bold cyan]  {input_path}\n"
        f"[bold cyan]Output:[/bold cyan] {output_path}",
        title="datapure clean",
        box=box.ROUNDED,
    ))

    # Load data
    with console.status("[bold]Loading file...[/bold]"):
        loader = DataLoader(size_threshold_mb=threshold)
        data = loader.load(input_path)

        # Convert Polars to pandas for pipeline (Polars path used for loading only)
        import polars as pl
        if isinstance(data, pl.LazyFrame):
            console.print("[yellow]Large file detected — using Polars engine[/yellow]")
            df = data.collect().to_pandas()
        else:
            df = data

    df_original = df.copy()
    console.print(f"[green]✓[/green] Loaded {len(df):,} rows × {len(df.columns)} cols")

    # AI mode
    if ai:
        pipeline = _build_ai_pipeline(df, ai_confidence)
    else:
        pipeline = _build_manual_pipeline(
            missing, outliers, outlier_action, duplicates, schema, text
        )

    # Run pipeline
    df_clean = pipeline.run(df)
    pipeline.print_summary()

    # Save output
    with console.status("[bold]Saving output...[/bold]"):
        writer = DataWriter()
        out = writer.write(
            df_clean,
            output_path,
            fmt=output_format,
            audit_log=pipeline.get_audit_log(),
        )
    console.print(f"[green]✓[/green] Saved to [bold]{out}[/bold]")

    # HTML report
    if report:
        report_path = output_path.parent / f"{output_path.stem}_report.html"
        with console.status("[bold]Generating report...[/bold]"):
            ReportGenerator().generate(
                df_before=df_original,
                df_after=df_clean,
                audit_log=pipeline.get_audit_log(),
                output_path=str(report_path),
            )
        console.print(f"[green]✓[/green] Report saved to [bold]{report_path}[/bold]")

    # Final summary
    rows_removed = len(df) - len(df_clean)
    nulls_fixed = sum(s["nulls_fixed"] for s in pipeline.get_audit_log())
    console.print(Panel(
        f"[green]Rows removed:[/green]  {rows_removed:,}\n"
        f"[green]Nulls fixed:[/green]   {nulls_fixed:,}\n"
        f"[green]Final shape:[/green]   {len(df_clean):,} rows × {len(df_clean.columns)} cols",
        title="[bold green]Done![/bold green]",
        box=box.ROUNDED,
    ))


# ── datapure profile ─────────────────────────────────────────────

@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--json-output", "-j",
    default=None,
    help="Save profile as JSON to this path.",
)
def profile(input_file, json_output):
    """
    Profile a data file and show quality statistics.

    \b
    Examples:
        datapure profile sales.csv
        datapure profile sales.csv --json-output profile.json
    """
    from datapure.core.profiler import DataProfiler
    from datapure.io.loader import DataLoader

    input_path = Path(input_file)

    with console.status("[bold]Loading and profiling...[/bold]"):
        df = DataLoader().load(input_path, force_pandas=True)
        profiler = DataProfiler()
        data_profile = profiler.run(df)

    profiler.print_summary(data_profile)

    if json_output:
        _save_profile_json(data_profile, json_output)
        console.print(f"[green]✓[/green] Profile saved to [bold]{json_output}[/bold]")


# ── datapure suggest ─────────────────────────────────────────────

@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Apply suggestions automatically after showing them.",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output path when --apply is used.",
)
@click.option(
    "--confidence", "-c",
    type=float,
    default=0.75,
    show_default=True,
    help="Minimum confidence threshold (0.0-1.0).",
)
def suggest(input_file, apply, output, confidence):
    """
    Use Claude AI to analyze a file and suggest cleaning steps.

    \b
    Examples:
        datapure suggest sales.csv
        datapure suggest sales.csv --apply --output clean.csv
        datapure suggest sales.csv --confidence 0.9
    """
    import os

    from datapure.ai.suggester import AISuggester
    from datapure.io.loader import DataLoader
    from datapure.io.writer import DataWriter
    if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-"):
        console.print(
            "[red]Error:[/red] ANTHROPIC_API_KEY not set.\n"
            "Set it with: [bold]$env:ANTHROPIC_API_KEY = 'sk-ant-...'[/bold]"
        )
        sys.exit(1)

    input_path = Path(input_file)

    with console.status("[bold]Loading file...[/bold]"):
        df = DataLoader().load(input_path, force_pandas=True)

    console.print(f"[green]✓[/green] Loaded {len(df):,} rows × {len(df.columns)} cols")

    with console.status("[bold cyan]Asking Claude AI for suggestions...[/bold cyan]"):
        suggester = AISuggester()
        plan = suggester.suggest(df)

    plan.print_plan()

    if apply:
        output_path = output or str(
            input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
        )
        pipeline = suggester.build_pipeline(plan, min_confidence=confidence)
        df_clean = pipeline.run(df)
        DataWriter().write(
            df_clean, output_path, audit_log=pipeline.get_audit_log()
        )
        console.print(f"[green]✓[/green] Saved to [bold]{output_path}[/bold]")


# ── helpers ───────────────────────────────────────────────────────

def _build_manual_pipeline(
    missing, outliers, outlier_action, duplicates, schema, text
):
    """Build a Pipeline from CLI flags."""
    from datapure.cleaners.duplicates import DuplicateCleaner
    from datapure.cleaners.missing import MissingValueCleaner
    from datapure.cleaners.outliers import OutlierCleaner
    from datapure.cleaners.schema import SchemaCleaner
    from datapure.cleaners.text import TextCleaner
    from datapure.core.pipeline import Pipeline

    pipeline = Pipeline(name="Manual Pipeline")
    pipeline.add(MissingValueCleaner(strategy=missing))

    if outliers != "none":
        pipeline.add(OutlierCleaner(method=outliers, action=outlier_action))

    if duplicates:
        pipeline.add(DuplicateCleaner(mode="exact"))

    if schema:
        pipeline.add(SchemaCleaner(normalize_strings=True))

    if text:
        pipeline.add(TextCleaner())

    return pipeline


def _build_ai_pipeline(df, ai_confidence):
    """Build a Pipeline using AI suggestions."""
    from datapure.ai.suggester import AISuggester

    with console.status("[bold cyan]Asking Claude AI...[/bold cyan]"):
        suggester = AISuggester()
        try:
            plan = suggester.suggest(df)
            plan.print_plan()
            return suggester.build_pipeline(plan, min_confidence=ai_confidence)
        except OSError as e:
            console.print(f"[red]AI Error:[/red] {e}")
            console.print("[yellow]Falling back to default pipeline...[/yellow]")
            return _build_manual_pipeline("median", "iqr", "winsorize", True, True, False)


def _save_profile_json(data_profile, path: str) -> None:
    """Save a DataProfile to JSON."""
    data = {
        "shape": {"rows": data_profile.n_rows, "cols": data_profile.n_cols},
        "memory_mb": data_profile.memory_mb,
        "duplicate_rows": data_profile.duplicate_rows,
        "total_nulls": data_profile.total_nulls,
        "columns": {
            col: {
                "dtype": p.dtype,
                "null_count": p.null_count,
                "null_pct": p.null_pct,
                "unique_count": p.unique_count,
                "issues": p.issues,
            }
            for col, p in data_profile.columns.items()
        },
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
