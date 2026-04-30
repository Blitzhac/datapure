# datapure

AI-powered data cleaning library for data scientists.
Clean GB+ files with a single command or a fluent Python API.

## Install

```bash
pip install datapure
```

## CLI — clean a file in one command

```bash
# Basic cleaning
datapure clean sales.csv

# Full options
datapure clean sales.csv \
  --output clean.csv \
  --missing median \
  --outliers iqr \
  --report

# AI-powered cleaning (requires ANTHROPIC_API_KEY)
datapure clean sales.csv --ai

# Profile a file before cleaning
datapure profile sales.csv

# Get AI suggestions without applying
datapure suggest sales.csv
```

## Python API

```python
import pandas as pd
from datapure.core import Pipeline, DataProfiler
from datapure.cleaners import (
    MissingValueCleaner,
    DuplicateCleaner,
    OutlierCleaner,
    SchemaCleaner,
    TextCleaner,
)

df = pd.read_csv("sales.csv")

# Profile first
profiler = DataProfiler()
profile = profiler.run(df)
profiler.print_summary(profile)

# Build and run pipeline
df_clean = (
    Pipeline()
    .add(MissingValueCleaner(strategy="median", col_strategy={"city": "mode"}))
    .add(OutlierCleaner(method="iqr", action="winsorize"))
    .add(DuplicateCleaner(mode="exact"))
    .add(SchemaCleaner(normalize_strings=True))
    .run(df)
)

# Auto-pipeline from profile
pipeline = Pipeline.from_profile(profile)
df_clean = pipeline.run(df)
```

## AI Suggestions

```python
import os
from datapure.ai import AISuggester

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

suggester = AISuggester()
plan = suggester.suggest(df)
plan.print_plan()

pipeline = suggester.build_pipeline(plan, min_confidence=0.8)
df_clean = pipeline.run(df)
```

## Large Files (GB+)

```python
from datapure.io import DataLoader, DataWriter
from datapure.cleaners.polars_missing import PolarsNativeMissingCleaner

# Auto-switches to Polars for files > 100 MB
lf = DataLoader().load("big_file.csv")  # returns pl.LazyFrame

cleaner = PolarsNativeMissingCleaner(strategy="median")
lf_clean = cleaner.clean(lf)

DataWriter().write_polars(lf_clean, "big_file_clean.parquet")
```

## Cleaners

| Cleaner | What it fixes | Key options |
|---|---|---|
| `MissingValueCleaner` | Nulls / NaN | `strategy`: median, mean, mode, knn, ffill, constant |
| `DuplicateCleaner` | Duplicate rows | `mode`: exact, subset, fuzzy |
| `OutlierCleaner` | Extreme values | `method`: iqr, zscore, isolation_forest |
| `SchemaCleaner` | Types, dates, emails | `date_columns`, `email_columns`, `bool_columns` |
| `TextCleaner` | Encoding, HTML, whitespace | `fix_encoding`, `strip_html`, `remove_urls` |

## Project structure

```
datapure/
├── cleaners/      # All 5 cleaners + Polars native cleaner
├── core/          # Pipeline, DataProfiler, ReportGenerator
├── ai/            # Claude API integration
├── io/            # Smart loader and writer
└── cli/           # Click + Rich terminal interface
```