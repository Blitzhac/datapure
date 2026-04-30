"""
AISuggester — sends data profile to Claude API and returns
a structured cleaning plan with confidence scores.

Requires ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os

import pandas as pd

from datapure.ai.sampler import DataSampler

logger = logging.getLogger(__name__)

# Prompt sent to Claude — structured to get JSON back
_SYSTEM_PROMPT = """You are a data quality expert helping data scientists
clean their datasets. You will receive a JSON profile of a DataFrame and
must respond with a structured cleaning plan in valid JSON only.

Your response must be ONLY a JSON object — no markdown, no explanation,
no code fences. Just the raw JSON object.

The JSON must follow this exact structure:
{
  "summary": "One sentence description of the main data quality issues",
  "overall_quality": "poor|fair|good|excellent",
  "suggestions": [
    {
      "column": "column_name or __all__ for whole-dataframe operations",
      "issue": "short description of the problem",
      "cleaner": "MissingValueCleaner|DuplicateCleaner|OutlierCleaner|SchemaCleaner|TextCleaner",
      "strategy": "the specific strategy or method to use",
      "confidence": 0.95,
      "reason": "one sentence explaining why this fix is appropriate",
      "priority": "high|medium|low"
    }
  ],
  "estimated_rows_at_risk": 0,
  "recommended_pipeline_order": ["cleaner1", "cleaner2"]
}"""

_USER_PROMPT_TEMPLATE = """Here is the profile of my DataFrame:

{profile_json}

Please analyze this data profile and return a cleaning plan following
the exact JSON structure specified. Focus on the most impactful issues.
"""


@dataclass
class CleaningSuggestion:
    """One AI-generated cleaning suggestion."""
    column: str
    issue: str
    cleaner: str
    strategy: str
    confidence: float
    reason: str
    priority: str

    def __str__(self) -> str:
        return (
            f"[{self.priority.upper()}] {self.cleaner} on '{self.column}': "
            f"{self.issue} (confidence: {self.confidence:.0%})"
        )


@dataclass
class CleaningPlan:
    """Full AI-generated cleaning plan for a DataFrame."""
    summary: str
    overall_quality: str
    suggestions: list[CleaningSuggestion]
    estimated_rows_at_risk: int
    recommended_pipeline_order: list[str]

    def high_priority(self) -> list[CleaningSuggestion]:
        return [s for s in self.suggestions if s.priority == "high"]

    def for_column(self, col: str) -> list[CleaningSuggestion]:
        return [s for s in self.suggestions if s.column == col]

    def print_plan(self) -> None:
        """Pretty print the cleaning plan to terminal."""
        print(f"\n{'='*60}")
        print("AI CLEANING PLAN")
        print(f"{'='*60}")
        print(f"Summary:  {self.summary}")
        print(f"Quality:  {self.overall_quality.upper()}")
        print(f"At risk:  ~{self.estimated_rows_at_risk} rows")
        print(f"Order:    {' → '.join(self.recommended_pipeline_order)}")
        print(f"\nSuggestions ({len(self.suggestions)}):")
        for i, s in enumerate(self.suggestions, 1):
            print(f"  {i}. {s}")
            print(f"     Reason: {s.reason}")
        print(f"{'='*60}\n")


class AISuggester:
    """
    Sends a DataFrame profile to Claude API and returns a CleaningPlan.

    Requires ANTHROPIC_API_KEY in environment variables.
    Set it with:
        $env:ANTHROPIC_API_KEY = "your-key-here"   # PowerShell
        os.environ["ANTHROPIC_API_KEY"] = "..."     # Python

    Usage:
        suggester = AISuggester()
        plan = suggester.suggest(df)
        plan.print_plan()

        # Auto-apply high-confidence suggestions
        pipeline = suggester.build_pipeline(plan, min_confidence=0.8)
        df_clean = pipeline.run(df)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1500,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._sampler = DataSampler()

    def suggest(self, df: pd.DataFrame) -> CleaningPlan:
        """
        Analyze a DataFrame and return an AI-generated CleaningPlan.

        Args:
            df: The DataFrame to analyze.

        Returns:
            CleaningPlan with suggestions, confidence scores, and ordering.

        Raises:
            EnvironmentError: If ANTHROPIC_API_KEY is not set.
            RuntimeError: If the API call fails or returns invalid JSON.
        """
        self._check_api_key()

        # Build compact profile — never send full data
        profile = self._sampler.build(df)
        profile_json = json.dumps(profile, indent=2, default=str)

        logger.info(
            "AISuggester: sending profile (%d chars) to Claude API",
            len(profile_json),
        )

        response_text = self._call_api(profile_json)
        plan = self._parse_response(response_text)

        logger.info(
            "AISuggester: received %d suggestions (quality: %s)",
            len(plan.suggestions), plan.overall_quality,
        )
        return plan

    def build_pipeline(
        self,
        plan: CleaningPlan,
        min_confidence: float = 0.75,
        interactive: bool = False,
    ):
        """
        Convert a CleaningPlan into a runnable Pipeline.

        Args:
            plan:            The CleaningPlan from suggest().
            min_confidence:  Only apply suggestions above this threshold.
            interactive:     If True, prompt user to approve each step.

        Returns:
            A configured Pipeline instance.
        """
        from datapure.core.pipeline import Pipeline

        pipeline = Pipeline(name="AI-Suggested Pipeline")
        added: set[str] = set()

        # Apply in recommended order
        ordered = self._order_suggestions(plan)

        for suggestion in ordered:
            if suggestion.confidence < min_confidence:
                logger.info(
                    "Skipping '%s' — confidence %.0f%% below threshold %.0f%%",
                    suggestion.cleaner,
                    suggestion.confidence * 100,
                    min_confidence * 100,
                )
                continue

            if interactive:
                approved = self._prompt_user(suggestion)
                if not approved:
                    continue

            # Avoid adding the same cleaner type twice
            cleaner_key = suggestion.cleaner
            if cleaner_key in added:
                continue

            cleaner = self._build_cleaner(suggestion)
            if cleaner is not None:
                pipeline.add(cleaner)
                added.add(cleaner_key)
                logger.info("Added %s to pipeline", suggestion.cleaner)

        return pipeline

    # ── private helpers ──────────────────────────────────────────

    def _check_api_key(self) -> None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or not key.startswith("sk-"):
            raise OSError(
                "ANTHROPIC_API_KEY not set or invalid. "
                "Set it with: $env:ANTHROPIC_API_KEY = 'sk-ant-...'"
            )

    def _call_api(self, profile_json: str) -> str:
        """Make the Anthropic API call. Returns raw response text."""
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": _USER_PROMPT_TEMPLATE.format(
                        profile_json=profile_json
                    ),
                }
            ],
        )
        return message.content[0].text

    def _parse_response(self, text: str) -> CleaningPlan:
        """Parse Claude's JSON response into a CleaningPlan."""
        # Strip any accidental markdown fences
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Claude returned invalid JSON: {e}\nRaw response:\n{text}"
            ) from e

        suggestions = [
            CleaningSuggestion(
                column=s.get("column", "__all__"),
                issue=s.get("issue", ""),
                cleaner=s.get("cleaner", ""),
                strategy=s.get("strategy", ""),
                confidence=float(s.get("confidence", 0.5)),
                reason=s.get("reason", ""),
                priority=s.get("priority", "medium"),
            )
            for s in data.get("suggestions", [])
        ]

        return CleaningPlan(
            summary=data.get("summary", ""),
            overall_quality=data.get("overall_quality", "unknown"),
            suggestions=suggestions,
            estimated_rows_at_risk=int(
                data.get("estimated_rows_at_risk", 0)
            ),
            recommended_pipeline_order=data.get(
                "recommended_pipeline_order", []
            ),
        )

    def _order_suggestions(
        self, plan: CleaningPlan
    ) -> list[CleaningSuggestion]:
        """Sort suggestions by priority then confidence."""
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            plan.suggestions,
            key=lambda s: (
                priority_order.get(s.priority, 1),
                -s.confidence,
            ),
        )

    def _build_cleaner(self, suggestion: CleaningSuggestion):
        """Map a CleaningSuggestion to a concrete cleaner instance."""
        from datapure.cleaners.duplicates import DuplicateCleaner
        from datapure.cleaners.missing import MissingValueCleaner
        from datapure.cleaners.outliers import OutlierCleaner
        from datapure.cleaners.schema import SchemaCleaner
        from datapure.cleaners.text import TextCleaner

        match suggestion.cleaner:
            case "MissingValueCleaner":
                return MissingValueCleaner(strategy=suggestion.strategy)
            case "DuplicateCleaner":
                return DuplicateCleaner(mode=suggestion.strategy)
            case "OutlierCleaner":
                return OutlierCleaner(method=suggestion.strategy)
            case "SchemaCleaner":
                return SchemaCleaner()
            case "TextCleaner":
                return TextCleaner()
            case _:
                logger.warning(
                    "Unknown cleaner '%s' — skipping", suggestion.cleaner
                )
                return None

    def _prompt_user(self, suggestion: CleaningSuggestion) -> bool:
        """Ask user to approve a suggestion interactively."""
        print(f"\n  Suggestion: {suggestion}")
        print(f"  Reason:     {suggestion.reason}")
        answer = input("  Apply this? [y/n]: ").strip().lower()
        return answer in ("y", "yes", "")
