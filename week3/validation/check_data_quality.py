"""
validation/check_data_quality.py
─────────────────────────────────────────────────────────────────────────────
Data quality validator for the NYC demand-enriched dataset.

Real issues found in demand_enriched_corrupted.parquet (post 2026-01-16):
  1. DUPLICATE ROWS           — 10,085 duplicate (PULocationID, time_bucket) keys
  2. NEGATIVE TRIP_COUNT      — 353 rows with trip_count < 0 (min = -5)
  3. EXTREME OUTLIERS         — trip_count up to 99,999 (baseline max = 310)
  4. CBD_PRICING_ACTIVE=100%  — corrupted window is ALL 1s; baseline is ~33.8%

Usage (CLI, called by GitHub Actions):
    cd week3
    python -m validation.check_data_quality

Usage (library, called by data.py):
    from validation.check_data_quality import DataQualityValidator
    validator = DataQualityValidator(baseline_df=baseline)
    result = validator.validate(corrupted_df)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
CUTOFF = pd.Timestamp("2026-01-16")

# Primary key columns — (PULocationID, time_bucket) must be unique
KEY_COLS = ["PULocationID", "time_bucket"]

# trip_count must be non-negative
TRIP_COUNT_MIN = 0

# Flag outliers more than N sigma above the baseline mean
OUTLIER_SIGMA = 5.0

# Duplicate rate threshold — flag if >1% of rows are duplicates
DUPLICATE_RATE_THRESHOLD = 0.01

# Categorical column: expected rate range (min_rate, max_rate) from baseline
# cbd_pricing_active baseline ≈ 33.8%; flag if corrupted rate deviates >20pp
CATEGORICAL_CHECKS = {
    "cbd_pricing_active": (0.0, 0.60),   # valid range: 0–60%
    "is_holiday":         (0.0, 0.20),   # valid range: 0–20%
}


# ── Helper ─────────────────────────────────────────────────────────────────
def _issue(issue_type: str, severity: str, description: str,
           count: int = 0, **extra: Any) -> dict:
    return {
        "type":        issue_type,
        "severity":    severity,
        "description": description,
        "count":       count,
        **extra,
    }


# ── Check functions ────────────────────────────────────────────────────────

def check_duplicates(df: pd.DataFrame) -> list[dict]:
    """Issue 1 — duplicate (PULocationID, time_bucket) rows."""
    issues = []
    dup_count = int(df.duplicated(subset=KEY_COLS, keep=False).sum())
    dup_rate  = dup_count / len(df) if len(df) > 0 else 0.0

    if dup_rate > DUPLICATE_RATE_THRESHOLD:
        issues.append(_issue(
            "duplicate_rows", "high",
            f"{dup_count:,} rows are duplicates on key {KEY_COLS} "
            f"({dup_rate:.1%} of data). "
            "Duplicate rows inflate aggregated demand and bias model training.",
            count=dup_count,
            duplicate_rate=round(dup_rate, 4),
        ))
    return issues


def check_negative_trip_count(df: pd.DataFrame) -> list[dict]:
    """Issue 2 — trip_count must be >= 0."""
    issues = []
    if "trip_count" not in df.columns:
        return issues

    neg_mask  = df["trip_count"] < TRIP_COUNT_MIN
    neg_count = int(neg_mask.sum())

    if neg_count > 0:
        issues.append(_issue(
            "negative_trip_count", "high",
            f"{neg_count:,} rows have trip_count < 0 "
            f"(min observed = {df['trip_count'].min():.0f}). "
            "Negative demand is physically impossible and will corrupt model predictions.",
            count=neg_count,
            min_value=float(df["trip_count"].min()),
        ))
    return issues


def check_trip_count_outliers(df: pd.DataFrame,
                               baseline_df: pd.DataFrame | None = None) -> list[dict]:
    """Issue 3 — extreme trip_count values far above the baseline range."""
    issues = []
    if "trip_count" not in df.columns:
        return issues

    if baseline_df is not None and "trip_count" in baseline_df.columns:
        b_mean = baseline_df["trip_count"].mean()
        b_std  = baseline_df["trip_count"].std()
        b_max  = float(baseline_df["trip_count"].max())
    else:
        # Fallback: hard cap based on domain knowledge (NYC historical max ~310)
        b_mean, b_std, b_max = 17.0, 21.6, 310.0

    # Use 2× the baseline maximum as the outlier ceiling.
    # This guarantees the baseline always passes its own validation
    # (no baseline value can exceed baseline_max, let alone 2× it),
    # while still catching extreme corruption like the observed max of 99,999.
    upper_bound = b_max * 2.0
    out_mask    = df["trip_count"] > upper_bound
    out_count   = int(out_mask.sum())

    if out_count > 0:
        issues.append(_issue(
            "trip_count_outliers", "high",
            f"{out_count:,} rows have trip_count > {upper_bound:.0f} "
            f"(2× baseline max of {b_max:.0f}). "
            f"Max observed: {df['trip_count'].max():.0f}. "
            "Values this extreme corrupt lag/rolling features for the affected zones "
            "and dominate model loss during retraining.",
            count=out_count,
            upper_bound=round(upper_bound, 2),
            observed_max=float(df["trip_count"].max()),
            baseline_mean=round(b_mean, 2),
            baseline_std=round(b_std, 2),
        ))
    return issues


def check_categorical_rates(df: pd.DataFrame,
                              baseline_df: pd.DataFrame | None = None) -> list[dict]:
    """Issue 4 — binary flag columns must stay within expected rate ranges."""
    issues = []
    for col, (rate_min, rate_max) in CATEGORICAL_CHECKS.items():
        if col not in df.columns:
            continue
        rate = float(df[col].mean())
        if not (rate_min <= rate <= rate_max):
            baseline_rate = float(baseline_df[col].mean()) if (
                baseline_df is not None and col in baseline_df.columns
            ) else None
            b_str = f" (baseline was {baseline_rate:.1%})" if baseline_rate is not None else ""
            issues.append(_issue(
                "categorical_rate_anomaly", "medium",
                f"Column '{col}' rate = {rate:.1%}{b_str}, "
                f"outside expected range [{rate_min:.0%}, {rate_max:.0%}]. "
                f"A constant or near-constant flag adds no signal and may indicate "
                f"upstream pipeline corruption.",
                count=len(df),
                column=col,
                observed_rate=round(rate, 4),
                expected_range=(rate_min, rate_max),
                baseline_rate=round(baseline_rate, 4) if baseline_rate is not None else None,
            ))
    return issues


# ── Main validator ─────────────────────────────────────────────────────────

class DataQualityValidator:
    """
    Orchestrates all quality checks.

    Parameters
    ----------
    baseline_df : optional reference window (pre-cutoff data) for
                  comparison-based checks (outliers, distribution shift).
                  If None, checks fall back to hard-coded domain thresholds.
    """

    def __init__(self, baseline_df: pd.DataFrame | None = None):
        self.baseline = baseline_df

    def validate(self, df: pd.DataFrame,
                 baseline_df: pd.DataFrame | None = None) -> dict:
        """
        Run all checks.

        Returns
        -------
        {
            'is_valid':  bool,
            'num_issues': int,
            'issues':    list[dict],
            'summary':   str,
        }
        """
        ref = baseline_df if baseline_df is not None else self.baseline
        all_issues: list[dict] = []

        all_issues.extend(check_duplicates(df))
        all_issues.extend(check_negative_trip_count(df))
        all_issues.extend(check_trip_count_outliers(df, baseline_df=ref))
        all_issues.extend(check_categorical_rates(df, baseline_df=ref))

        is_valid = len(all_issues) == 0
        summary = (
            "Data quality check PASSED — no issues found."
            if is_valid
            else f"Data quality check FAILED — {len(all_issues)} issue(s): "
                 + ", ".join(i["type"] for i in all_issues)
        )

        return {
            "is_valid":   is_valid,
            "num_issues": len(all_issues),
            "issues":     all_issues,
            "summary":    summary,
        }


# ── CLI entry point ────────────────────────────────────────────────────────

def _run_cli() -> int:
    """
    Called by GitHub Actions:
        cd week3 && python -m validation.check_data_quality
    Returns exit code 0 (pass) or 1 (fail).
    """
    data_path = Path(__file__).parent.parent / "data" / "demand_enriched_corrupted.parquet"

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        return 1

    print(f"Loading {data_path} …")
    df        = pd.read_parquet(data_path)
    baseline  = df[df["time_bucket"] < CUTOFF].copy()
    corrupted = df[df["time_bucket"] >= CUTOFF].copy()

    print(f"  Baseline rows : {len(baseline):,}")
    print(f"  Corrupted rows: {len(corrupted):,}")

    validator = DataQualityValidator(baseline_df=baseline)
    result    = validator.validate(corrupted)

    print(f"\n{result['summary']}")
    if not result["is_valid"]:
        print("\nIssues:")
        for issue in result["issues"]:
            print(f"  [{issue['severity'].upper():8s}] {issue['type']:30s}: {issue['description']}")
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(_run_cli())
