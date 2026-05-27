"""
validation/test_data_quality.py
─────────────────────────────────────────────────────────────────────────────
Tests for DataQualityValidator.

Run from repo root:
    python -m pytest week3/validation/test_data_quality.py -v

Design notes
────────────
- Fixtures build SYNTHETIC DataFrames that reproduce each known issue exactly.
  This lets the tests run in CI without the real parquet file.
- The "real_baseline / real_corrupted" fixtures load from the actual parquet
  (skipped automatically if the file is absent — safe for CI with no data).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation.check_data_quality import (
    CUTOFF,
    DataQualityValidator,
    check_duplicates,
    check_negative_trip_count,
    check_trip_count_outliers,
    check_categorical_rates,
)

# ── Path to real data (optional — tests skip if absent) ───────────────────
_DATA_PATH = Path(__file__).parent.parent / "data" / "demand_enriched_corrupted.parquet"
try:
    import pyarrow.parquet as _pq
    if _DATA_PATH.exists():
        _pq.read_schema(_DATA_PATH)
    _HAS_REAL_DATA = _DATA_PATH.exists()
except Exception:
    _HAS_REAL_DATA = False


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

def _make_clean(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Well-formed DataFrame — should pass all checks."""
    rng  = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01")
    times = [base + pd.Timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame({
        "PULocationID":      rng.integers(1, 263, size=n),
        "time_bucket":       times,
        "trip_count":        rng.integers(0, 200, size=n).astype(float),
        "cbd_pricing_active": rng.choice([0, 1], size=n, p=[0.66, 0.34]).astype(np.int8),
        "is_holiday":        rng.choice([0, 1], size=n, p=[0.96, 0.04]).astype(np.int8),
    })


@pytest.fixture()
def clean_df():
    return _make_clean()


@pytest.fixture()
def baseline_df():
    return _make_clean(n=500, seed=0)


# ═════════════════════════════════════════════════════════════════════════════
# REAL-DATA FIXTURES (skip if parquet absent)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def real_baseline():
    if not _HAS_REAL_DATA:
        pytest.skip("Real parquet not available — skipping real-data test")
    df = pd.read_parquet(_DATA_PATH)
    return df[df["time_bucket"] < CUTOFF].copy()


@pytest.fixture(scope="session")
def real_corrupted():
    if not _HAS_REAL_DATA:
        pytest.skip("Real parquet not available — skipping real-data test")
    df = pd.read_parquet(_DATA_PATH)
    return df[df["time_bucket"] >= CUTOFF].copy()


# ═════════════════════════════════════════════════════════════════════════════
# 1 · BASELINE DATA PASSES
# ═════════════════════════════════════════════════════════════════════════════

class TestBaselinePasses:
    def test_clean_is_valid(self, clean_df, baseline_df):
        result = DataQualityValidator(baseline_df).validate(clean_df)
        assert result["is_valid"], f"Baseline should pass, got: {result['issues']}"

    def test_clean_zero_issues(self, clean_df, baseline_df):
        result = DataQualityValidator(baseline_df).validate(clean_df)
        assert result["num_issues"] == 0

    def test_no_duplicates_in_clean(self, clean_df):
        # clean_df has unique (PULocationID, time_bucket) by construction
        issues = check_duplicates(clean_df)
        assert len(issues) == 0

    def test_no_negatives_in_clean(self, clean_df):
        issues = check_negative_trip_count(clean_df)
        assert len(issues) == 0

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="No parquet file")
    def test_real_baseline_passes(self, real_baseline):
        """Real pre-cutoff data should have no issues."""
        result = DataQualityValidator(real_baseline).validate(real_baseline)
        assert result["is_valid"], f"Real baseline failed: {result['issues']}"


# ═════════════════════════════════════════════════════════════════════════════
# 2 · ISSUE 1 — DUPLICATE ROWS
# ═════════════════════════════════════════════════════════════════════════════

class TestDuplicateDetection:
    def _inject_dups(self, df, frac=0.15):
        extra = df.sample(frac=frac, random_state=7)
        return pd.concat([df, extra], ignore_index=True)

    def test_detects_duplicates(self, clean_df):
        corrupted = self._inject_dups(clean_df)
        issues = check_duplicates(corrupted)
        assert len(issues) > 0

    def test_duplicate_issue_type(self, clean_df):
        corrupted = self._inject_dups(clean_df)
        issues = check_duplicates(corrupted)
        assert issues[0]["type"] == "duplicate_rows"

    def test_duplicate_severity_is_high(self, clean_df):
        corrupted = self._inject_dups(clean_df)
        issues = check_duplicates(corrupted)
        assert issues[0]["severity"] == "high"

    def test_duplicate_count_correct(self, clean_df):
        extra = clean_df.sample(frac=0.20, random_state=7)
        corrupted = pd.concat([clean_df, extra], ignore_index=True)
        issues = check_duplicates(corrupted)
        expected_dup_count = int(
            corrupted.duplicated(subset=["PULocationID", "time_bucket"], keep=False).sum()
        )
        assert issues[0]["count"] == expected_dup_count

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="No parquet file")
    def test_real_corrupted_has_duplicates(self, real_corrupted):
        """Real corrupted window should have 10,085 duplicate rows."""
        issues = check_duplicates(real_corrupted)
        assert len(issues) > 0
        assert issues[0]["count"] >= 10_000


# ═════════════════════════════════════════════════════════════════════════════
# 3 · ISSUE 2 — NEGATIVE TRIP_COUNT
# ═════════════════════════════════════════════════════════════════════════════

class TestNegativeTripCount:
    def test_detects_negatives(self, clean_df):
        corrupted = clean_df.copy()
        corrupted.loc[:9, "trip_count"] = -5
        issues = check_negative_trip_count(corrupted)
        assert len(issues) > 0

    def test_negative_issue_type(self, clean_df):
        corrupted = clean_df.copy()
        corrupted.loc[:4, "trip_count"] = -3
        issues = check_negative_trip_count(corrupted)
        assert issues[0]["type"] == "negative_trip_count"

    def test_negative_count_correct(self, clean_df):
        corrupted = clean_df.copy()
        corrupted.loc[:14, "trip_count"] = -1
        issues = check_negative_trip_count(corrupted)
        assert issues[0]["count"] == 15

    def test_zero_is_valid(self, clean_df):
        """trip_count = 0 is allowed (no demand in that slot)."""
        with_zeros = clean_df.copy()
        with_zeros.loc[:9, "trip_count"] = 0
        issues = check_negative_trip_count(with_zeros)
        assert len(issues) == 0

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="No parquet file")
    def test_real_corrupted_has_negatives(self, real_corrupted):
        """Real corrupted window should have 353 negative trip_count rows."""
        issues = check_negative_trip_count(real_corrupted)
        assert len(issues) > 0
        assert issues[0]["count"] >= 300


# ═════════════════════════════════════════════════════════════════════════════
# 4 · ISSUE 3 — EXTREME OUTLIERS IN TRIP_COUNT
# ═════════════════════════════════════════════════════════════════════════════

class TestTripCountOutliers:
    def test_detects_extreme_outlier(self, clean_df, baseline_df):
        corrupted = clean_df.copy()
        corrupted.loc[:4, "trip_count"] = 99_999
        issues = check_trip_count_outliers(corrupted, baseline_df=baseline_df)
        assert len(issues) > 0

    def test_outlier_issue_type(self, clean_df, baseline_df):
        corrupted = clean_df.copy()
        corrupted.loc[:4, "trip_count"] = 50_000
        issues = check_trip_count_outliers(corrupted, baseline_df=baseline_df)
        assert issues[0]["type"] == "trip_count_outliers"

    def test_outlier_severity_is_high(self, clean_df, baseline_df):
        corrupted = clean_df.copy()
        corrupted.loc[:4, "trip_count"] = 50_000
        issues = check_trip_count_outliers(corrupted, baseline_df=baseline_df)
        assert issues[0]["severity"] == "high"

    def test_normal_high_value_not_flagged(self, clean_df, baseline_df):
        """A value at exactly baseline_max should NOT be flagged (threshold is 2× max)."""
        b_max = float(baseline_df["trip_count"].max())
        within_range = clean_df.copy()
        within_range.loc[:4, "trip_count"] = b_max   # exactly at max, well below 2× max
        issues = check_trip_count_outliers(within_range, baseline_df=baseline_df)
        assert len(issues) == 0

    def test_works_without_baseline(self, clean_df):
        """Falls back to hard-coded thresholds when no baseline provided."""
        corrupted = clean_df.copy()
        corrupted.loc[:4, "trip_count"] = 99_999
        issues = check_trip_count_outliers(corrupted, baseline_df=None)
        assert len(issues) > 0

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="No parquet file")
    def test_real_corrupted_has_outliers(self, real_corrupted, real_baseline):
        """Real corrupted window: max=99,999, well above baseline max=310."""
        issues = check_trip_count_outliers(real_corrupted, baseline_df=real_baseline)
        assert len(issues) > 0
        assert issues[0]["observed_max"] >= 99_000


# ═════════════════════════════════════════════════════════════════════════════
# 5 · ISSUE 4 — CBD_PRICING_ACTIVE STUCK AT 100%
# ═════════════════════════════════════════════════════════════════════════════

class TestCategoricalRates:
    def test_detects_cbd_always_one(self, clean_df, baseline_df):
        corrupted = clean_df.copy()
        corrupted["cbd_pricing_active"] = 1   # always active — impossible
        issues = check_categorical_rates(corrupted, baseline_df=baseline_df)
        types = [i["type"] for i in issues]
        assert "categorical_rate_anomaly" in types

    def test_cbd_column_flagged(self, clean_df, baseline_df):
        corrupted = clean_df.copy()
        corrupted["cbd_pricing_active"] = 1
        issues = check_categorical_rates(corrupted, baseline_df=baseline_df)
        cbd_issues = [i for i in issues if i.get("column") == "cbd_pricing_active"]
        assert len(cbd_issues) > 0

    def test_normal_cbd_rate_passes(self, clean_df, baseline_df):
        """cbd_pricing_active ≈ 34% should NOT be flagged."""
        issues = check_categorical_rates(clean_df, baseline_df=baseline_df)
        cbd_issues = [i for i in issues if i.get("column") == "cbd_pricing_active"]
        assert len(cbd_issues) == 0

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="No parquet file")
    def test_real_corrupted_cbd_is_100_pct(self, real_corrupted, real_baseline):
        """Real corrupted window: cbd_pricing_active = 100%."""
        issues = check_categorical_rates(real_corrupted, baseline_df=real_baseline)
        cbd_issues = [i for i in issues if i.get("column") == "cbd_pricing_active"]
        assert len(cbd_issues) > 0
        assert cbd_issues[0]["observed_rate"] == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 6 · FULL CORRUPTED DATASET FAILS OVERALL
# ═════════════════════════════════════════════════════════════════════════════

class TestCorruptedDatasetFails:
    def _make_corrupted(self, base: pd.DataFrame) -> pd.DataFrame:
        """Build a synthetic DF that exhibits all 4 issues."""
        df = base.copy()
        # Issue 1: duplicates
        extra = df.sample(frac=0.15, random_state=3)
        df = pd.concat([df, extra], ignore_index=True)
        # Issue 2: negatives
        df.loc[:9, "trip_count"] = -5
        # Issue 3: extreme outliers
        df.loc[10:14, "trip_count"] = 99_999
        # Issue 4: cbd stuck at 1
        df["cbd_pricing_active"] = 1
        return df

    def test_corrupted_is_invalid(self, clean_df, baseline_df):
        corrupted = self._make_corrupted(clean_df)
        result = DataQualityValidator(baseline_df).validate(corrupted)
        assert not result["is_valid"]

    def test_corrupted_detects_at_least_two_issues(self, clean_df, baseline_df):
        corrupted = self._make_corrupted(clean_df)
        result = DataQualityValidator(baseline_df).validate(corrupted)
        assert result["num_issues"] >= 2

    def test_all_four_issue_types_detected(self, clean_df, baseline_df):
        corrupted = self._make_corrupted(clean_df)
        result = DataQualityValidator(baseline_df).validate(corrupted)
        found = {i["type"] for i in result["issues"]}
        assert "duplicate_rows"          in found
        assert "negative_trip_count"     in found
        assert "trip_count_outliers"     in found
        assert "categorical_rate_anomaly" in found

    @pytest.mark.skipif(not _HAS_REAL_DATA, reason="No parquet file")
    def test_real_corrupted_fails(self, real_corrupted, real_baseline):
        """End-to-end: real corrupted window must fail validation."""
        result = DataQualityValidator(real_baseline).validate(real_corrupted)
        assert not result["is_valid"]
        assert result["num_issues"] >= 2


# ═════════════════════════════════════════════════════════════════════════════
# 7 · GRACEFUL DEGRADATION — API NEVER CRASHES
# ═════════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    def test_validator_handles_empty_df(self):
        empty = pd.DataFrame(columns=["PULocationID", "time_bucket",
                                       "trip_count", "cbd_pricing_active"])
        try:
            result = DataQualityValidator().validate(empty)
            assert "is_valid" in result
        except Exception as exc:
            pytest.fail(f"Validator crashed on empty DataFrame: {exc}")

    def test_validator_handles_all_negatives(self, clean_df):
        broken = clean_df.copy()
        broken["trip_count"] = -99
        try:
            result = DataQualityValidator().validate(broken)
            assert not result["is_valid"]
        except Exception as exc:
            pytest.fail(f"Validator crashed on all-negative input: {exc}")

    def test_validator_handles_no_baseline(self, clean_df):
        """Validator must not crash when baseline_df=None."""
        corrupted = clean_df.copy()
        corrupted.loc[:4, "trip_count"] = 99_999
        try:
            result = DataQualityValidator(baseline_df=None).validate(corrupted)
            assert "is_valid" in result
        except Exception as exc:
            pytest.fail(f"Validator crashed without baseline: {exc}")

    def test_check_and_log_does_not_raise(self, tmp_path, caplog):
        """
        Simulates what data.py check_and_log_data_quality() does at startup.
        Must log warnings but never raise.
        """
        # Write a tiny corrupted parquet to a temp dir
        corrupted = _make_clean(n=50)
        corrupted.loc[:4, "trip_count"] = 99_999
        pq_path = tmp_path / "demand_enriched_corrupted.parquet"
        corrupted.to_parquet(pq_path, index=False)

        test_logger = logging.getLogger("test_graceful")
        try:
            df        = pd.read_parquet(pq_path)
            validator = DataQualityValidator(baseline_df=None)
            result    = validator.validate(df)
            if not result["is_valid"]:
                for issue in result["issues"]:
                    test_logger.warning(
                        f"[{issue['severity'].upper()}] "
                        f"{issue['type']}: {issue['description']}"
                    )
        except Exception as exc:
            pytest.fail(f"check_and_log equivalent raised: {exc}")

    def test_missing_column_does_not_crash(self, clean_df):
        """If trip_count column is absent, checks skip gracefully."""
        no_tc = clean_df.drop(columns=["trip_count"])
        try:
            result = DataQualityValidator().validate(no_tc)
            assert "is_valid" in result
        except Exception as exc:
            pytest.fail(f"Validator crashed on missing trip_count: {exc}")
