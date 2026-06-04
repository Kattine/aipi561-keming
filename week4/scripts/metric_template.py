"""
Monitoring metrics skeleton.

This file defines 8 metric stubs for monitoring data and model health.
Implement at least 5 of the 8 metrics based on your monitoring framework design.
Each metric should compute a specific health signal about your data/model,
and return a dict (or float) that can be checked against your alert thresholds.

8 metrics covering 4 categories:
  - Model performance: metric_1 (overall acc), metric_2 (per-zone acc)
  - Data quality:      metric_3 (nulls),       metric_8 (duplicates)
  - Distribution drift: metric_4 (KS test),    metric_5 (PSI)
  - Model / infra:     metric_6 (pred dist),   metric_7 (freshness)

"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime, timezone


class MetricComputer:
    """Compute monitoring metrics for drift detection."""

    # Critical columns expected in every batch
    CRITICAL_COLS = ["trip_count", "PULocationID", "time_bucket"]
    LAG_COLS = ["lag_15min", "lag_1h", "lag_2h", "lag_1day", "lag_1week"]

    # Accuracy tolerance: prediction is "correct" if within ±5 trips
    # OR within 50% relative error. Calibrated so baseline ~86% accurate.
    ABS_TOL = 5
    REL_TOL = 0.50

    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline_df = baseline_df
        self.baseline_trip_count = baseline_df["trip_count"].values

    # ---- internal helpers ----
    def _accuracy_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Return boolean array: prediction within tolerance."""
        err = (df["trip_count"] - df["zone_slot_baseline"]).abs()
        rel = err / df["zone_slot_baseline"].clip(lower=1)
        return ((err <= self.ABS_TOL) | (rel <= self.REL_TOL)).values

    # Metric 1 — Overall Accuracy (concept drift)
    def metric_1_accuracy(self, new_df: pd.DataFrame) -> float:
        """Fraction of records where naive prediction is within tolerance.

        Baseline ~86%. Alert thresholds:
          INFO     >= 0.84
          WARNING  0.78 - 0.84
          CRITICAL < 0.7
        """
        return float(self._accuracy_mask(new_df).mean())

    def metric_2_accuracy_by_zone(self, new_df: pd.DataFrame) -> dict:
        """Per-zone accuracy. Returns dict {zone_id: accuracy}.

        Alert if any zone falls below 0.70.
        """
        new_df = new_df.copy()
        new_df["__correct"] = self._accuracy_mask(new_df)
        by_zone = new_df.groupby("PULocationID")["__correct"].mean()
        return {int(z): float(a) for z, a in by_zone.items()}

    def metric_3_null_rates(self, new_df: pd.DataFrame) -> dict:
        """Null rate per critical column.

        Baseline: ~0%. Alert > 1%.
        """
        cols = self.CRITICAL_COLS + self.LAG_COLS
        return {
            c: float(new_df[c].isnull().mean())
            for c in cols
            if c in new_df.columns
        }

    def metric_4_ks_test(self, new_df: pd.DataFrame) -> dict:
        """KS test on key features. Returns {feature: {stat, pvalue}}.

        Alert: pvalue < 0.01 indicates significant distribution shift.
        """
        features = ["trip_count", "hour", "dayofweek", "lag_1day", "roll_mean_1day"]
        out = {}
        for f in features:
            if f not in self.baseline_df.columns or f not in new_df.columns:
                continue
            b = self.baseline_df[f].dropna().values
            c = new_df[f].dropna().values
            if len(b) == 0 or len(c) == 0:
                continue
            stat, pval = ks_2samp(b, c)
            out[f] = {"statistic": float(stat), "pvalue": float(pval)}
        return out

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Compute PSI with quantile binning + Laplace smoothing.

        PSI < 0.10  : stable
        0.10 - 0.25 : moderate shift (monitor)
        > 0.25      : significant shift (act)
        """
        breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
        if len(breakpoints) < 2:
            return 0.0
        e_hist, _ = np.histogram(expected, bins=breakpoints)
        a_hist, _ = np.histogram(actual, bins=breakpoints)
        # Laplace smoothing to avoid log(0)
        e_pct = (e_hist + 1) / (e_hist.sum() + len(e_hist))
        a_pct = (a_hist + 1) / (a_hist.sum() + len(a_hist))
        return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

    def metric_5_psi(self, new_df: pd.DataFrame, bins: int = 10) -> dict:
        """PSI for trip_count and key lag features.

        Returns dict {feature: psi_value}.
        """
        features = ["trip_count", "lag_1day", "lag_1week", "roll_mean_1day"]
        out = {}
        for f in features:
            if f not in self.baseline_df.columns or f not in new_df.columns:
                continue
            b = self.baseline_df[f].dropna().values
            c = new_df[f].dropna().values
            if len(b) == 0 or len(c) == 0:
                continue
            out[f] = self._psi(b, c, bins=bins)
        return out

    def metric_6_prediction_distribution(self, new_df: pd.DataFrame) -> dict:
        """Sanity-check naive predictor distribution.

        Compares prediction (zone_slot_baseline) statistics to baseline.
        Catches: model collapse, constant predictions, prediction shift.
        """
        baseline_pred = self.baseline_df["zone_slot_baseline"].values
        new_pred = new_df["zone_slot_baseline"].values
        b_std = float(baseline_pred.std())
        n_std = float(new_pred.std())

        return {
            "baseline_mean": float(baseline_pred.mean()),
            "new_mean": float(new_pred.mean()),
            "baseline_std": b_std,
            "new_std": n_std,
            "std_ratio": n_std / b_std if b_std > 0 else 0.0,
            "collapsed": n_std < 0.1 * b_std,  # <10% of baseline std = collapsed
        }

    def metric_7_data_freshness(self, new_df: pd.DataFrame,
                                now: datetime | None = None) -> dict:
        """How old is the most recent record?

        Alert: > 6 hours behind = upstream pipeline issue.
        """
        if now is None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
        ts = pd.to_datetime(new_df["time_bucket"])
        latest = ts.max()
        age = now - latest.to_pydatetime()
        age_minutes = age.total_seconds() / 60
        return {
            "latest_record": latest.isoformat(),
            "age_minutes": float(age_minutes),
            "age_hours": float(age_minutes / 60),
            "stale": age_minutes > 360,  # >6h
        }

    def metric_8_duplicate_rate(self, new_df: pd.DataFrame) -> dict:
        """Fraction of rows that are exact duplicates on (zone, time_bucket).

        Baseline: 0. Alert > 0.5%.
        """
        key_cols = ["PULocationID", "time_bucket"]
        dup_count = int(new_df.duplicated(subset=key_cols).sum())
        return {
            "duplicate_count": dup_count,
            "duplicate_rate": float(dup_count / max(1, len(new_df))),
        }

    def compute_all_metrics(self, new_df: pd.DataFrame,
                            now: datetime | None = None) -> dict:
        """Run every metric. Returns nested results dict."""
        return {
            "n_rows": int(len(new_df)),
            "metric_1_accuracy": self.metric_1_accuracy(new_df),
            "metric_2_accuracy_by_zone": self.metric_2_accuracy_by_zone(new_df),
            "metric_3_null_rates": self.metric_3_null_rates(new_df),
            "metric_4_ks_test": self.metric_4_ks_test(new_df),
            "metric_5_psi": self.metric_5_psi(new_df),
            "metric_6_prediction_distribution": self.metric_6_prediction_distribution(new_df),
            "metric_7_data_freshness": self.metric_7_data_freshness(new_df, now=now),
            "metric_8_duplicate_rate": self.metric_8_duplicate_rate(new_df),
        }


# ---- Alert threshold spec (used by compute_metrics.py + detect_drift.py) ----
THRESHOLDS = {
    "metric_1_accuracy":           {"info": 0.84, "warning": 0.78, "critical": 0.70},
    "metric_2_zone_accuracy_min":  {"info": 0.80, "warning": 0.70, "critical": 0.60},
    "metric_3_null_rate":          {"info": 0.005, "warning": 0.01, "critical": 0.02},
    "metric_4_ks_pvalue":          {"info": 0.05, "warning": 0.01, "critical": 0.001},
    "metric_5_psi":                {"info": 0.10, "warning": 0.25, "critical": 0.50},
    "metric_6_std_ratio_min":      {"info": 0.80, "warning": 0.50, "critical": 0.10},
    "metric_7_age_hours":          {"info": 1, "warning": 6, "critical": 24},
    "metric_8_duplicate_rate":     {"info": 0.001, "warning": 0.005, "critical": 0.01},
}
