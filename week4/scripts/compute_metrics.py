#!/usr/bin/env python3
"""
compute_metrics.py — CI entry point for monitoring metrics.

Workflow:
  1. Load baseline parquet (Jan 1-15 healthy data) and new data parquet.
  2. Filter new data to the most recent monitoring window (default: last 7 days).
  3. Instantiate MetricComputer with baseline.
  4. Compute all 8 metrics.
  5. Compare each result to the tiered thresholds in THRESHOLDS.
  6. Write metrics-<timestamp>.json with full results + alert status.
  7. Exit non-zero if any CRITICAL alert fires (so CI can create an issue).

Run locally:
    python3 scripts/compute_metrics.py \
        --baseline data/demand_enriched_baseline.parquet \
        --current  data/demand_enriched_week4.parquet \
        --window-days 7
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from metric_template import MetricComputer, THRESHOLDS


# Alert classification helpers
def classify(value: float, thresholds: dict, *, lower_is_worse: bool) -> str:
    """Return one of: 'healthy', 'info', 'warning', 'critical'."""
    if lower_is_worse:
        if value < thresholds["critical"]:
            return "critical"
        if value < thresholds["warning"]:
            return "warning"
        if value < thresholds["info"]:
            return "info"
        return "healthy"
    else:
        if value > thresholds["critical"]:
            return "critical"
        if value > thresholds["warning"]:
            return "warning"
        if value > thresholds["info"]:
            return "info"
        return "healthy"


def evaluate_alerts(results: dict) -> dict:
    """Walk through results and assign alert level per metric."""
    alerts = {}

    # M1: accuracy — lower is worse
    alerts["metric_1_accuracy"] = classify(
        results["metric_1_accuracy"],
        THRESHOLDS["metric_1_accuracy"],
        lower_is_worse=True,
    )

    # M2: zone accuracy — alert on the worst zone
    zone_min = min(results["metric_2_accuracy_by_zone"].values())
    alerts["metric_2_zone_accuracy_min"] = classify(
        zone_min,
        THRESHOLDS["metric_2_zone_accuracy_min"],
        lower_is_worse=True,
    )
    alerts["_zone_accuracy_min_value"] = zone_min

    # M3: nulls — higher is worse; alert on worst column
    null_max = max(results["metric_3_null_rates"].values()) if results["metric_3_null_rates"] else 0
    alerts["metric_3_null_rate"] = classify(
        null_max,
        THRESHOLDS["metric_3_null_rate"],
        lower_is_worse=False,
    )

    # M4: KS p-values — lower is worse; alert on smallest p
    if results["metric_4_ks_test"]:
        ks_min = min(v["pvalue"] for v in results["metric_4_ks_test"].values())
        alerts["metric_4_ks_pvalue"] = classify(
            ks_min,
            THRESHOLDS["metric_4_ks_pvalue"],
            lower_is_worse=True,
        )

    # M5: PSI — higher is worse; alert on max PSI
    if results["metric_5_psi"]:
        psi_max = max(results["metric_5_psi"].values())
        alerts["metric_5_psi"] = classify(
            psi_max,
            THRESHOLDS["metric_5_psi"],
            lower_is_worse=False,
        )

    # M6: prediction collapse — std_ratio lower is worse
    std_ratio = results["metric_6_prediction_distribution"]["std_ratio"]
    alerts["metric_6_std_ratio"] = classify(
        std_ratio,
        THRESHOLDS["metric_6_std_ratio_min"],
        lower_is_worse=True,
    )

    # M7: freshness — higher hours = worse
    age_h = results["metric_7_data_freshness"]["age_hours"]
    alerts["metric_7_age_hours"] = classify(
        age_h,
        THRESHOLDS["metric_7_age_hours"],
        lower_is_worse=False,
    )

    # M8: duplicates — higher is worse
    alerts["metric_8_duplicate_rate"] = classify(
        results["metric_8_duplicate_rate"]["duplicate_rate"],
        THRESHOLDS["metric_8_duplicate_rate"],
        lower_is_worse=False,
    )

    # Overall severity
    severities = [v for k, v in alerts.items() if not k.startswith("_") and isinstance(v, str)]
    if "critical" in severities:
        alerts["_overall"] = "critical"
    elif "warning" in severities:
        alerts["_overall"] = "warning"
    elif "info" in severities:
        alerts["_overall"] = "info"
    else:
        alerts["_overall"] = "healthy"
    return alerts


# Data loading
def load_data(path: str) -> pd.DataFrame:
    """Load parquet or csv based on extension."""
    p = Path(path)
    if p.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix in {".csv", ".tsv"}:
        return pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    raise ValueError(f"Unsupported file type: {p.suffix}")


def filter_window(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Filter to most recent `window_days` of data."""
    if "time_bucket" not in df.columns:
        return df
    df = df.copy()
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    cutoff = df["time_bucket"].max() - pd.Timedelta(days=window_days)
    return df[df["time_bucket"] > cutoff].reset_index(drop=True)



def main() -> int:
    ap = argparse.ArgumentParser(description="Compute monitoring metrics and write JSON.")
    ap.add_argument("--baseline", default="data/demand_enriched_baseline.parquet")
    ap.add_argument("--current",  default="data/demand_enriched_week4.parquet")
    ap.add_argument("--window-days", type=int, default=7,
                    help="Monitor the last N days of current data (default 7).")
    ap.add_argument("--output-dir", default=".",
                    help="Where to write metrics-<ts>.json (default cwd).")
    ap.add_argument("--fail-on", default="critical",
                    choices=["never", "critical", "warning"],
                    help="Exit non-zero if overall severity >= this level.")
    args = ap.parse_args()

    print(f"[compute_metrics] Loading baseline:  {args.baseline}")
    baseline = load_data(args.baseline)
    print(f"[compute_metrics] Loading current:   {args.current}")
    current = load_data(args.current)
    print(f"[compute_metrics] Filtering to last {args.window_days} days...")
    current = filter_window(current, args.window_days)
    print(f"[compute_metrics] Window rows: {len(current):,}")

    print("[compute_metrics] Computing 8 metrics...")
    mc = MetricComputer(baseline)
    # Use max(time_bucket) as 'now' so freshness check is meaningful for
    # historical data. In real CI, omit this to compare against wall clock.
    now = pd.to_datetime(current["time_bucket"]).max().to_pydatetime() \
        if "time_bucket" in current.columns else datetime.now(timezone.utc).replace(tzinfo=None)
    results = mc.compute_all_metrics(current, now=now)

    print("[compute_metrics] Evaluating alerts...")
    alerts = evaluate_alerts(results)

    # Print readable summary 
    print("\n" + "=" * 60)
    print(f"OVERALL SEVERITY: {alerts['_overall'].upper()}")
    print("=" * 60)
    print(f"  M1  overall accuracy  : {results['metric_1_accuracy']:.3f}     [{alerts['metric_1_accuracy']}]")
    print(f"  M2  worst-zone accuracy: {alerts['_zone_accuracy_min_value']:.3f}     [{alerts['metric_2_zone_accuracy_min']}]")
    print(f"  M3  max null rate     : {max(results['metric_3_null_rates'].values()):.4f}    [{alerts['metric_3_null_rate']}]")
    if "metric_4_ks_pvalue" in alerts:
        ks_min = min(v['pvalue'] for v in results['metric_4_ks_test'].values())
        print(f"  M4  min KS p-value    : {ks_min:.2e}  [{alerts['metric_4_ks_pvalue']}]")
    if "metric_5_psi" in alerts:
        psi_max = max(results['metric_5_psi'].values())
        print(f"  M5  max PSI           : {psi_max:.3f}     [{alerts['metric_5_psi']}]")
    print(f"  M6  std ratio         : {results['metric_6_prediction_distribution']['std_ratio']:.3f}     [{alerts['metric_6_std_ratio']}]")
    print(f"  M7  data age (hours)  : {results['metric_7_data_freshness']['age_hours']:.1f}    [{alerts['metric_7_age_hours']}]")
    print(f"  M8  duplicate rate    : {results['metric_8_duplicate_rate']['duplicate_rate']:.4f}    [{alerts['metric_8_duplicate_rate']}]")
    print("=" * 60)

    # Write JSON artifact with full results + alert status
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.output_dir) / f"metrics-{ts}.json"
    payload = {
        "timestamp_utc": ts,
        "window_days": args.window_days,
        "n_rows_evaluated": int(len(current)),
        "results": results,
        "alerts": alerts,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[compute_metrics] Wrote {out_path}")

    # Decide exit code based on overall alert severity and --fail-on threshold
    fail_levels = {
        "never":    set(),
        "critical": {"critical"},
        "warning":  {"warning", "critical"},
    }
    if alerts["_overall"] in fail_levels[args.fail_on]:
        print(f"[compute_metrics] FAILING: overall={alerts['_overall']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
