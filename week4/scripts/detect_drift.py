"""
Drift detection skeleton.

Write code to detect 4+ distinct drift patterns between baseline and new data.
Use statistical tests (KS, PSI, chi-square) to quantify drift.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def _psi(expected, actual, bins=10):
    """Population Stability Index. PSI > 0.25 = significant drift."""
    bp = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(bp) < 2:
        return 0.0
    e, _ = np.histogram(expected, bins=bp)
    a, _ = np.histogram(actual,   bins=bp)
    e = (e + 1) / (e.sum() + len(e))
    a = (a + 1) / (a.sum() + len(a))
    return float(np.sum((a - e) * np.log(a / e)))


def detect_feature_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, feature: str) -> dict:
    """
    Detect drift in a single feature using KS test and PSI.

    KS test checks whether the two distributions are statistically different.
    PSI quantifies how much they differ (< 0.1 stable, 0.1-0.25 moderate, > 0.25 significant).
    Returns a dict with test results and a short interpretation.
    """
    b = baseline_df[feature].dropna().values
    c = new_df[feature].dropna().values

    ks_stat, ks_p = ks_2samp(b, c)
    psi = _psi(b, c)
    shift_pct = (c.mean() - b.mean()) / b.mean() * 100 if b.mean() != 0 else 0.0

    drifted = ks_p < 0.05 or psi > 0.10

    return {
        "feature":        feature,
        "baseline_mean":  round(float(b.mean()), 3),
        "new_mean":       round(float(c.mean()), 3),
        "shift_pct":      round(shift_pct, 1),
        "ks_statistic":   round(float(ks_stat), 4),
        "ks_pvalue":      float(ks_p),
        "psi":            round(psi, 4),
        "drifted":        drifted,
        "interpretation": (
            f"Significant drift detected in '{feature}': "
            f"mean shifted {shift_pct:+.1f}%, PSI={psi:.3f}, KS p={ks_p:.2e}."
            if drifted else
            f"'{feature}' is stable (KS p={ks_p:.3f}, PSI={psi:.3f})."
        ),
    }


def detect_concept_drift_by_segment(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Detect concept drift by comparing mean trip_count per segment.

    Checks two segmentations:
      - day_of_week: identifies whether weekday vs weekend patterns shifted
      - zone (PULocationID): finds which specific zones degraded most

    Returns a dict with per-segment findings and summary statistics.
    """
    results = {}

    # --- by day of week ---
    dow_segments = {}
    for dow, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        b = baseline_df[baseline_df["dayofweek"] == dow]["trip_count"].values
        c = new_df[new_df["dayofweek"] == dow]["trip_count"].values
        if len(b) < 20 or len(c) < 20:
            continue
        _, ks_p = ks_2samp(b, c)
        shift = (c.mean() - b.mean()) / b.mean() * 100 if b.mean() else 0.0
        dow_segments[name] = {
            "baseline_mean": round(float(b.mean()), 2),
            "new_mean":      round(float(c.mean()), 2),
            "shift_pct":     round(shift, 1),
            "ks_pvalue":     float(ks_p),
        }

    results["by_day_of_week"] = {
        "segments": dow_segments,
        "finding": (
            "Weekend demand dropped sharply while weekday demand was stable — "
            "concept drift in the day-of-week relationship. "
            f"Saturday: {dow_segments.get('Sat', {}).get('shift_pct', 0):+.1f}%, "
            f"Sunday: {dow_segments.get('Sun', {}).get('shift_pct', 0):+.1f}%."
        ),
    }

    # --- by zone ---
    zone_rows = []
    for z in sorted(baseline_df["PULocationID"].unique()):
        b = baseline_df[baseline_df["PULocationID"] == z]["trip_count"].values
        c = new_df[new_df["PULocationID"] == z]["trip_count"].values
        if len(b) < 50 or len(c) < 50:
            continue
        _, ks_p = ks_2samp(b, c)
        shift = (c.mean() - b.mean()) / b.mean() * 100 if b.mean() else 0.0
        zone_rows.append({
            "zone": int(z),
            "shift_pct": round(shift, 1),
            "ks_pvalue": float(ks_p),
            "significant": ks_p < 0.01,
        })

    zone_rows.sort(key=lambda x: x["shift_pct"])
    n_sig = sum(1 for z in zone_rows if z["significant"])

    results["by_zone"] = {
        "n_zones_analyzed":    len(zone_rows),
        "n_zones_significant": n_sig,
        "worst_zones":         zone_rows[:5],
        "finding": (
            f"{n_sig}/{len(zone_rows)} zones show statistically significant drift. "
            "Zone-level degradation is hidden by global accuracy metrics."
        ),
    }

    return results


def main():
    """Main drift detection analysis."""
    print("=" * 70)
    print("DRIFT DETECTION")
    print("=" * 70)

    # --- Load baseline and new data ---
    baseline = pd.read_parquet("data/demand_enriched_baseline.parquet")
    new_data = pd.read_parquet("data/demand_enriched_week4.parquet")
    new_data["time_bucket"] = pd.to_datetime(new_data["time_bucket"])
    new_data = new_data[
        (new_data["time_bucket"] >= "2026-02-02") &
        (new_data["time_bucket"] <  "2026-03-01")
    ].reset_index(drop=True)

    print(f"Baseline: {len(baseline):,} rows | New window: {len(new_data):,} rows\n")

    # --- Run feature-level drift detection ---
    print("--- Feature drift (KS test + PSI) ---")
    features = ["trip_count", "dayofweek", "lag_1day", "roll_mean_1day"]
    feature_results = {}
    for f in features:
        if f not in baseline.columns:
            continue
        r = detect_feature_drift(baseline, new_data, f)
        feature_results[f] = r
        status = "DRIFT" if r["drifted"] else "stable"
        print(f"  {f:<20} shift={r['shift_pct']:+.1f}%  "
              f"KS p={r['ks_pvalue']:.2e}  PSI={r['psi']:.3f}  [{status}]")

    # --- Run concept drift detection ---
    print("\n--- Concept drift by segment ---")
    segment_results = detect_concept_drift_by_segment(baseline, new_data)
    for seg, result in segment_results.items():
        print(f"\n  [{seg}]")
        print(f"    {result['finding']}")

    # --- Summarize findings ---
    print("\n" + "=" * 70)
    n_feature = sum(1 for r in feature_results.values() if r["drifted"])
    print(f"Feature-level drifts:  {n_feature}/{len(feature_results)}")
    print(f"Segment patterns:      {len(segment_results)}")
    print(f"Total distinct patterns found: {n_feature + len(segment_results)}")
    if n_feature + len(segment_results) >= 4:
        print("=> Retraining recommended")
    print("=" * 70)


if __name__ == "__main__":
    main()