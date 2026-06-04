"""
tests for metric computation and drift detection.

Run:
    python3 -m unittest scripts/test_monitoring.py -v
    pytest scripts/test_monitoring.py -v        # also works
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_template import MetricComputer, THRESHOLDS
from compute_metrics import classify, evaluate_alerts


def make_clean_df(n_days: int = 7, n_zones: int = 5, seed: int = 0,
                  start: str = "2026-01-01") -> pd.DataFrame:
    """Generate a baseline-like dataframe with realistic hourly demand."""
    rng = np.random.default_rng(seed)
    rows = []
    start_ts = pd.Timestamp(start)
    for d in range(n_days):
        for slot in range(96):
            ts = start_ts + pd.Timedelta(days=d, minutes=15 * slot)
            for z in range(n_zones):
                hour = ts.hour
                rate = 5 + 15 * np.sin(np.pi * hour / 24) ** 2
                rows.append({
                    "PULocationID": z,
                    "time_bucket":  ts,
                    "trip_count":   int(rng.poisson(rate)),
                    "hour":         hour,
                    "minute":       ts.minute,
                    "dayofweek":    ts.dayofweek,
                    "is_weekend":   int(ts.dayofweek >= 5),
                    "zone_slot_baseline": float(rate),
                    "lag_15min":    float(rng.poisson(rate)),
                    "lag_1h":       float(rng.poisson(rate)),
                    "lag_2h":       float(rng.poisson(rate)),
                    "lag_1day":     float(rng.poisson(rate)),
                    "lag_1week":    float(rng.poisson(rate)),
                    "roll_mean_1h":   float(rate),
                    "roll_mean_2h":   float(rate),
                    "roll_mean_1day": float(rate),
                })
    return pd.DataFrame(rows)


class TestMetric1Accuracy(unittest.TestCase):
    def setUp(self):
        self.bl  = make_clean_df(n_days=7, seed=42)
        self.cur = make_clean_df(n_days=3, seed=43, start="2026-01-08")

    def test_clean_data_high_accuracy(self):
        acc = MetricComputer(self.bl).metric_1_accuracy(self.cur)
        self.assertGreater(acc, THRESHOLDS["metric_1_accuracy"]["warning"])

    def test_drifted_data_lower_accuracy(self):
        drifted = self.cur.copy()
        drifted["trip_count"] = drifted["trip_count"] + 30
        mc = MetricComputer(self.bl)
        self.assertLess(mc.metric_1_accuracy(drifted),
                        mc.metric_1_accuracy(self.cur))


class TestMetric2ZoneAccuracy(unittest.TestCase):
    def setUp(self):
        self.bl = make_clean_df(seed=42)
        self.cur = make_clean_df(seed=43)

    def test_returns_dict_per_zone(self):
        d = MetricComputer(self.bl).metric_2_accuracy_by_zone(self.cur)
        self.assertEqual(set(d.keys()), set(self.cur["PULocationID"].unique()))
        self.assertTrue(all(0 <= v <= 1 for v in d.values()))

    def test_detects_zone_specific_drift(self):
        drifted = self.cur.copy()
        mask = drifted["PULocationID"] == 0
        drifted.loc[mask, "trip_count"] = drifted.loc[mask, "trip_count"] + 50
        mc = MetricComputer(self.bl)
        clean_d = mc.metric_2_accuracy_by_zone(self.cur)
        drift_d = mc.metric_2_accuracy_by_zone(drifted)
        self.assertLess(drift_d[0], clean_d[0] - 0.3)
        for z in [1, 2, 3, 4]:
            self.assertLess(abs(drift_d[z] - clean_d[z]), 0.05)


class TestMetric3NullRates(unittest.TestCase):
    def setUp(self):
        self.bl = make_clean_df(seed=42)
        self.cur = make_clean_df(seed=43)

    def test_clean_data_zero_nulls(self):
        nulls = MetricComputer(self.bl).metric_3_null_rates(self.cur)
        self.assertTrue(all(v == 0.0 for v in nulls.values()))

    def test_detects_injected_nulls(self):
        broken = self.cur.copy()
        broken.loc[broken.sample(frac=0.05, random_state=0).index, "trip_count"] = np.nan
        nulls = MetricComputer(self.bl).metric_3_null_rates(broken)
        self.assertGreater(nulls["trip_count"], 0.03)
        self.assertLess(nulls["trip_count"], 0.08)


class TestMetric4KSTest(unittest.TestCase):
    def setUp(self):
        self.bl = make_clean_df(seed=42)
        self.cur = make_clean_df(seed=43)

    def test_identical_distribution_high_pvalue(self):
        result = MetricComputer(self.bl).metric_4_ks_test(self.bl)
        self.assertGreater(result["trip_count"]["pvalue"], 0.5)

    def test_shifted_distribution_low_pvalue(self):
        shifted = self.cur.copy()
        shifted["trip_count"] = shifted["trip_count"] + 20
        result = MetricComputer(self.bl).metric_4_ks_test(shifted)
        self.assertLess(result["trip_count"]["pvalue"], 0.01)


class TestMetric5PSI(unittest.TestCase):
    def setUp(self):
        self.bl = make_clean_df(seed=42)
        self.cur = make_clean_df(seed=43)

    def test_psi_identical_near_zero(self):
        psi = MetricComputer(self.bl)._psi(
            self.bl["trip_count"].values, self.bl["trip_count"].values)
        self.assertLess(psi, 0.01)

    def test_psi_shifted_critical(self):
        shifted = self.cur.copy()
        shifted["trip_count"] = shifted["trip_count"] * 3 + 10
        result = MetricComputer(self.bl).metric_5_psi(shifted)
        self.assertGreater(result["trip_count"], 0.25)


class TestMetric6PredictionDistribution(unittest.TestCase):
    def setUp(self):
        self.bl = make_clean_df(seed=42)
        self.cur = make_clean_df(seed=43)

    def test_normal_distribution_not_collapsed(self):
        r = MetricComputer(self.bl).metric_6_prediction_distribution(self.cur)
        self.assertFalse(r["collapsed"])

    def test_constant_predictions_detect_collapse(self):
        broken = self.cur.copy()
        broken["zone_slot_baseline"] = 5.0
        r = MetricComputer(self.bl).metric_6_prediction_distribution(broken)
        self.assertTrue(r["collapsed"])


class TestMetric7DataFreshness(unittest.TestCase):
    def setUp(self):
        self.bl = make_clean_df(seed=42)
        self.cur = make_clean_df(seed=43)

    def test_fresh_data_not_stale(self):
        latest = self.cur["time_bucket"].max()
        now = (latest + pd.Timedelta(minutes=15)).to_pydatetime()
        r = MetricComputer(self.bl).metric_7_data_freshness(self.cur, now=now)
        self.assertFalse(r["stale"])

    def test_old_data_stale(self):
        latest = self.cur["time_bucket"].max()
        now = (latest + pd.Timedelta(hours=10)).to_pydatetime()
        r = MetricComputer(self.bl).metric_7_data_freshness(self.cur, now=now)
        self.assertTrue(r["stale"])


class TestMetric8Duplicates(unittest.TestCase):
    def setUp(self):
        self.bl = make_clean_df(seed=42)
        self.cur = make_clean_df(seed=43)

    def test_clean_no_duplicates(self):
        r = MetricComputer(self.bl).metric_8_duplicate_rate(self.cur)
        self.assertEqual(r["duplicate_count"], 0)

    def test_detects_duplicates(self):
        dups = self.cur.head(20)
        broken = pd.concat([self.cur, dups], ignore_index=True)
        r = MetricComputer(self.bl).metric_8_duplicate_rate(broken)
        self.assertGreaterEqual(r["duplicate_count"], 20)


class TestClassify(unittest.TestCase):
    LOWER_T  = {"info": 0.9, "warning": 0.8, "critical": 0.7}
    HIGHER_T = {"info": 0.1, "warning": 0.25, "critical": 0.5}

    def test_lower_is_worse_healthy(self):
        self.assertEqual(classify(0.95, self.LOWER_T, lower_is_worse=True), "healthy")
    def test_lower_is_worse_info(self):
        self.assertEqual(classify(0.85, self.LOWER_T, lower_is_worse=True), "info")
    def test_lower_is_worse_warning(self):
        self.assertEqual(classify(0.75, self.LOWER_T, lower_is_worse=True), "warning")
    def test_lower_is_worse_critical(self):
        self.assertEqual(classify(0.65, self.LOWER_T, lower_is_worse=True), "critical")
    def test_higher_is_worse_healthy(self):
        self.assertEqual(classify(0.05, self.HIGHER_T, lower_is_worse=False), "healthy")
    def test_higher_is_worse_critical(self):
        self.assertEqual(classify(0.60, self.HIGHER_T, lower_is_worse=False), "critical")


class TestEvaluateAlerts(unittest.TestCase):
    def setUp(self):
        # Both windows must cover full weeks; otherwise dayofweek KS
        # flags a window artifact rather than real drift.
        self.bl  = make_clean_df(n_days=14, seed=42, start="2026-01-01")
        self.cur = make_clean_df(n_days=7,  seed=43, start="2026-01-15")

    def test_clean_data_yields_healthy_or_info(self):
        mc = MetricComputer(self.bl)
        now = (self.cur["time_bucket"].max() + pd.Timedelta(minutes=15)).to_pydatetime()
        results = mc.compute_all_metrics(self.cur, now=now)
        alerts = evaluate_alerts(results)
        self.assertIn(alerts["_overall"], {"healthy", "info"})

    def test_drifted_data_critical_overall(self):
        drifted = self.cur.copy()
        drifted["trip_count"] = drifted["trip_count"] * 3 + 20
        mc = MetricComputer(self.bl)
        now = (drifted["time_bucket"].max() + pd.Timedelta(minutes=15)).to_pydatetime()
        results = mc.compute_all_metrics(drifted, now=now)
        alerts = evaluate_alerts(results)
        self.assertIn(alerts["_overall"], {"warning", "critical"})


class TestRealDataIntegration(unittest.TestCase):
    """Verify real Feb data flags critical (skipped if files absent)."""

    @classmethod
    def setUpClass(cls):
        bl_path = None
        for p in ["/mnt/user-data/uploads/baseline.csv",
                  "data/baseline.csv",
                  "data/demand_enriched_baseline.parquet"]:
            if Path(p).exists():
                bl_path = p; break
        cur_path = None
        for p in ["/mnt/user-data/uploads/week4_feb.csv",
                  "data/week4_feb.csv",
                  "data/demand_enriched_week4.parquet"]:
            if Path(p).exists():
                cur_path = p; break
        if not bl_path or not cur_path:
            raise unittest.SkipTest("Real data not mounted")
        load = (lambda p: pd.read_parquet(p)) if str(bl_path).endswith(".parquet") \
               else (lambda p: pd.read_csv(p))
        cls.bl  = load(bl_path)
        cls.cur = load(cur_path)
        if "time_bucket" in cls.cur.columns:
            cls.cur["time_bucket"] = pd.to_datetime(cls.cur["time_bucket"])

    def test_real_feb_data_flags_critical(self):
        mc = MetricComputer(self.bl)
        now = self.cur["time_bucket"].max().to_pydatetime()
        results = mc.compute_all_metrics(self.cur, now=now)
        alerts = evaluate_alerts(results)
        self.assertEqual(alerts["_overall"], "critical",
                         f"expected critical, got {alerts['_overall']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
