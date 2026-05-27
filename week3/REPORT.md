# Week 3 Report

## Part 1: Issues Found

The dataset splits into a clean baseline with 6,079,392 rows before Jan 16 2026. And a new corrupted window with 250,853 rows from Jan 16 onward. I found four issues by comparing the two windows.

**Issue 1: Duplicate rows**

The corrupted window has 20,170 rows that share the same PULocationID and time_bucket key. The baseline has zero duplicates. This matters because the model's lag and rolling-mean features are computed from trip counts per slot, so duplicated rows directly inflate those features for the affected zones. The likely cause is a double-write in the upstream ETL job that reprocessed the same time window twice.

**Issue 2: Negative trip counts**

353 rows in the corrupted window have trip_count below zero, with a minimum of -5. The baseline minimum is 0. Negative demand is impossible and would corrupt any lag feature derived from that slot. So my best guess for the cause is a delta-encoding bug where a correction value was applied on top of the raw count without a floor of zero.

**Issue 3: Extreme outliers in trip_count**

The corrupted window has a maximum trip_count of 99,999. The baseline maximum is 310. So I flagged any value above 2x the baseline max (620) as an outlier, which caught 311 rows. Values this large would dominate model loss during retraining and propagate bad lag values forward in time for those zones. Possible causes include a unit conversion error or a test value that leaked into production.

**Issue 4: cbd_pricing_active stuck at 100%**

In the baseline there is about 33.8% of rows have cbd_pricing_active = 1, which makes sense since congestion pricing only applies during peak hours in certain zones. In the corrupted window every single row has the flag set to 1. A constant feature carries no information for the model. Any business logic that branches on this flag like surge pricing and CBD dispatch would always take the same path regardless of actual conditions. This looks like a stale cache or a hardcoded override in the upstream data pipeline.

## Part 2: Validation and Graceful Degradation

I implemented a DataQualityValidator class in validation/check_data_quality.py with one check per issue. The checks are:

- check_duplicates: flags if duplicate rate on PULocationID and time_bucket exceeds 1%
- check_negative_trip_count: flags any row where trip_count less than 0
- check_trip_count_outliers: flags values above 2x the baseline maximum
- check_categorical_rates: flags cbd_pricing_active if its mean falls outside 0-60%

Each check returns a structured dict with type, severity, description, and count. The validate() method runs all four and returns is_valid plus the full issues list.

For graceful degradation I added check_and_log_data_quality() to data.py, called at API startup. It loads the corrupted parquet, splits the baseline and new window, runs the validator, and logs every issue as a WARNING.

The key design decision here was to wrap the entire function in a try/except so the API starts regardless of what happens. If the file is missing or the import fails, it logs an error and moves on. The API continues serving the clean Week 2 data while operators can see the warnings in logs.

The principle here is that degradation should always be visible. The API returning wrong answers silently is worse than the API logging a warning and serving slightly stale data.

## Part 3: Validation Schedule and Trade-offs

I chose hourly validation. The main alternatives were every 15 minutes, daily, and startup-only. For this taxi demand forecasting API, a full day of corrupted predictions is too long to wait. At the same time, the upstream data is a batch parquet file that gets updated a few times per day at most, so checking every 15 minutes would burn CI minutes without catching anything new.

Hourly means that if bad data lands at 9am, the on-call team knows by 10am and can intervene within the same shift. That feels like the right tradeoff for this use case.

The workflow also triggers on any push to main that touches the data or validation directories, so code changes are checked immediately. I have two validation layers. GitHub Actions catches issues before deployment, and the startup check in data.py catches anything that slips through or appears after deployment. They check the same things but serve different purposes.

One limitation worth noting is that the parquet file is not committed to the repo, so the CI job will exit early if the file is not present. For a real production setup the data would be pulled from a storage bucket at the start of the workflow.
