"""Unit and acceptance tests for the Phase 2 Data Cleaning Engine."""

from pathlib import Path
import numpy as np
import pandas as pd

from modules.cleaning import (
    clean_currency_and_numeric_strings,
    clean_data,
    clean_datetime_columns,
    correct_negative_anomalies,
    deduplicate_data,
    handle_future_dates,
    handle_outliers,
    handle_structural_defects,
    impute_missing_values,
    standardize_text,
)
from modules.ingestion import load_file
from modules.profiling import profile_data

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_clean_data_immutability() -> None:
    """Verify clean_data guarantees deep-copy immutability of original DataFrame."""
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_orig, _ = load_file(messy_path)
    df_copy = df_orig.copy(deep=True)

    cleaned_df, _ = clean_data(df_orig)

    # Original must remain untouched
    pd.testing.assert_frame_equal(df_orig, df_copy)
    assert cleaned_df is not df_orig


def test_text_standardization_region_and_gender_variants() -> None:
    """Acceptance test: text standardization merges region variants to 4 and gender to 2.

    Specifically verifies that region variants (east/East/EAST/spaces) merge down
    to exactly 4 regions and gender variants (M/Male/male/F/Female/female) merge
    down to exactly 2 values.
    """
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)

    # Run clean_data
    cleaned_df, logs = clean_data(df_messy)

    # 1. Exact count assertion for region
    unique_regions = set(cleaned_df["region"].dropna().unique())
    assert (
        cleaned_df["region"].nunique() == 4
    ), f"Expected exactly 4 regions, got {cleaned_df['region'].nunique()}: {unique_regions}"
    assert unique_regions == {
        "East",
        "North",
        "South",
        "West",
    }, f"Unexpected regions: {unique_regions}"

    # 2. Exact count assertion for gender
    unique_genders = set(cleaned_df["gender"].dropna().unique())
    assert (
        cleaned_df["gender"].nunique() == 2
    ), f"Expected exactly 2 genders, got {cleaned_df['gender'].nunique()}: {unique_genders}"
    assert unique_genders == {"Female", "Male"}, f"Unexpected genders: {unique_genders}"

    # 3. Verify audit log captures text standardization
    text_logs = [log for log in logs if log["step"] == "standardize_text"]
    assert len(text_logs) >= 2
    log_cols = [log["column"] for log in text_logs]
    assert "region" in log_cols
    assert "gender" in log_cols


def test_standardize_text_isolated() -> None:
    """Verify standardize_text handles whitespace, title casing, and gender mapping."""
    df_sample = pd.DataFrame(
        {
            "region": [" north", "NORTH", "North ", "south", "East", "west"],
            "gender": ["m", "MALE", "Male", "f", "Female", "FEMALE"],
        }
    )
    cleaned_df, logs = standardize_text(df_sample)
    assert set(cleaned_df["region"].unique()) == {"North", "South", "East", "West"}
    assert set(cleaned_df["gender"].unique()) == {"Male", "Female"}
    assert len(logs) == 2


def test_deduplication_step() -> None:
    """Verify duplicate row detection, count, and removal."""
    df_sample = pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 3, 3],
            "val": ["a", "b", "b", "c", "c", "c"],
        }
    )
    cleaned_df, log = deduplicate_data(df_sample)
    assert len(cleaned_df) == 3
    assert log["details"]["rows_removed"] == 3
    assert log["step"] == "deduplication"


def test_structural_defects_dropped() -> None:
    """Verify 100% all-null column is dropped and constant column is flagged or dropped."""
    df_sample = pd.DataFrame(
        {
            "valid": [1, 2, 3],
            "all_null": [np.nan, np.nan, np.nan],
            "constant": ["FIXED", "FIXED", "FIXED"],
        }
    )
    cleaned_df, logs = handle_structural_defects(df_sample, drop_all_null=True, drop_constant=True)
    assert "all_null" not in cleaned_df.columns
    assert "constant" not in cleaned_df.columns
    assert "valid" in cleaned_df.columns
    assert len(logs) == 2


def test_clean_currency_strings() -> None:
    """Verify currency symbols, spaces, and commas are parsed to float64."""
    df_sample = pd.DataFrame(
        {
            "revenue": ["₹1,200.50", "$3,400.00", "5,000", "(250.00)", np.nan],
        }
    )
    cleaned_df, logs = clean_currency_and_numeric_strings(df_sample, columns=["revenue"])
    assert pd.api.types.is_float_dtype(cleaned_df["revenue"])
    assert cleaned_df["revenue"].iloc[0] == 1200.50
    assert cleaned_df["revenue"].iloc[1] == 3400.00
    assert cleaned_df["revenue"].iloc[2] == 5000.00
    assert cleaned_df["revenue"].iloc[3] == -250.00
    assert pd.isna(cleaned_df["revenue"].iloc[4])
    assert len(logs) == 1


def test_clean_datetime_columns() -> None:
    """Verify mixed date strings are normalized into datetime64."""
    df_sample = pd.DataFrame(
        {
            "order_date": ["2023-01-15", "15/02/2023", "Mar 10, 2023", "04-20-2023"],
        }
    )
    cleaned_df, logs = clean_datetime_columns(df_sample, columns=["order_date"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["order_date"])
    assert len(logs) == 1


def test_correct_negative_anomalies() -> None:
    """Verify negative values in positive domains (e.g. age) are set to missing."""
    df_sample = pd.DataFrame(
        {
            "age": [25, -18, 30, -42],
            "quantity": [2, -5, 10, 1],
        }
    )
    cleaned_df, logs = correct_negative_anomalies(df_sample)
    assert cleaned_df["age"].isna().sum() == 2
    assert cleaned_df["quantity"].isna().sum() == 1
    assert (cleaned_df["age"].dropna() > 0).all()
    assert (cleaned_df["quantity"].dropna() > 0).all()
    assert len(logs) == 2
    assert logs[0]["step"] == "correct_negatives"
    assert logs[0]["method"] == "set_to_missing"
    assert logs[0]["action"] == "Set 2 negative entries in 'age' to missing"
    assert logs[0]["rows_affected"] == 2
    assert logs[0]["details"]["negatives_flagged"] == 2


def test_handle_outliers_clipping() -> None:
    """Verify IQR boundary clipping treats extreme outlier leverage."""
    df_sample = pd.DataFrame(
        {
            "revenue": [100, 105, 110, 95, 102, 98, 104, 101, 103, 100000],
        }
    )
    cleaned_df, logs = handle_outliers(df_sample, columns=["revenue"], method="iqr", action="clip")
    assert cleaned_df["revenue"].max() < 100000
    assert len(logs) == 1
    assert logs[0]["step"] == "handle_outliers"


def test_small_integer_quantity_validated_not_iqr_capped() -> None:
    """Verify small integer columns like quantity are validated >= 1, not IQR capped."""
    df_sample = pd.DataFrame(
        {
            "quantity": [1, 2, 1, 3, 2, 1, 10, 10, 0, -2],
        }
    )
    cleaned_df, logs = handle_outliers(df_sample, columns=["quantity"])
    assert len(logs) == 1
    assert logs[0]["method"] == "domain_validation"
    # Values of 10 must NOT be capped
    assert (cleaned_df["quantity"] == 10).sum() == 2
    # Invalid values (0 and -2) must be flagged and set to NaN
    assert logs[0]["rows_affected"] == 2
    assert cleaned_df["quantity"].isna().sum() == 2


def test_non_negative_iqr_lower_bounds_clamped_to_zero() -> None:
    """Verify IQR lower bounds for revenue and price do not go below 0, while profit may."""
    df_sample = pd.DataFrame(
        {
            "revenue": [50.0, 55.0, 60.0, 52.0, 58.0, 54.0, 56.0, 53.0, 55.0, 50000.0],
            "profit": [10.0, -10.0, 5.0, -5.0, 8.0, -8.0, 6.0, -6.0, 7.0, 20000.0],
        }
    )
    _, logs = handle_outliers(df_sample, columns=["revenue", "profit"], method="iqr")
    rev_log = next(entry for entry in logs if entry["column"] == "revenue")
    profit_log = next(entry for entry in logs if entry["column"] == "profit")

    assert rev_log["details"]["lower_limit"] >= 0.0, "Revenue lower bound must be >= 0"
    assert profit_log["details"]["lower_limit"] < 0.0, "Profit lower bound may remain negative"


def test_impute_missing_values() -> None:
    """Verify missing value imputation for both numeric and categorical columns."""
    df_sample = pd.DataFrame(
        {
            "num": [10.0, 20.0, np.nan, 30.0],
            "cat": ["A", "B", "A", np.nan],
        }
    )
    cleaned_df, logs = impute_missing_values(
        df_sample, numeric_strategy="median", categorical_strategy="mode"
    )
    assert cleaned_df["num"].isna().sum() == 0
    assert cleaned_df["num"].iloc[2] == 20.0  # median
    assert cleaned_df["cat"].isna().sum() == 0
    assert cleaned_df["cat"].iloc[3] == "A"  # mode
    assert len(logs) == 2


def test_clean_data_end_to_end_quality_score_improvement() -> None:
    """Verify end-to-end cleaning transforms messy data into Grade A quality."""
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)

    before_profile = profile_data(df_messy)
    before_score = before_profile["quality_score"]["score"]
    assert before_score < 80.0

    cleaned_df, logs = clean_data(df_messy)
    after_profile = profile_data(cleaned_df)
    after_score = after_profile["quality_score"]["score"]

    # Verify score improves to Grade A (>= 90)
    assert after_score >= 90.0, f"Expected cleaned score >= 90.0, got {after_score}"
    assert after_profile["quality_score"]["grade"] == "A"
    assert after_profile["duplicate_rows"] == 0
    # Flagged future dates default to NaT, while all missing tabular cells are imputed
    assert after_profile["total_missing_cells"] == 35
    assert len(logs) > 0


def test_future_dates_capping_and_nullify() -> None:
    """Verify future dates beyond max_date default to NaT and can be capped via config."""
    df_sample = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2028-12-31"]),
        }
    )
    max_d = pd.Timestamp("2024-01-01")

    # 1. Default action: nullify (flag and set to NaT)
    cleaned_df_null, logs_null = handle_future_dates(
        df_sample, columns=["order_date"], max_date=max_d
    )
    assert pd.isna(cleaned_df_null["order_date"].iloc[2])
    assert len(logs_null) == 1
    assert logs_null[0]["rows_affected"] == 1
    assert logs_null[0]["method"] == "flag_and_nullify"
    assert logs_null[0]["details"]["future_dates_flagged"] == 1
    assert logs_null[0]["details"]["dates_set_to_missing"] == 1
    assert "missing" in logs_null[0]["action"]

    # 2. Config override: cap at maximum valid date
    cleaned_df_cap, logs_cap = handle_future_dates(
        df_sample, columns=["order_date"], max_date=max_d, action="cap"
    )
    assert cleaned_df_cap["order_date"].max() <= max_d
    assert len(logs_cap) == 1
    assert logs_cap[0]["rows_affected"] == 1
    assert logs_cap[0]["method"] == "flag_and_cap"
    assert logs_cap[0]["details"]["future_dates_flagged"] == 1


def test_negative_age_fixed_before_median_imputation() -> None:
    """Verify negative ages are set to missing so median is computed on legitimate positive ages."""
    # Sample has valid ages 30, 40, one negative age -10, and one NaN
    # Without fixing negatives to NaN, median of (30, 40, -10) is 30.0 (distorted)
    # When -10 is set to NaN, valid positive ages are (30, 40), median is 35.0
    # Both index 2 (formerly -10) and index 3 (formerly NaN) get imputed with 35.0
    df_sample = pd.DataFrame(
        {
            "age": [30.0, 40.0, -10.0, np.nan],
        }
    )
    cleaned_df, _ = clean_data(
        df_sample,
        config={
            "deduplicate": False,
            "drop_all_null": False,
            "clean_currency": False,
            "clean_datetime": False,
            "standardize_text": False,
            "handle_outliers": False,
            "correct_negatives": True,
            "impute_missing": True,
            "numeric_impute_strategy": "median",
        },
    )
    assert cleaned_df["age"].iloc[2] == 35.0
    assert cleaned_df["age"].iloc[3] == 35.0
    assert (cleaned_df["age"] > 0).all()


def test_audit_log_fields_reconciliation() -> None:
    """Verify all audit log entries have standard keys: step, column, method, rows_affected."""
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    _, logs = clean_data(df_messy)

    required_keys = {"timestamp", "step", "column", "method", "rows_affected", "action", "details"}
    for log in logs:
        assert required_keys.issubset(
            set(log.keys())
        ), f"Log entry missing required keys: {log}"
        assert isinstance(log["rows_affected"], int)


def test_order_date_diversity_and_multi_year_span() -> None:
    """Acceptance test: cleaned order_date does not collapse to a single date.

    Asserts:
    1. Cleaned order_date has more than 500 unique dates on messy_sales.csv.
    2. Min and max order_date span at least 2 full years (>= 730 days).
    3. First 10 visible rows exhibit varied dates rather than a constant fallback.
    """
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    cleaned_df, logs = clean_data(df_messy)

    assert "order_date" in cleaned_df.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["order_date"])

    # 1. More than 500 unique dates
    unique_dates = cleaned_df["order_date"].nunique()
    assert (
        unique_dates > 500
    ), f"Expected > 500 unique dates, but found {unique_dates}"

    # 2. Min and max span at least 2 years (730 days)
    min_date = cleaned_df["order_date"].min()
    max_date = cleaned_df["order_date"].max()
    span_days = (max_date - min_date).days
    assert (
        span_days >= 730
    ), (
        f"Expected date span >= 730 days (2 years), but got {span_days} days "
        f"({min_date} to {max_date})"
    )

    # 3. First 10 rows must have multiple distinct dates
    first_10_unique = cleaned_df["order_date"].head(10).nunique()
    assert (
        first_10_unique > 1
    ), f"First 10 rows collapsed to single date: {cleaned_df['order_date'].head(10).tolist()}"


def test_outlier_capping_under_five_percent_and_product_price_preservation() -> None:
    """Acceptance test: Outlier treatment does not squash legitimate catalog prices.

    Asserts:
    1. Capped values are under 5% of total rows for ANY numeric column.
    2. Distinct products (e.g. Laptop Pro and Smartphone 5G) preserve their authentic prices.
    3. The audit log explicitly records rows affected and boundary limits.
    """
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    cleaned_df, logs = clean_data(df_messy)

    total_rows = len(cleaned_df)
    outlier_logs = [log for log in logs if log["step"] == "handle_outliers"]

    # 1. Capped values are under 5% of any column
    for o_log in outlier_logs:
        affected = o_log["rows_affected"]
        col = o_log["column"]
        pct = (affected / total_rows) * 100
        assert (
            affected < 0.05 * total_rows
        ), f"Column '{col}' had {affected} rows capped ({pct:.2f}%), exceeding 5% threshold"

    # 2. Distinct products keep their authentic prices
    if "product" in cleaned_df.columns and "unit_price" in cleaned_df.columns:
        laptop_prices = cleaned_df[cleaned_df["product"] == "Laptop Pro"]["unit_price"]
        phone_prices = cleaned_df[cleaned_df["product"] == "Smartphone 5G"]["unit_price"]

        assert len(laptop_prices) > 0 and len(phone_prices) > 0
        assert laptop_prices.mean() > 50000.0, "Laptop Pro price incorrectly squashed"
        assert (
            28000.0 < phone_prices.mean() < 36000.0
        ), f"Smartphone 5G price outside expected range: {phone_prices.mean()}"
        assert abs(laptop_prices.mean() - phone_prices.mean()) > 10000.0, (
            "Prices collapsed to identical ceiling"
        )

    # 3. Log shows capped counts, percent of rows, and limits
    for o_log in outlier_logs:
        details = o_log["details"]
        assert "lower_limit" in details or "lower_bound" in details
        assert "upper_limit" in details or "upper_bound" in details
        assert o_log["rows_affected"] >= 0
        if details.get("method") != "domain_validation":
            assert "percent_capped" in details
            assert "flagged_exceeds_5pct" in details
            assert not details["flagged_exceeds_5pct"]
            assert "%" in o_log["action"]


def test_rows_affected_matches_actual_changed_cells_deduplication() -> None:
    """Verify log 'rows_affected' strictly equals actual removed rows for deduplication."""
    # 1. Synthetic case: 3 duplicate rows out of 7
    df_sample = pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 3, 3, 4],
            "val": ["a", "b", "b", "c", "c", "c", "d"],
        }
    )
    cleaned_df, log = deduplicate_data(df_sample)
    actual_removed = len(df_sample) - len(cleaned_df)
    assert actual_removed == 3
    assert log["rows_affected"] == actual_removed
    assert log["details"]["rows_removed"] == actual_removed

    # 2. Benchmark dataset case: exactly 140 duplicate rows
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    cleaned_messy, dup_log = deduplicate_data(df_messy)
    actual_messy_removed = len(df_messy) - len(cleaned_messy)
    assert actual_messy_removed == 140
    assert dup_log["rows_affected"] == actual_messy_removed
    assert dup_log["details"]["rows_removed"] == actual_messy_removed


def test_rows_affected_matches_actual_changed_cells_text_standardization() -> None:
    """Verify log 'rows_affected' strictly equals actual changed cells for text standardization."""
    # 1. Synthetic case: 3 out of 6 regions and 2 out of 6 genders need standardization
    df_sample = pd.DataFrame(
        {
            "region": ["East", "east ", "NORTH", "North", "South", "south"],
            "gender": ["Male", "male", "Female", "FEMALE", "Male", "Female"],
        }
    )
    cleaned_df, logs = standardize_text(df_sample)

    for log in logs:
        col = log["column"]
        orig_s = df_sample[col].astype(str)
        cleaned_s = cleaned_df[col].astype(str)
        actual_changed = int((orig_s != cleaned_s).sum())
        assert log["rows_affected"] == actual_changed
        assert log["details"]["cells_changed"] == actual_changed

    region_log = next(log for log in logs if log["column"] == "region")
    assert region_log["rows_affected"] == 3

    gender_log = next(log for log in logs if log["column"] == "gender")
    assert gender_log["rows_affected"] == 2

    # 2. Benchmark messy_sales.csv case: verify exact changed cell count per column
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    df_dedup, _ = deduplicate_data(df_messy)
    cleaned_messy, m_logs = standardize_text(df_dedup)

    for m_log in m_logs:
        col = m_log["column"]
        orig_s = df_dedup[col].dropna().astype(str)
        clean_s = cleaned_messy[col].dropna().astype(str)
        actual_changed = int((orig_s != clean_s).sum())
        assert m_log["rows_affected"] == actual_changed
        assert m_log["details"]["cells_changed"] == actual_changed


def test_rows_affected_matches_actual_changed_cells_outlier_capping() -> None:
    """Verify log 'rows_affected' strictly equals actual clipped cells for outlier capping."""
    # 1. Synthetic case: exactly 2 outliers above IQR fence (1000 and 2000)
    df_sample = pd.DataFrame(
        {
            "sales": [10.0, 11.0, 10.5, 9.8, 10.2, 10.1, 10.4, 10.3, 1000.0, 2000.0],
        }
    )
    cleaned_df, logs = handle_outliers(
        df_sample, columns=["sales"], method="iqr", factor=1.5, action="clip"
    )
    actual_clipped = int((df_sample["sales"] != cleaned_df["sales"]).sum())
    assert actual_clipped == 2
    assert logs[0]["rows_affected"] == actual_clipped
    assert logs[0]["details"]["outliers_treated"] == actual_clipped

    # 2. Benchmark dataset case: verify every outlier log entry matches actual cell diffs
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    df_stage = df_messy.copy(deep=True)
    df_stage, _ = deduplicate_data(df_stage)
    df_stage, _ = clean_currency_and_numeric_strings(df_stage)
    df_stage, _ = correct_negative_anomalies(df_stage)

    df_cleaned_outliers, outlier_logs = handle_outliers(
        df_stage, method="iqr", factor=3.0, action="clip"
    )
    for o_log in outlier_logs:
        col = o_log["column"]
        actual_diff = int((df_stage[col] != df_cleaned_outliers[col]).sum())
        assert o_log["rows_affected"] == actual_diff
        assert o_log["details"]["outliers_treated"] == actual_diff


def test_standardize_constant_column_reports_zero_changed_cells() -> None:
    """Verify standardizing a constant column reports 0 changed cells, not row count."""
    # 1. Synthetic constant column with identical values
    df_const = pd.DataFrame(
        {
            "store_code": ["RETAIL-CORP-HQ"] * 50,
        }
    )
    cleaned_df, logs = standardize_text(df_const)
    assert len(logs) == 1
    assert logs[0]["rows_affected"] == 0
    assert logs[0]["details"]["cells_changed"] == 0
    assert (cleaned_df["store_code"] == "RETAIL-CORP-HQ").all()

    # 2. Synthetic constant column with whitespace on a subset of rows
    df_ws = pd.DataFrame(
        {
            "store_code": ["STORE-1", " STORE-1 ", "STORE-1", "STORE-1 "],
        }
    )
    cleaned_ws, logs_ws = standardize_text(df_ws)
    assert len(logs_ws) == 1
    # Only 2 rows actually differed due to whitespace
    assert logs_ws[0]["rows_affected"] == 2
    assert logs_ws[0]["details"]["cells_changed"] == 2
    assert (cleaned_ws["store_code"] == "STORE-1").all()

    # 3. Benchmark messy_sales.csv store_code: 4,860 rows after dedup must report 0 changed
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    cleaned_messy, full_logs = clean_data(df_messy)
    store_code_std_log = next(
        log
        for log in full_logs
        if log["step"] == "standardize_text" and log["column"] == "store_code"
    )
    assert store_code_std_log["rows_affected"] == 0
    assert store_code_std_log["details"]["cells_changed"] == 0
    assert (cleaned_messy["store_code"] == "RETAIL-CORP-HQ").all()


def test_outliers_capping_percent_and_flag_exceeding_threshold() -> None:
    """Verify log reports capping as percent of rows and flags columns exceeding 5%."""
    # 1. Column with capping <= 5% (2 outliers out of 100 rows = 2.0%)
    normal_vals = [float(x) for x in range(10, 108)] + [5000.0, 6000.0]
    df_low = pd.DataFrame({"metric": normal_vals})
    _, logs_low = handle_outliers(
        df_low, columns=["metric"], method="iqr", factor=1.5, action="clip"
    )
    assert len(logs_low) == 1
    low_log = logs_low[0]
    assert low_log["details"]["percent_capped"] == 2.0
    assert low_log["details"]["flagged_exceeds_5pct"] is False
    assert "2.0%" in low_log["action"]
    assert "[FLAGGED" not in low_log["action"]

    # 2. Column with capping > 5% (10 outliers out of 100 rows = 10.0%)
    high_outlier_vals = [float(x) for x in range(10, 100)] + [5000.0] * 10
    df_high = pd.DataFrame({"metric": high_outlier_vals})
    _, logs_high = handle_outliers(
        df_high, columns=["metric"], method="iqr", factor=1.5, action="clip"
    )
    assert len(logs_high) == 1
    high_log = logs_high[0]
    assert high_log["details"]["percent_capped"] == 10.0
    assert high_log["details"]["flagged_exceeds_5pct"] is True
    assert "10.0%" in high_log["action"]
    assert "[FLAGGED: >5% capped]" in high_log["action"]


def test_cleaned_order_date_never_contains_future_dates_and_no_fabricated_defaults() -> None:
    """Verify cleaned order_date never contains future dates and sets them to NaT."""
    # 1. Benchmark messy_sales.csv case: verify all 35 future dates are set to NaT, not fabricated
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    cleaned_df, logs = clean_data(df_messy)

    now = pd.Timestamp.now()
    valid_dates = cleaned_df["order_date"].dropna()

    # Never contains any date after today
    assert (valid_dates <= now).all(), "Cleaned order_date contains dates occurring after today"

    # Future dates are set to NaT (missing) rather than capped to an invented date.
    # With nullification, all 35 future dates become NaT, and legitimate 2024-12-31 orders
    # remain untouched at exactly 5 (rather than swelling to 40 via fabricated capping).
    assert cleaned_df["order_date"].isna().sum() == 35, "Expected 35 nullified future date entries"
    assert int((cleaned_df["order_date"] == pd.Timestamp("2024-12-31")).sum()) == 5

    future_logs = [log for log in logs if log["step"] == "handle_future_dates"]
    assert len(future_logs) == 1
    f_log = future_logs[0]
    assert f_log["method"] == "flag_and_nullify"
    assert f_log["rows_affected"] == 35
    assert f_log["details"]["dates_set_to_missing"] == 35
    assert f_log["details"]["action"] == "nullify"
    assert "missing" in f_log["action"]

    # 2. Synthetic dataset: verify no fabricated capping date is introduced by default
    df_syn = pd.DataFrame(
        {
            "order_date": ["2023-01-15", "2023-06-20", "2099-01-01", "2099-12-31"],
        }
    )
    cleaned_syn, syn_logs = clean_data(df_syn)
    assert (cleaned_syn["order_date"].dropna() <= now).all()
    assert cleaned_syn["order_date"].isna().sum() == 2
    # The historical max "2023-06-20" was NOT fabricated onto future orders
    assert int((cleaned_syn["order_date"] == pd.Timestamp("2023-06-20")).sum()) == 1
    assert pd.isna(cleaned_syn["order_date"].iloc[2])
    assert pd.isna(cleaned_syn["order_date"].iloc[3])
