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
    """Verify negative values in positive domains (e.g. age) are inverted."""
    df_sample = pd.DataFrame(
        {
            "age": [25, -18, 30, -42],
            "quantity": [2, -5, 10, 1],
        }
    )
    cleaned_df, logs = correct_negative_anomalies(df_sample)
    assert (cleaned_df["age"] >= 0).all()
    assert (cleaned_df["quantity"] >= 0).all()
    assert cleaned_df["age"].tolist() == [25, 18, 30, 42]
    assert cleaned_df["quantity"].tolist() == [2, 5, 10, 1]
    assert len(logs) == 2


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
    assert after_score >= 95.0, f"Expected cleaned score >= 95.0, got {after_score}"
    assert after_profile["quality_score"]["grade"] == "A"
    assert after_profile["duplicate_rows"] == 0
    assert after_profile["total_missing_cells"] == 0
    assert len(logs) > 0


def test_future_dates_capping() -> None:
    """Verify future dates beyond max_date are capped or nullified."""
    df_sample = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2028-12-31"]),
        }
    )
    max_d = pd.Timestamp("2024-01-01")
    cleaned_df, logs = handle_future_dates(
        df_sample, columns=["order_date"], max_date=max_d, action="cap"
    )
    assert cleaned_df["order_date"].max() <= max_d
    assert len(logs) == 1
    assert logs[0]["rows_affected"] == 1
    assert logs[0]["method"] == "cap_to_max_valid_date"


def test_negative_age_fixed_before_median_imputation() -> None:
    """Verify negative ages are corrected before median is computed so imputation is accurate."""
    # If negatives were not fixed: values [-20, 20, 40, np.nan], median of (-20, 20, 40) is 20
    # After negatives fixed: values [20, 20, 40, np.nan], median of (20, 20, 40) is 20
    # Stronger case: [-50, -40, 10, np.nan]
    # Unfixed median: -40. Fixed median: (10, 40, 50) -> 40.
    df_sample = pd.DataFrame(
        {
            "age": [-50, -40, 10, np.nan],
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
    # Since negatives were fixed to [50, 40, 10], median is 40.0
    imputed_value = cleaned_df["age"].iloc[3]
    assert imputed_value == 40.0, f"Expected median 40.0, got {imputed_value}"
    assert (cleaned_df["age"] >= 0).all()


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

    # 3. Log shows capped counts and limits
    for o_log in outlier_logs:
        details = o_log["details"]
        assert "lower_limit" in details or "lower_bound" in details
        assert "upper_limit" in details or "upper_bound" in details
        assert o_log["rows_affected"] >= 0
