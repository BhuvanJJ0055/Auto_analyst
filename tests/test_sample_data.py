"""Tests for benchmark sample datasets."""

from pathlib import Path
import json
import pandas as pd

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_clean_sales_properties() -> None:
    """Verify clean_sales.csv has expected shape and zero data defects."""
    file_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    assert file_path.exists(), "clean_sales.csv must exist"

    df = pd.read_csv(file_path)
    assert len(df) == 1000, "clean_sales must have exactly 1,000 rows"
    assert df.duplicated().sum() == 0, "clean_sales should have 0 duplicate rows"
    assert df.isna().sum().sum() == 0, "clean_sales should have 0 missing values"

    expected_cols = [
        "order_id",
        "order_date",
        "region",
        "state",
        "category",
        "product",
        "gender",
        "age",
        "quantity",
        "unit_price",
        "discount",
        "revenue",
        "profit",
        "customer_rating",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} missing from clean_sales"

    assert (df["age"] >= 18).all() and (
        df["age"] <= 70
    ).all(), "Ages in clean_sales should be within [18, 70]"
    assert (df["discount"] >= 0).all() and (
        df["discount"] <= 100
    ).all(), "Discounts must be valid percentages"


def test_messy_sales_defects() -> None:
    """Verify messy_sales.csv contains all specified deliberate defects."""
    file_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    assert file_path.exists(), "messy_sales.csv must exist"

    df = pd.read_csv(file_path)
    assert len(df) == 5000, "messy_sales must have 5,000 rows"

    # 1. Duplicate rows
    dup_count = df.duplicated().sum()
    assert 130 <= dup_count <= 150, f"Expected ~140 duplicate rows, found {dup_count}"

    # 2. Structural columns
    assert "store_code" in df.columns, "store_code constant column should be present"
    assert df["store_code"].nunique() == 1, "store_code should be constant"
    assert (
        "cancellation_notes" in df.columns
    ), "cancellation_notes all-null column should be present"
    assert df["cancellation_notes"].isna().all(), "cancellation_notes should be all null"

    # 3. Missing values in 3 distinct columns
    missing_cols = [c for c in ["customer_rating", "age", "discount"] if df[c].isna().sum() > 0]
    assert len(missing_cols) == 3, "Expected 3 columns with missing values"

    # 4. Currency strings in revenue
    assert (
        pd.api.types.is_string_dtype(df["revenue"]) or df["revenue"].dtype == object
    ), "Revenue in messy_sales should be stored as string"
    has_symbols = df["revenue"].astype(str).str.contains(r"[₹$,]", regex=True).any()
    assert has_symbols, "Revenue should contain currency symbols or comma separators"

    # 5. Negative ages
    neg_ages = (df["age"] < 0).sum()
    assert neg_ages > 0, "messy_sales should contain negative age values"


def test_timeseries_sales() -> None:
    """Verify timeseries_sales.csv covers 3 years of daily daily volume."""
    file_path = SAMPLE_DATA_DIR / "timeseries_sales.csv"
    assert file_path.exists(), "timeseries_sales.csv must exist"

    df = pd.read_csv(file_path)
    assert len(df) > 1000, "Timeseries data should have substantial transaction volume"
    dates = pd.to_datetime(df["order_date"])
    assert dates.min().year == 2022
    assert dates.max().year == 2024


def test_tiny_and_multiformat_sales() -> None:
    """Verify tiny.csv, sales.xlsx, and sales.json load properly."""
    tiny_path = SAMPLE_DATA_DIR / "tiny.csv"
    assert tiny_path.exists()
    tiny_df = pd.read_csv(tiny_path)
    assert len(tiny_df) == 50

    excel_path = SAMPLE_DATA_DIR / "sales.xlsx"
    assert excel_path.exists()
    excel_df = pd.read_excel(excel_path, engine="openpyxl")
    assert len(excel_df) == 100

    json_path = SAMPLE_DATA_DIR / "sales.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 100

    parquet_path = SAMPLE_DATA_DIR / "sales.parquet"
    assert parquet_path.exists()
    parquet_df = pd.read_parquet(parquet_path, engine="pyarrow")
    assert len(parquet_df) == 100
