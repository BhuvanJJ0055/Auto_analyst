"""Sample Data Generator for AutoAnalyst.

Generates realistic benchmark datasets across clean, messy, timeseries,
and multi-format variations (CSV, Excel, JSON) to test the analytical pipeline.
"""

from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Any, Dict
import numpy as np
import pandas as pd

# Set deterministic seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

REGIONS_STATES = {
    "North": ["Delhi", "Uttar Pradesh", "Punjab", "Haryana"],
    "South": ["Karnataka", "Tamil Nadu", "Telangana", "Kerala"],
    "West": ["Maharashtra", "Gujarat", "Rajasthan", "Goa"],
    "East": ["West Bengal", "Bihar", "Odisha", "Assam"],
}

CATEGORIES_PRODUCTS = {
    "Technology": [
        ("Laptop Pro", 65000),
        ("Smartphone 5G", 32000),
        ("Wireless Earbuds", 4500),
        ("Ultra-wide Monitor", 28000),
        ("Smart Fitness Watch", 6000),
    ],
    "Furniture": [
        ("Ergonomic Mesh Chair", 12500),
        ("Motorized Standing Desk", 24000),
        ("Modular Bookshelf", 8500),
        ("Executive Leather Sofa", 35000),
        ("LED Desk Lamp", 2200),
    ],
    "Office Supplies": [
        ("Recycled Paper Ream", 450),
        ("Executive Rollerball Pen", 850),
        ("Heavy-Duty Binder", 350),
        ("Steel Stapler Set", 600),
        ("Magnetic Whiteboard", 3200),
    ],
    "Clothing": [
        ("Premium Cotton T-Shirt", 1200),
        ("Slim-fit Denim Jeans", 2800),
        ("Wrinkle-Free Formal Shirt", 2200),
        ("Fleece Hooded Sweatshirt", 3100),
        ("Breathable Running Shoes", 4500),
    ],
}


def _generate_clean_record(order_num: int, order_date: datetime) -> Dict[str, Any]:
    """Generate a single pristine sales record."""
    region = random.choice(list(REGIONS_STATES.keys()))
    state = random.choice(REGIONS_STATES[region])
    category = random.choice(list(CATEGORIES_PRODUCTS.keys()))
    product, base_price = random.choice(CATEGORIES_PRODUCTS[category])

    gender = random.choice(["Female", "Male", "Non-Binary"])
    age = int(np.random.randint(18, 68))
    quantity = int(
        np.random.choice(
            [1, 2, 3, 4, 5, 6, 8, 10], p=[0.40, 0.25, 0.15, 0.08, 0.05, 0.04, 0.02, 0.01]
        )
    )

    price_jitter = float(np.random.uniform(0.92, 1.08))
    unit_price = round(base_price * price_jitter, 2)

    discount_rate = float(
        np.random.choice(
            [0.0, 0.05, 0.10, 0.15, 0.20, 0.25], p=[0.35, 0.25, 0.20, 0.10, 0.07, 0.03]
        )
    )
    discount_amount = round(quantity * unit_price * discount_rate, 2)
    revenue = round((quantity * unit_price) - discount_amount, 2)

    margin_rate = float(np.random.uniform(0.12, 0.38))
    profit = round(revenue * margin_rate, 2)

    rating = float(np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], p=[0.04, 0.06, 0.18, 0.42, 0.30]))

    return {
        "order_id": f"ORD-{order_num:06d}",
        "order_date": order_date.strftime("%Y-%m-%d"),
        "region": region,
        "state": state,
        "category": category,
        "product": product,
        "gender": gender,
        "age": age,
        "quantity": quantity,
        "unit_price": unit_price,
        "discount": round(discount_rate * 100, 1),
        "revenue": revenue,
        "profit": profit,
        "customer_rating": rating,
    }


def make_clean_sales(num_rows: int = 1000) -> pd.DataFrame:
    """Generate 1,000 clean sales transactions."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    days_range = (end_date - start_date).days

    records = []
    for i in range(1, num_rows + 1):
        order_date = start_date + timedelta(days=random.randint(0, days_range))
        records.append(_generate_clean_record(i, order_date))

    df = pd.DataFrame(records)
    df.sort_values("order_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def make_messy_sales(num_rows: int = 5000) -> pd.DataFrame:
    """Generate 5,000 sales transactions with deliberate data anomalies."""
    num_duplicates = 140
    base_rows = num_rows - num_duplicates
    df = make_clean_sales(num_rows=base_rows)

    # 1. Add structural defect columns:
    # Constant column
    df["store_code"] = "RETAIL-CORP-HQ"
    # All-null column
    df["cancellation_notes"] = np.nan
    # ID-like column already present: 'order_id'

    # 2. Missing values in 3 distinct columns (5-15%)
    n = len(df)
    missing_rating_idx = np.random.choice(df.index, size=int(0.12 * n), replace=False)
    df.loc[missing_rating_idx, "customer_rating"] = np.nan

    missing_age_idx = np.random.choice(df.index, size=int(0.08 * n), replace=False)
    df.loc[missing_age_idx, "age"] = np.nan

    missing_discount_idx = np.random.choice(df.index, size=int(0.10 * n), replace=False)
    df.loc[missing_discount_idx, "discount"] = np.nan

    # 3. Dates stored as text in mixed formats
    date_objs = pd.to_datetime(df["order_date"])
    mixed_dates = []
    for i, d in enumerate(date_objs):
        fmt_choice = i % 5
        if fmt_choice == 0:
            mixed_dates.append(d.strftime("%Y-%m-%d"))  # ISO
        elif fmt_choice == 1:
            mixed_dates.append(d.strftime("%d/%m/%Y"))  # European
        elif fmt_choice == 2:
            mixed_dates.append(d.strftime("%m-%d-%Y"))  # US
        elif fmt_choice == 3:
            mixed_dates.append(d.strftime("%b %d, %Y"))  # Textual month
        else:
            mixed_dates.append(d.strftime("%d-%b-%Y"))  # DD-Mon-YYYY
    df["order_date"] = mixed_dates

    # 4. Future dates injected into ~35 rows
    future_indices = np.random.choice(df.index, size=35, replace=False)
    for idx in future_indices:
        future_year = random.choice([2028, 2029, 2031])
        df.loc[idx, "order_date"] = f"{future_year}-06-15"

    # 5. Currency stored as strings with symbols and separators
    currency_rev = []
    for rev in df["revenue"]:
        if pd.isna(rev):
            currency_rev.append(rev)
            continue
        style = random.choice([0, 1, 2, 3])
        if style == 0:
            currency_rev.append(f"₹{rev:,.2f}")
        elif style == 1:
            currency_rev.append(f"₹ {rev:,.2f}")
        elif style == 2:
            currency_rev.append(f"{rev:,.2f}")
        else:
            currency_rev.append(f"${rev:,.2f}")
    df["revenue"] = currency_rev

    # 6. Inconsistent text categories
    gender_variants = ["Male", "male", "M", "MALE", "Female", "female", "F", "FEMALE", "Non-Binary"]
    region_variants = {
        "North": ["North", "north ", "NORTH", " north"],
        "South": ["South", "south", "SOUTH ", " south"],
        "East": ["East", "east", "EAST ", " east"],
        "West": ["West", "west", "WEST ", " west"],
    }
    df["gender"] = [random.choice(gender_variants) for _ in range(len(df))]
    df["region"] = [random.choice(region_variants.get(r, [r])) for r in df["region"]]

    # 7. Outliers in Revenue (injected into numeric base or formatted values)
    outlier_indices = np.random.choice(df.index, size=25, replace=False)
    for idx in outlier_indices:
        crazy_value = float(random.choice([1500000, 2450000, 8900000]))
        df.loc[idx, "revenue"] = f"₹{crazy_value:,.2f}"

    # 8. Negative ages
    neg_age_indices = np.random.choice(df.index, size=30, replace=False)
    for idx in neg_age_indices:
        df.loc[idx, "age"] = -int(random.choice([4, 8, 12, 19]))

    # 9. Now sample exactly ~140 duplicate rows from the anomalous dataset
    dup_indices = np.random.choice(df.index, size=num_duplicates, replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def make_timeseries_sales() -> pd.DataFrame:
    """Generate 3 continuous years of daily transaction volume with trend and seasonality."""
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(1096)]  # 3 full years: 2022, 2023, 2024

    records = []
    order_num = 1
    for day_idx, d in enumerate(dates):
        # Base daily order volume: baseline trend + day-of-week seasonality + annual seasonality
        day_of_week = d.weekday()
        weekend_boost = 1.35 if day_of_week in [5, 6] else 1.0
        annual_cycle = 1.0 + 0.3 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
        trend = 1.0 + (0.45 * (day_idx / 1096.0))  # 45% growth over 3 years
        q4_holiday_spike = 1.5 if d.month in [10, 11, 12] else 1.0

        daily_volume = max(
            1,
            int(
                np.random.poisson(lam=12 * weekend_boost * annual_cycle * trend * q4_holiday_spike)
            ),
        )

        for _ in range(daily_volume):
            record = _generate_clean_record(order_num, d)
            records.append(record)
            order_num += 1

    df = pd.DataFrame(records)
    return df


def make_tiny_sales(num_rows: int = 50) -> pd.DataFrame:
    """Generate minimal 50-row dataset for quick verification and smoke tests."""
    return make_clean_sales(num_rows=num_rows)


def make_large_sales(num_rows: int = 200000) -> pd.DataFrame:
    """Generate high-volume 200,000-row dataset to benchmark filter performance."""
    chunk_size = 20000
    chunks = []
    start_date = datetime(2023, 1, 1)
    for chunk_id in range(num_rows // chunk_size):
        chunk_records = []
        for i in range(chunk_size):
            order_idx = (chunk_id * chunk_size) + i + 1
            order_date = start_date + timedelta(days=(order_idx % 365))
            chunk_records.append(_generate_clean_record(order_idx, order_date))
        chunks.append(pd.DataFrame(chunk_records))
    return pd.concat(chunks, ignore_index=True)


def generate_all_datasets(generate_large: bool = True) -> None:
    """Orchestrate generation and serialization of all sample datasets."""
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating clean_sales.csv (1,000 rows)...")
    clean_df = make_clean_sales(1000)
    clean_df.to_csv(SAMPLE_DATA_DIR / "clean_sales.csv", index=False)

    print("Generating messy_sales.csv (5,000 rows with deliberate anomalies)...")
    messy_df = make_messy_sales(5000)
    messy_df.to_csv(SAMPLE_DATA_DIR / "messy_sales.csv", index=False)

    print("Generating timeseries_sales.csv (3 years daily trend and seasonality)...")
    ts_df = make_timeseries_sales()
    ts_df.to_csv(SAMPLE_DATA_DIR / "timeseries_sales.csv", index=False)

    print("Generating tiny.csv (50 rows)...")
    tiny_df = make_tiny_sales(50)
    tiny_df.to_csv(SAMPLE_DATA_DIR / "tiny.csv", index=False)

    print("Generating sales.xlsx, sales.json, and sales.parquet (100 rows sample)...")
    sample_format_df = clean_df.head(100)
    with pd.ExcelWriter(SAMPLE_DATA_DIR / "sales.xlsx", engine="openpyxl") as writer:
        sample_format_df.to_excel(writer, sheet_name="Transactions", index=False)
        sample_format_df.head(30).to_excel(writer, sheet_name="Regional_Summary", index=False)
    sample_format_df.to_json(SAMPLE_DATA_DIR / "sales.json", orient="records", indent=2)
    sample_format_df.to_parquet(SAMPLE_DATA_DIR / "sales.parquet", index=False, engine="pyarrow")

    if generate_large:
        print("Generating large.csv (200,000 rows, git-ignored)...")
        large_df = make_large_sales(200000)
        large_df.to_csv(SAMPLE_DATA_DIR / "large.csv", index=False)

    print("All sample datasets successfully generated in:", SAMPLE_DATA_DIR)


if __name__ == "__main__":
    generate_all_datasets(generate_large=True)
