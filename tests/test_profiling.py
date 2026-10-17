"""Unit tests for the profiling and quality scoring engine."""

from pathlib import Path
import pandas as pd

from modules.ingestion import load_file
from modules.profiling import infer_column_semantic_type, profile_data

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_semantic_type_inference_primitives() -> None:
    """Verify inference across isolated synthetic columns."""
    # Boolean
    s_bool = pd.Series(["yes", "no", "yes", "no", "yes"])
    sem_type, _ = infer_column_semantic_type(s_bool)
    assert sem_type == "boolean"

    # Numeric currency
    s_curr = pd.Series(["₹1,200.50", "₹3,400.00", "$500", "2,100"])
    sem_type, details = infer_column_semantic_type(s_curr)
    assert sem_type == "numeric"
    assert details["is_string_encoded_number"] is True

    # Mixed dates
    s_dates = pd.Series(["2023-01-15", "15/02/2023", "Mar 10, 2023", "2023-04-01"])
    sem_type, details = infer_column_semantic_type(s_dates)
    assert sem_type == "datetime"
    assert details["is_string_encoded_date"] is True

    # Constant
    s_const = pd.Series(["FIXED_CODE"] * 20)
    sem_type, _ = infer_column_semantic_type(s_const)
    assert sem_type == "constant"

    # ID-like
    s_id = pd.Series([f"ORD-{i:05d}" for i in range(100)], name="order_id")
    sem_type, _ = infer_column_semantic_type(s_id)
    assert sem_type == "id_like"


def test_profile_clean_sales() -> None:
    """Verify profiling on clean_sales.csv exhibits pristine quality."""
    clean_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    df, _ = load_file(clean_path)
    profile = profile_data(df)

    assert profile["total_rows"] == 1000
    assert profile["total_columns"] == 14
    assert profile["duplicate_rows"] == 0
    assert profile["total_missing_cells"] == 0

    cols = profile["columns"]
    assert cols["order_id"]["semantic_type"] == "id_like"
    assert cols["order_date"]["semantic_type"] == "datetime"
    assert cols["revenue"]["semantic_type"] == "numeric"
    assert cols["region"]["semantic_type"] == "categorical"

    # Clean data score should be 100
    score_info = profile["quality_score"]
    assert score_info["score"] >= 98.0
    assert score_info["grade"] == "A"


def test_profile_messy_sales_anomalies_and_score() -> None:
    """Verify profiling detects deliberate anomalies and scores messy data lower."""
    clean_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"

    df_clean, _ = load_file(clean_path)
    df_messy, _ = load_file(messy_path)

    clean_profile = profile_data(df_clean)
    messy_profile = profile_data(df_messy)

    # Acceptance criterion: messy_sales shows a clearly lower quality score than clean_sales
    clean_score = clean_profile["quality_score"]["score"]
    messy_score = messy_profile["quality_score"]["score"]
    assert clean_score >= 95.0, f"Clean score ({clean_score}) should be >= 95.0"
    assert messy_score < 80.0, f"Messy score ({messy_score}) should be < 80.0"
    assert (
        messy_score < clean_score - 15.0
    ), f"Messy score ({messy_score}) should be substantially lower than clean score ({clean_score})"

    # Column level defect assertions
    m_cols = messy_profile["columns"]
    assert m_cols["store_code"]["semantic_type"] == "constant"
    assert m_cols["cancellation_notes"]["semantic_type"] == "constant"
    assert m_cols["revenue"]["details"]["is_string_encoded_number"] is True
    assert m_cols["order_date"]["details"]["is_string_encoded_date"] is True
    assert m_cols["age"]["details"]["has_negative_values"] is True


def test_original_dataframe_immutability() -> None:
    """Verify profile_data never modifies the original DataFrame."""
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_orig, _ = load_file(messy_path)
    df_copy = df_orig.copy(deep=True)

    _ = profile_data(df_orig)
    pd.testing.assert_frame_equal(df_orig, df_copy)
