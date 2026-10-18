"""Data Profiling Module for AutoAnalyst.

Performs dynamic semantic type inference, statistical summaries, missingness
assessment, and transparent 0-100 data quality scoring without hardcoded column names.
"""

import re
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from modules.utils import format_bytes


def _clean_currency_string(val: Any) -> Optional[float]:
    """Attempt converting currency string with symbols and separators to float."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    # Remove currency symbols and thousand commas
    cleaned = re.sub(r"[₹$€£¥,\s]", "", s)
    # Check parenthesized negative e.g. (120)
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def infer_column_semantic_type(series: pd.Series) -> Tuple[str, Dict[str, Any]]:
    """Infer semantic type of a Pandas Series without hardcoded column names.

    Possible types:
        - 'constant': Column has only 1 unique value or is completely null.
        - 'id_like': Primary key or high-cardinality sequential/alphanumeric code.
        - 'boolean': Binary boolean flags or yes/no representations.
        - 'numeric': Quantitative measures (float or int), including parsed currency strings.
        - 'datetime': Chronological timestamps or parsed date strings.
        - 'categorical': Low-cardinality discrete groupings.
        - 'text': High-cardinality unstructured free text.

    Args:
        series: Pandas Series to analyze.

    Returns:
        Tuple of (inferred_semantic_type, details_dict).
    """
    total_len = len(series)
    non_null = series.dropna()
    non_null_len = len(non_null)

    details: Dict[str, Any] = {
        "is_string_encoded_number": False,
        "is_string_encoded_date": False,
        "has_negative_values": False,
        "text_casing_inconsistencies": False,
    }

    if non_null_len == 0 or non_null.nunique() == 1:
        return "constant", details

    # 1. Native or Object Boolean Check
    if pd.api.types.is_bool_dtype(series):
        return "boolean", details

    if non_null.nunique() == 2:
        str_vals = {str(v).strip().lower() for v in non_null.unique()}
        bool_pairs = [
            {"true", "false"},
            {"yes", "no"},
            {"t", "f"},
            {"y", "n"},
            {"1", "0"},
            {"1.0", "0.0"},
        ]
        if any(str_vals.issubset(pair) for pair in bool_pairs):
            return "boolean", details

    # 2. Native Datetime Check
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", details

    # 3. Native Numeric Check
    if pd.api.types.is_numeric_dtype(series):
        details["has_negative_values"] = bool((non_null < 0).any())
        col_name = str(series.name).lower()
        # Check if integer ID
        if pd.api.types.is_integer_dtype(series):
            if non_null.nunique() == non_null_len and (
                any(col_name.endswith(k) for k in ["id", "_id", "id_", "code", "key"])
                or total_len >= 50
            ):
                return "id_like", details
        return "numeric", details

    # 4. Object / String Analysis
    sample_size = min(500, non_null_len)
    sample = (
        non_null.sample(n=sample_size, random_state=42) if non_null_len > sample_size else non_null
    )
    sample_strs = [str(x).strip() for x in sample]

    # 4a. Check string-encoded numbers (currency symbols, commas, scientific notation)
    cleaned_nums = [_clean_currency_string(x) for x in sample_strs]
    valid_nums = [n for n in cleaned_nums if n is not None]
    if len(valid_nums) / len(sample_strs) >= 0.85:
        details["is_string_encoded_number"] = True
        details["has_negative_values"] = any(n < 0 for n in valid_nums)
        return "numeric", details

    DATE_PATTERN = (
        r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|"
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"
    )
    date_pattern_count = sum(
        1
        for x in sample_strs
        if re.search(
            DATE_PATTERN,
            x,
            re.IGNORECASE,
        )
    )
    if date_pattern_count / len(sample_strs) >= 0.60:
        parsed_dates = pd.to_datetime(sample_strs, errors="coerce", format="mixed")
        if parsed_dates.notna().sum() / len(sample_strs) >= 0.70:
            details["is_string_encoded_date"] = True
            iso_count = sum(1 for x in sample_strs if re.match(r"^\d{4}-\d{2}-\d{2}$", x))
            details["has_mixed_date_formats"] = iso_count < len(sample_strs)
            return "datetime", details

    # 4c. Check ID-like strings
    col_name = str(series.name).lower()
    is_id_name = any(col_name.endswith(k) for k in ["id", "_id", "code", "key"]) or col_name in [
        "id",
        "order_id",
        "customer_id",
        "transaction_id",
        "uuid",
    ]
    unique_ratio = non_null.nunique() / non_null_len
    if (unique_ratio >= 0.95 and total_len >= 20) or (is_id_name and unique_ratio >= 0.80):
        return "id_like", details

    # 4d. Check text casing inconsistencies in categories (e.g. Male, male, MALE)
    lowered_uniques = len({s.lower() for s in sample_strs})
    exact_uniques = len(set(sample_strs))
    if exact_uniques > lowered_uniques:
        details["text_casing_inconsistencies"] = True

    # 4e. Categorical vs Text
    nunique_val = non_null.nunique()
    avg_len = sum(len(s) for s in sample_strs) / len(sample_strs)
    if nunique_val <= 60 or (unique_ratio <= 0.10 and avg_len < 40):
        return "categorical", details

    return "text", details


def compute_data_quality_score(
    df: pd.DataFrame,
    column_profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate an objective 0-100 data quality score with transparent breakdown.

    Formulation:
        - Completeness (30%): Evaluates missing cell density and columns with null values.
        - Uniqueness (25%): Evaluates duplicate transaction row presence.
        - Validity (25%): Evaluates type conformance (string numbers, mixed date strings,
          negative values in restricted domains, and casing inconsistencies).
        - Consistency (20%): Evaluates dataset structural health (constant and all-null columns).

    Args:
        df: Input DataFrame.
        column_profiles: Column-level profiling dictionary.

    Returns:
        Dictionary containing overall score, letter grade, and component breakdowns.
    """
    total_cells = int(df.shape[0] * df.shape[1])
    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])

    if total_cells == 0 or total_rows == 0:
        return {
            "score": 0.0,
            "grade": "F",
            "completeness_score": 0.0,
            "uniqueness_score": 0.0,
            "validity_score": 0.0,
            "consistency_score": 0.0,
            "formula_explanation": "Empty dataset",
        }

    # 1. Completeness (Weight: 30%)
    missing_cells = int(df.isna().sum().sum())
    cell_missing_ratio = missing_cells / total_cells
    cols_with_missing = sum(
        1 for col in column_profiles.values() if col.get("missing_count", 0) > 0
    )
    col_missing_ratio = cols_with_missing / max(1, total_cols)

    completeness_factor = max(0.0, 1.0 - (0.5 * cell_missing_ratio + 0.5 * col_missing_ratio))
    completeness_score = round(completeness_factor * 30.0, 1)

    # 2. Uniqueness (Weight: 25%)
    duplicate_rows = int(df.duplicated().sum())
    dup_ratio = duplicate_rows / total_rows
    # Scale duplicate penalty so duplicates are meaningfully flagged
    uniqueness_penalty = min(1.0, dup_ratio * 8.0)
    uniqueness_factor = max(0.0, 1.0 - uniqueness_penalty)
    uniqueness_score = round(uniqueness_factor * 25.0, 1)

    # 3. Validity (Weight: 25%)
    invalid_col_count = 0
    for col_info in column_profiles.values():
        details = col_info.get("details", {})
        has_issue = (
            details.get("is_string_encoded_number")
            or details.get("has_mixed_date_formats")
            or (
                details.get("has_negative_values")
                and col_info.get("name", "").lower() in ["age", "quantity", "price"]
            )
            or details.get("text_casing_inconsistencies")
        )
        if has_issue:
            invalid_col_count += 1

    validity_penalty = min(1.0, (invalid_col_count / max(1, total_cols)) * 1.8)
    validity_factor = max(0.0, 1.0 - validity_penalty)
    validity_score = round(validity_factor * 25.0, 1)

    # 4. Consistency (Weight: 20%)
    constant_or_null_cols = sum(
        1 for col in column_profiles.values() if col.get("semantic_type") == "constant"
    )
    # Deduct 5 points per problematic structural column up to 20
    consistency_penalty = min(20.0, float(constant_or_null_cols * 5.0))
    consistency_score = round(max(0.0, 20.0 - consistency_penalty), 1)

    overall_score = round(
        completeness_score + uniqueness_score + validity_score + consistency_score, 1
    )
    overall_score = min(100.0, max(0.0, overall_score))

    if overall_score >= 90.0:
        grade = "A"
    elif overall_score >= 80.0:
        grade = "B"
    elif overall_score >= 70.0:
        grade = "C"
    elif overall_score >= 60.0:
        grade = "D"
    else:
        grade = "F"

    formula_explanation = (
        "Quality Score = Completeness (30%) + Uniqueness (25%) + "
        "Validity (25%) + Consistency (20%)"
    )

    return {
        "score": overall_score,
        "grade": grade,
        "completeness_score": completeness_score,
        "uniqueness_score": uniqueness_score,
        "validity_score": validity_score,
        "consistency_score": consistency_score,
        "completeness_ratio": round((completeness_score / 30.0) * 100, 1),
        "uniqueness_ratio": round((uniqueness_score / 25.0) * 100, 1),
        "validity_ratio": round((validity_score / 25.0) * 100, 1),
        "consistency_ratio": round((consistency_score / 20.0) * 100, 1),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "invalid_columns": invalid_col_count,
        "constant_or_null_columns": constant_or_null_cols,
        "formula_explanation": formula_explanation,
    }


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive profiling summary of a tabular dataset.

    Args:
        df: Input DataFrame (will not be modified).

    Returns:
        Structured dictionary containing shape, memory usage, per-column profiling,
        and data quality scoring metrics.
    """
    total_rows, total_cols = df.shape
    memory_bytes = int(df.memory_usage(deep=True).sum())
    dup_rows = int(df.duplicated().sum())
    total_missing = int(df.isna().sum().sum())

    columns_summary: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        non_null_s = series.dropna()
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique())
        unique_pct = round((unique_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        sem_type, details = infer_column_semantic_type(series)

        col_dict: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "semantic_type": sem_type,
            "details": details,
            "missing_count": null_count,
            "missing_pct": null_pct,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
            "stats": {},
            "top_values": [],
        }

        # Numeric Stats
        if sem_type == "numeric":
            if pd.api.types.is_numeric_dtype(series):
                num_clean = pd.to_numeric(non_null_s, errors="coerce").dropna()
            else:
                num_clean = non_null_s.apply(_clean_currency_string).dropna()

            if not num_clean.empty:
                col_dict["stats"] = {
                    "min": float(num_clean.min()),
                    "max": float(num_clean.max()),
                    "mean": round(float(num_clean.mean()), 2),
                    "median": round(float(num_clean.median()), 2),
                    "std": round(float(num_clean.std()), 2) if len(num_clean) > 1 else 0.0,
                    "q25": round(float(num_clean.quantile(0.25)), 2),
                    "q75": round(float(num_clean.quantile(0.75)), 2),
                }

        # Categorical / Text Top Values
        if sem_type in ["categorical", "text", "boolean", "constant", "id_like"]:
            val_counts = non_null_s.value_counts().head(5)
            col_dict["top_values"] = [
                {
                    "value": str(val),
                    "count": int(count),
                    "pct": round((count / total_rows) * 100, 1),
                }
                for val, count in val_counts.items()
            ]

        columns_summary[str(col)] = col_dict

    quality_score = compute_data_quality_score(df, columns_summary)

    return {
        "shape": (total_rows, total_cols),
        "total_rows": total_rows,
        "total_columns": total_cols,
        "memory_bytes": memory_bytes,
        "memory_formatted": format_bytes(memory_bytes),
        "duplicate_rows": dup_rows,
        "duplicate_pct": round((dup_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0,
        "total_missing_cells": total_missing,
        "columns": columns_summary,
        "quality_score": quality_score,
    }
