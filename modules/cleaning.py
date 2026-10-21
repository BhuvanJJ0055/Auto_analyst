"""Data Cleaning Engine Module for AutoAnalyst.

Provides automated, configurable, and non-destructive data cleaning pipelines:
- Deduplication with logged duplicate indices
- Structural defect handling (dropping all-null columns, constant column handling)
- String-encoded currency and numeric normalization
- Mixed datetime parsing and standardization
- Future date anomaly detection and treatment
- Text standardization (whitespace trimming, case normalization, category canonicalization)
- Negative anomaly correction for positive domain variables
- Outlier detection and treatment (IQR clamping, z-score capping)
- Missing value imputation (mean/median for numeric, mode/constant for categorical)
- Comprehensive structured audit log recording all transformations.
"""

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from modules.profiling import infer_column_semantic_type


GENDER_SYNONYMS: Dict[str, str] = {
    "m": "Male",
    "male": "Male",
    "man": "Male",
    "boy": "Male",
    "f": "Female",
    "female": "Female",
    "woman": "Female",
    "girl": "Female",
}


def _clean_currency_value(val: Any) -> Optional[float]:
    """Parse string-encoded numeric or currency value into float."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    cleaned = re.sub(r"[₹$€£¥,\s]", "", s)
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def deduplicate_data(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Identify and remove duplicate rows from DataFrame.

    Args:
        df: Input DataFrame.
        subset: Optional list of columns to consider for identifying duplicates.
        keep: Which duplicate occurrence to keep ('first', 'last', False).

    Returns:
        Tuple of (deduplicated DataFrame copy, audit log entry).
    """
    dup_mask = df.duplicated(subset=subset, keep=keep)
    duplicate_count = int(dup_mask.sum())
    dup_indices = df.index[dup_mask].tolist()

    cleaned_df = df[~dup_mask].copy().reset_index(drop=True)

    log_entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "step": "deduplication",
        "column": ", ".join(subset) if subset else "all_columns",
        "method": "exact_row_match",
        "rows_affected": duplicate_count,
        "action": f"Removed {duplicate_count} duplicate rows",
        "details": {
            "rows_removed": duplicate_count,
            "rows_remaining": len(cleaned_df),
            "duplicate_indices_sample": dup_indices[:10],
        },
    }
    return cleaned_df, log_entry


def handle_structural_defects(
    df: pd.DataFrame,
    drop_all_null: bool = True,
    drop_constant: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Identify and handle all-null and constant columns.

    Args:
        df: Input DataFrame.
        drop_all_null: Whether to drop columns that have 100% missing values.
        drop_constant: Whether to drop columns that have only 1 unique non-null value.

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cleaned_df = df.copy()
    logs: List[Dict[str, Any]] = []

    for col in list(cleaned_df.columns):
        series = cleaned_df[col]
        non_null_count = int(series.dropna().count())
        unique_count = int(series.dropna().nunique())

        if non_null_count == 0:
            if drop_all_null:
                cleaned_df.drop(columns=[col], inplace=True)
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "drop_all_null_column",
                        "column": col,
                        "method": "drop_empty_column",
                        "rows_affected": 1,
                        "action": f"Dropped 1 column '{col}' (100% empty)",
                        "details": {
                            "columns_dropped": 1,
                            "non_null_count": 0,
                            "total_rows": len(df),
                        },
                    }
                )
        elif unique_count == 1:
            if drop_constant:
                cleaned_df.drop(columns=[col], inplace=True)
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "drop_constant_column",
                        "column": col,
                        "method": "drop_single_value_column",
                        "rows_affected": 1,
                        "action": (
                            f"Dropped 1 column '{col}' "
                            f"(constant: {str(series.dropna().iloc[0])[:20]})"
                        ),
                        "details": {
                            "columns_dropped": 1,
                            "constant_value": str(series.dropna().iloc[0]),
                            "unique_count": 1,
                        },
                    }
                )
            else:
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "flag_constant_column",
                        "column": col,
                        "method": "retain_business_constant",
                        "rows_affected": 0,
                        "action": f"Flagged constant column '{col}' (retained, 0 rows changed)",
                        "details": {
                            "columns_dropped": 0,
                            "constant_value": str(series.dropna().iloc[0]),
                            "unique_count": 1,
                        },
                    }
                )

    return cleaned_df, logs


def clean_currency_and_numeric_strings(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Convert string-encoded currency and numbers into native float64.

    Args:
        df: Input DataFrame.
        columns: Optional explicit list of columns to convert. If None,
                 auto-detects columns with string currency patterns.

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cleaned_df = df.copy()
    logs: List[Dict[str, Any]] = []

    target_cols = columns or list(cleaned_df.columns)

    for col in target_cols:
        if col not in cleaned_df.columns:
            continue
        series = cleaned_df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue

        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        sample_vals = non_null.astype(str).head(50)
        has_currency_symbols = sample_vals.str.contains(r"[₹$€£¥]", regex=True).any()
        has_thousand_commas = sample_vals.str.contains(r"\d{1,3},\d{3}", regex=True).any()

        if columns or has_currency_symbols or has_thousand_commas:
            parsed = series.map(_clean_currency_value)
            success_count = int(parsed.dropna().count())
            if success_count > 0 and (
                columns or (success_count / len(non_null) > 0.6)
            ):
                changed_mask = series.notna() & series.apply(lambda v: isinstance(v, str))
                changed_count = int(changed_mask.sum())
                cleaned_df[col] = parsed.astype("float64")
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "clean_currency_strings",
                        "column": col,
                        "method": "regex_currency_parse",
                        "rows_affected": changed_count,
                        "action": (
                            f"Normalized {changed_count} currency strings in '{col}' to float64"
                        ),
                        "details": {
                            "cells_changed": changed_count,
                            "parsed_values": success_count,
                            "original_dtype": str(series.dtype),
                            "new_dtype": "float64",
                        },
                    }
                )

    return cleaned_df, logs


def clean_datetime_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Parse string-encoded or mixed date columns into datetime64[ns].

    Args:
        df: Input DataFrame.
        columns: Optional explicit list of columns. If None, uses semantic inference.

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cleaned_df = df.copy()
    logs: List[Dict[str, Any]] = []

    target_cols = columns or list(cleaned_df.columns)

    for col in target_cols:
        if col not in cleaned_df.columns:
            continue
        series = cleaned_df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        is_date_col = False
        if columns and col in columns:
            is_date_col = True
        else:
            sem_type, details = infer_column_semantic_type(series)
            if sem_type == "datetime" or details.get("is_string_encoded_date"):
                is_date_col = True

        if is_date_col:
            try:
                parsed_dates = pd.to_datetime(series, format="mixed", errors="coerce")
                valid_count = int(parsed_dates.dropna().count())
                if valid_count > 0:
                    cleaned_df[col] = parsed_dates
                    failed_count = int(parsed_dates.isna().sum()) - int(series.isna().sum())
                    changed_mask = series.notna() & series.apply(
                        lambda v: not pd.api.types.is_datetime64_any_dtype(type(v))
                    )
                    changed_count = int(changed_mask.sum())
                    logs.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": "clean_datetime",
                            "column": col,
                            "method": "mixed_format_inference",
                            "rows_affected": changed_count,
                            "action": (
                                f"Standardized {changed_count} date entries in '{col}' to datetime"
                            ),
                            "details": {
                                "cells_changed": changed_count,
                                "parsed_count": valid_count,
                                "failed_parse_count": max(0, failed_count),
                                "unique_dates_count": int(parsed_dates.dropna().nunique()),
                                "min_date": str(parsed_dates.dropna().min()),
                                "max_date": str(parsed_dates.dropna().max()),
                            },
                        }
                    )
            except Exception as e:
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "clean_datetime_failed",
                        "column": col,
                        "method": "error_fallback",
                        "rows_affected": 0,
                        "action": f"Could not parse dates in '{col}': {str(e)}",
                        "details": {"error": str(e)},
                    }
                )

    return cleaned_df, logs


def handle_future_dates(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_date: Optional[Union[pd.Timestamp, datetime]] = None,
    action: str = "nullify",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Detect and handle dates that occur beyond allowable historical bounds.

    Args:
        df: Input DataFrame.
        columns: Optional list of datetime columns.
        max_date: Maximum allowable timestamp. Defaults to pd.Timestamp.now().
        action: 'nullify' (default: flag and set to NaT), 'cap' (cap at max valid historical date),
                or 'flag' (flag and retain).

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cleaned_df = df.copy()
    logs: List[Dict[str, Any]] = []

    target_cols = columns or [
        c for c in cleaned_df.columns if pd.api.types.is_datetime64_any_dtype(cleaned_df[c])
    ]
    threshold_date = pd.Timestamp(max_date) if max_date else pd.Timestamp(datetime.now())

    for col in target_cols:
        if col not in cleaned_df.columns:
            continue
        series = cleaned_df[col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            continue

        future_mask = series > threshold_date
        future_count = int(future_mask.sum())

        if future_count > 0:
            if action == "cap":
                valid_series = series[~future_mask].dropna()
                cap_target = (
                    pd.Timestamp(max_date)
                    if max_date
                    else (valid_series.max() if len(valid_series) > 0 else threshold_date)
                )
                cleaned_df.loc[future_mask, col] = cap_target
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "handle_future_dates",
                        "column": col,
                        "method": "flag_and_cap",
                        "rows_affected": future_count,
                        "action": (
                            f"Flagged and capped {future_count} future dates in '{col}' to "
                            f"{cap_target.strftime('%Y-%m-%d')}"
                        ),
                        "details": {
                            "future_dates_flagged": future_count,
                            "future_dates_count": future_count,
                            "cap_target": cap_target.strftime("%Y-%m-%d"),
                            "action": "cap",
                            "threshold_date": threshold_date.strftime("%Y-%m-%d"),
                        },
                    }
                )
            elif action == "nullify":
                cleaned_df.loc[future_mask, col] = pd.NaT
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "handle_future_dates",
                        "column": col,
                        "method": "flag_and_nullify",
                        "rows_affected": future_count,
                        "action": (
                            f"Flagged {future_count} future dates in '{col}' "
                            f"and set to missing (NaT)"
                        ),
                        "details": {
                            "future_dates_flagged": future_count,
                            "future_dates_count": future_count,
                            "dates_set_to_missing": future_count,
                            "action": "nullify",
                            "threshold_date": threshold_date.strftime("%Y-%m-%d"),
                        },
                    }
                )
            elif action == "flag":
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "handle_future_dates",
                        "column": col,
                        "method": "flag_future_dates",
                        "rows_affected": future_count,
                        "action": f"Flagged {future_count} future dates in '{col}' (retained)",
                        "details": {
                            "future_dates_flagged": future_count,
                            "future_dates_count": future_count,
                            "action": "flag",
                            "threshold_date": threshold_date.strftime("%Y-%m-%d"),
                        },
                    }
                )

    return cleaned_df, logs


def standardize_text(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Standardize string/categorical columns: strip whitespace, title-case, merge synonyms.

    Specifically:
    - Merges region variants (east/East/EAST) down to 4 canonical title-case regions.
    - Merges gender variants (male/MALE/M/Male, female/FEMALE/F/Female) down to 2 values.

    Args:
        df: Input DataFrame.
        columns: Optional list of text columns to standardize. If None, targets
                 all object/categorical columns excluding ID-like columns.

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cleaned_df = df.copy()
    logs: List[Dict[str, Any]] = []

    target_cols = columns or list(cleaned_df.columns)

    for col in target_cols:
        if col not in cleaned_df.columns:
            continue
        series = cleaned_df[col]

        is_text_type = (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if not is_text_type:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        sem_type, _ = infer_column_semantic_type(series)
        if sem_type in ("id_like", "numeric", "datetime"):
            continue

        orig_unique = int(series.dropna().nunique())

        # 1. Whitespace stripping and internal space compression
        cleaned_series = (
            series.dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        col_lower = col.lower()

        # 2. Gender synonym merging (Male / Female)
        if "gender" in col_lower or "sex" in col_lower:
            cleaned_series = cleaned_series.apply(
                lambda x: GENDER_SYNONYMS.get(x.lower(), x.title())
            )
        # 3. Constant columns: do not title-case constant codes / single-value columns
        elif sem_type == "constant" or orig_unique <= 1 or cleaned_series.nunique() <= 1:
            pass
        # 4. Region and general categoricals: standardize to Title Case if casing variants exist
        elif cleaned_series.str.lower().nunique() < cleaned_series.nunique():
            cleaned_series = cleaned_series.str.title()

        # Count cells that actually changed before overwriting
        orig_values = series.dropna().astype(str)
        changed_mask = orig_values != cleaned_series
        cells_changed = int(changed_mask.sum())

        cleaned_df[col] = cleaned_df[col].copy()
        cleaned_df.loc[cleaned_series.index, col] = cleaned_series

        new_unique = int(cleaned_df[col].dropna().nunique())

        method_name = (
            "whitespace_trim"
            if (sem_type == "constant" or orig_unique <= 1 or cleaned_series.nunique() <= 1)
            else "title_case_and_synonyms"
        )

        logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": "standardize_text",
                "column": col,
                "method": method_name,
                "rows_affected": cells_changed,
                "action": (
                    f"Standardized '{col}': {orig_unique} variants "
                    f"-> {new_unique} ({cells_changed} cells changed)"
                ),
                "details": {
                    "cells_changed": cells_changed,
                    "original_unique_values": orig_unique,
                    "standardized_unique_values": new_unique,
                    "canonical_values": cleaned_df[col].dropna().unique().tolist()[:10],
                },
            }
        )

    return cleaned_df, logs


def correct_negative_anomalies(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Correct negative values in inherently positive domains (e.g. age, quantity).

    Args:
        df: Input DataFrame.
        columns: Optional list of numeric columns. If None, inspects columns
                 known to be strictly positive (age, quantity, price, rating).

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cleaned_df = df.copy()
    logs: List[Dict[str, Any]] = []

    positive_keywords = ["age", "quantity", "unit_price", "price", "rating", "rating_score"]
    candidate_cols = columns or [
        c for c in cleaned_df.columns if any(k in c.lower() for k in positive_keywords)
    ]

    for col in candidate_cols:
        if col not in cleaned_df.columns:
            continue
        series = cleaned_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        neg_mask = series < 0
        neg_count = int(neg_mask.sum())
        if neg_count > 0:
            cleaned_df.loc[neg_mask, col] = np.nan
            logs.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step": "correct_negatives",
                    "column": col,
                    "method": "set_to_missing",
                    "rows_affected": neg_count,
                    "action": f"Set {neg_count} negative entries in '{col}' to missing",
                    "details": {
                        "negatives_flagged": neg_count,
                        "action": "set_to_missing",
                    },
                }
            )

    return cleaned_df, logs


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    factor: float = 3.0,
    action: str = "flag",
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Detect and treat numerical outliers using IQR or Z-score thresholds.

    Applies Tukey's extreme outlier boundary (default factor=3.0) and supports
    context-aware boundary evaluation (per product/category) to prevent flattening
    multimodal business metrics like catalog prices or diverse revenues.

    Args:
        df: Input DataFrame.
        columns: Optional list of numeric columns. If None, targets all numeric columns.
        method: Detection strategy ('iqr' or 'zscore').
        factor: Threshold multiplier (default 3.0 for extreme IQR fences, 3.0 for Z-score).
        action: Treatment action ('flag' to flag without mutating, 'clip' to cap at boundaries, 'remove', or 'nullify').
        group_col: Optional explicit column name to group by for context-aware boundaries.

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cleaned_df = df.copy()
    logs: List[Dict[str, Any]] = []

    target_cols = columns or [
        c for c in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[c])
    ]

    candidate_product_col = None
    candidate_category_col = None
    for c in cleaned_df.columns:
        c_lower = c.lower()
        if c_lower in ["product", "product_name", "item", "sku"]:
            candidate_product_col = c
        elif c_lower in ["category", "dept", "department", "segment"]:
            candidate_category_col = c

    for col in target_cols:
        if col not in cleaned_df.columns:
            continue
        series = cleaned_df[col].dropna()
        if len(series) < 10 or series.nunique() <= 2:
            continue

        col_lower = col.lower()
        is_non_negative_metric = any(
            k in col_lower for k in ["revenue", "price", "unit_price", "cost", "quantity", "sales"]
        )

        # For small integer columns (e.g. quantity), validate instead of IQR capping
        is_small_integer = (
            col_lower in ["quantity", "qty", "items_count"]
            or (
                pd.api.types.is_integer_dtype(series)
                and series.max() <= 100
                and series.nunique() <= 30
            )
        )
        if is_small_integer:
            invalid_mask = cleaned_df[col] < 1
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                cleaned_df.loc[invalid_mask, col] = np.nan
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "handle_outliers",
                        "column": col,
                        "method": "domain_validation",
                        "rows_affected": invalid_count,
                        "action": (
                            f"Validated small integer column '{col}' "
                            f"(flagged {invalid_count} values < 1)"
                        ),
                        "details": {
                            "method": "domain_validation",
                            "min_allowed": 1,
                            "lower_limit": 1,
                            "upper_limit": None,
                            "outliers_treated": invalid_count,
                            "action": "flag_invalid",
                        },
                    }
                )
            else:
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "handle_outliers",
                        "column": col,
                        "method": "domain_validation",
                        "rows_affected": 0,
                        "action": (
                            f"Validated small integer column '{col}' "
                            f"(all values >= 1; 0 flagged)"
                        ),
                        "details": {
                            "method": "domain_validation",
                            "min_allowed": 1,
                            "lower_limit": 1,
                            "upper_limit": None,
                            "outliers_treated": 0,
                            "action": "validated",
                        },
                    }
                )
            continue

        active_group = group_col
        if not active_group:
            if ("price" in col_lower or "cost" in col_lower) and candidate_product_col:
                active_group = candidate_product_col
            elif ("price" in col_lower or "cost" in col_lower) and candidate_category_col:
                active_group = candidate_category_col
            elif col_lower in ["revenue", "profit", "sales"] and candidate_category_col:
                active_group = candidate_category_col

        if active_group and active_group in cleaned_df.columns:
            total_outliers = 0
            lower_bounds = []
            upper_bounds = []

            for g_val, g_df in cleaned_df.groupby(active_group):
                g_series = g_df[col].dropna()
                if len(g_series) < 5 or g_series.nunique() <= 1:
                    continue

                if method == "iqr":
                    q1 = float(g_series.quantile(0.25))
                    q3 = float(g_series.quantile(0.75))
                    iqr = q3 - q1
                    if iqr <= 0:
                        continue
                    low = q1 - (factor * iqr)
                    high = q3 + (factor * iqr)
                elif method == "zscore":
                    mean = float(g_series.mean())
                    std = float(g_series.std())
                    if std <= 0:
                        continue
                    low = mean - (factor * std)
                    high = mean + (factor * std)
                else:
                    raise ValueError(f"Unknown outlier detection method '{method}'")

                # Ensure non-negative columns do not have negative lower bounds
                if is_non_negative_metric:
                    low = max(0.0, low)

                lower_bounds.append(low)
                upper_bounds.append(high)

                g_mask = (cleaned_df[active_group] == g_val) & (
                    (cleaned_df[col] < low) | (cleaned_df[col] > high)
                )
                g_count = int(g_mask.sum())
                if g_count > 0:
                    total_outliers += g_count
                    if action == "clip":
                        cleaned_df.loc[g_mask, col] = cleaned_df.loc[g_mask, col].clip(
                            lower=low, upper=high
                        )
                    elif action == "nullify":
                        cleaned_df.loc[g_mask, col] = np.nan
                    elif action == "remove":
                        cleaned_df = cleaned_df[~g_mask].copy().reset_index(drop=True)
                    elif action == "flag":
                        pass

            if total_outliers > 0:
                min_low = round(min(lower_bounds), 4) if lower_bounds else 0.0
                if is_non_negative_metric:
                    min_low = max(0.0, min_low)
                max_high = round(max(upper_bounds), 4) if upper_bounds else 0.0
                total_rows = len(cleaned_df)
                pct_capped = (
                    round((total_outliers / total_rows) * 100, 2) if total_rows > 0 else 0.0
                )
                exceeds_threshold = pct_capped > 5.0
                flag_str = " [FLAGGED: >5% capped]" if exceeds_threshold else ""
                rows_aff = 0 if action == "flag" else total_outliers
                action_verb = "Flagged" if action == "flag" else "Capped"
                ret_str = " (values retained)" if action == "flag" else ""
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "step": "handle_outliers",
                        "column": col,
                        "method": f"{method}_grouped_by_{active_group}",
                        "rows_affected": rows_aff,
                        "action": (
                            f"{action_verb} {total_outliers} outliers ({pct_capped}%) in '{col}' "
                            f"grouped by '{active_group}' "
                            f"[limits: {min_low} to {max_high}]{flag_str}{ret_str}"
                        ),
                        "details": {
                            "method": method,
                            "factor": factor,
                            "group_by": active_group,
                            "lower_limit": min_low,
                            "upper_limit": max_high,
                            "outliers_treated": rows_aff,
                            "outliers_flagged": total_outliers,
                            "total_rows": total_rows,
                            "percent_capped": pct_capped,
                            "capped_percent": pct_capped,
                            "flagged_exceeds_5pct": exceeds_threshold,
                            "action": action,
                        },
                    }
                )
        else:
            if method == "iqr":
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                if iqr <= 0:
                    continue
                lower_bound = q1 - (factor * iqr)
                upper_bound = q3 + (factor * iqr)
            elif method == "zscore":
                mean = float(series.mean())
                std = float(series.std())
                if std <= 0:
                    continue
                lower_bound = mean - (factor * std)
                upper_bound = mean + (factor * std)
            else:
                raise ValueError(f"Unknown outlier detection method '{method}'")

            # Ensure non-negative columns do not have negative lower bounds
            if is_non_negative_metric:
                lower_bound = max(0.0, lower_bound)

            outlier_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
            outlier_count = int(outlier_mask.sum())

            if outlier_count > 0:
                total_rows = len(cleaned_df)
                pct_treated = (
                    round((outlier_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
                )
                exceeds_threshold = pct_treated > 5.0
                flag_str = " [FLAGGED: >5% capped]" if exceeds_threshold else ""

                if action == "clip":
                    cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                    logs.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": "handle_outliers",
                            "column": col,
                            "method": f"{method}_clipping",
                            "rows_affected": outlier_count,
                            "action": (
                                f"Capped {outlier_count} outliers ({pct_treated}%) in '{col}' "
                                f"using {method.upper()} "
                                f"[limits: {round(lower_bound, 4)} to {round(upper_bound, 4)}]"
                                f"{flag_str}"
                            ),
                            "details": {
                                "method": method,
                                "factor": factor,
                                "lower_limit": round(lower_bound, 4),
                                "upper_limit": round(upper_bound, 4),
                                "outliers_treated": outlier_count,
                                "total_rows": total_rows,
                                "percent_capped": pct_treated,
                                "capped_percent": pct_treated,
                                "flagged_exceeds_5pct": exceeds_threshold,
                                "action": action,
                            },
                        }
                    )
                elif action == "flag":
                    logs.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": "handle_outliers",
                            "column": col,
                            "method": f"{method}_flagging",
                            "rows_affected": 0,
                            "action": (
                                f"Flagged {outlier_count} outliers ({pct_treated}%) in '{col}' "
                                f"using {method.upper()} "
                                f"[limits: {round(lower_bound, 4)} to {round(upper_bound, 4)}]"
                                f"{flag_str} (values retained)"
                            ),
                            "details": {
                                "method": method,
                                "factor": factor,
                                "lower_limit": round(lower_bound, 4),
                                "upper_limit": round(upper_bound, 4),
                                "outliers_treated": 0,
                                "outliers_flagged": outlier_count,
                                "total_rows": total_rows,
                                "percent_capped": pct_treated,
                                "capped_percent": pct_treated,
                                "flagged_exceeds_5pct": exceeds_threshold,
                                "action": action,
                            },
                        }
                    )
                elif action == "nullify":
                    cleaned_df.loc[outlier_mask, col] = np.nan
                    logs.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": "handle_outliers",
                            "column": col,
                            "method": f"{method}_nullify",
                            "rows_affected": outlier_count,
                            "action": (
                                f"Set {outlier_count} outliers ({pct_treated}%) in '{col}' "
                                f"to NaN{flag_str}"
                            ),
                            "details": {
                                "method": method,
                                "factor": factor,
                                "lower_limit": round(lower_bound, 4),
                                "upper_limit": round(upper_bound, 4),
                                "outliers_treated": outlier_count,
                                "total_rows": total_rows,
                                "percent_capped": pct_treated,
                                "capped_percent": pct_treated,
                                "flagged_exceeds_5pct": exceeds_threshold,
                                "action": action,
                            },
                        }
                    )
                elif action == "remove":
                    cleaned_df = cleaned_df[~outlier_mask].copy().reset_index(drop=True)
                    logs.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": "handle_outliers",
                            "column": col,
                            "method": f"{method}_row_removal",
                            "rows_affected": outlier_count,
                            "action": (
                                f"Dropped {outlier_count} rows with outliers ({pct_treated}%) "
                                f"in '{col}'{flag_str}"
                            ),
                            "details": {
                                "method": method,
                                "factor": factor,
                                "lower_limit": round(lower_bound, 4),
                                "upper_limit": round(upper_bound, 4),
                                "outliers_treated": outlier_count,
                                "total_rows": total_rows,
                                "percent_capped": pct_treated,
                                "capped_percent": pct_treated,
                                "flagged_exceeds_5pct": exceeds_threshold,
                                "action": action,
                            },
                        }
                    )

    return cleaned_df, logs


def impute_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    columns: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Impute missing values in numeric and categorical columns.

    Args:
        df: Input DataFrame.
        numeric_strategy: Default strategy for numeric columns ('median', 'mean', 'zero').
        categorical_strategy: Default strategy for categoricals ('mode', 'constant').
        columns: Optional dictionary mapping column names to custom strategies.

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cleaned_df = df.copy()
    logs: List[Dict[str, Any]] = []

    custom_rules = columns or {}

    for col in cleaned_df.columns:
        series = cleaned_df[col]
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue

        strategy = custom_rules.get(col)
        imputed_val: Any = None

        if pd.api.types.is_numeric_dtype(series):
            eff_strategy = strategy or numeric_strategy
            if eff_strategy == "median":
                imputed_val = series.median()
                if pd.api.types.is_integer_dtype(series):
                    imputed_val = round(float(imputed_val))
            elif eff_strategy == "mean":
                imputed_val = series.mean()
            elif eff_strategy == "zero":
                imputed_val = 0
            else:
                imputed_val = series.median()

            cleaned_df[col] = series.fillna(imputed_val)

        elif pd.api.types.is_datetime64_any_dtype(series):
            eff_strategy = custom_rules.get(col)
            if eff_strategy:
                cleaned_df[col] = series.ffill().bfill()
            else:
                continue

        else:
            eff_strategy = strategy or categorical_strategy
            if eff_strategy == "mode":
                mode_vals = series.mode()
                imputed_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            elif eff_strategy == "constant":
                imputed_val = "Missing"
            else:
                mode_vals = series.mode()
                imputed_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"

            cleaned_df[col] = series.fillna(imputed_val)

        logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": "impute_missing",
                "column": col,
                "method": eff_strategy,
                "rows_affected": missing_count,
                "action": f"Imputed {missing_count} missing values in '{col}' using {eff_strategy}",
                "details": {
                    "strategy": eff_strategy,
                    "missing_imputed": missing_count,
                    "imputed_value": str(imputed_val),
                },
            }
        )

    return cleaned_df, logs


def clean_data(
    df: pd.DataFrame, config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Execute complete, non-destructive data cleaning pipeline with comprehensive audit log.

    Orchestrates:
    1. Duplicate row removal
    2. Structural defect handling (drops 100% null columns)
    3. Currency string parsing to float64
    4. Mixed datetime parsing and standardization
    5. Future date anomaly detection and treatment
    6. Text standardization (whitespace trimming, case normalization, category canonicalization)
    7. Negative value correction for positive domain metrics (prior to imputation)
    8. Outlier detection and treatment (IQR boundary clipping)
    9. Missing value imputation

    Args:
        df: Raw input DataFrame (guaranteed not modified).
        config: Optional configuration overrides. Supports:
            - 'deduplicate': bool (default True)
            - 'drop_all_null': bool (default True)
            - 'drop_constant': bool (default False)
            - 'clean_currency': bool (default True)
            - 'clean_datetime': bool (default True)
            - 'handle_future_dates': bool (default True)
            - 'future_date_action': 'nullify', 'cap', 'flag' (default 'nullify')
            - 'standardize_text': bool (default True)
            - 'correct_negatives': bool (default True)
            - 'handle_outliers': bool (default True)
            - 'outlier_method': 'iqr' or 'zscore' (default 'iqr')
            - 'outlier_factor': float (default 3.0)
            - 'outlier_action': 'flag', 'clip', 'remove', 'nullify' (default 'flag')
            - 'impute_missing': bool (default True)
            - 'numeric_impute_strategy': 'median' or 'mean' (default 'median')
            - 'categorical_impute_strategy': 'mode' or 'constant' (default 'mode')

    Returns:
        Tuple of (cleaned DataFrame copy, list of audit log entries).
    """
    cfg = config or {}
    cleaned_df = df.copy(deep=True)
    audit_trail: List[Dict[str, Any]] = []

    # Track top numeric pair correlation prior to cleaning
    init_numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    top_pair = None
    if len(init_numeric_cols) >= 2:
        try:
            corr_init = df[init_numeric_cols].corr()
            best_abs_r = -1.0
            for i, c1 in enumerate(init_numeric_cols):
                for j, c2 in enumerate(init_numeric_cols):
                    if i < j:
                        r_val = corr_init.loc[c1, c2]
                        if not np.isnan(r_val) and abs(r_val) > best_abs_r:
                            best_abs_r = abs(r_val)
                            top_pair = (c1, c2, float(r_val))
        except Exception:
            top_pair = None

    # 1. Deduplication
    if cfg.get("deduplicate", True):
        cleaned_df, dup_log = deduplicate_data(cleaned_df)
        if dup_log["details"]["rows_removed"] > 0:
            audit_trail.append(dup_log)

    # 2. Structural defects (all-null and constant columns)
    cleaned_df, struct_logs = handle_structural_defects(
        cleaned_df,
        drop_all_null=cfg.get("drop_all_null", True),
        drop_constant=cfg.get("drop_constant", False),
    )
    audit_trail.extend(struct_logs)

    # 3. Currency and numeric strings
    if cfg.get("clean_currency", True):
        cleaned_df, curr_logs = clean_currency_and_numeric_strings(cleaned_df)
        audit_trail.extend(curr_logs)

    # 4. Mixed datetime formats
    if cfg.get("clean_datetime", True):
        cleaned_df, date_logs = clean_datetime_columns(cleaned_df)
        audit_trail.extend(date_logs)

    # 5. Future date anomaly detection and treatment
    if cfg.get("handle_future_dates", True):
        cleaned_df, future_logs = handle_future_dates(
            cleaned_df,
            action=cfg.get("future_date_action", "nullify"),
        )
        audit_trail.extend(future_logs)

    # 6. Text standardization
    if cfg.get("standardize_text", True):
        cleaned_df, text_logs = standardize_text(cleaned_df)
        audit_trail.extend(text_logs)

    # 7. Correct negative anomalies (MUST happen before imputation so median is uncorrupted)
    if cfg.get("correct_negatives", True):
        cleaned_df, neg_logs = correct_negative_anomalies(cleaned_df)
        audit_trail.extend(neg_logs)

    # 8. Outliers (default: flag without mutating; clips only if explicitly configured)
    if cfg.get("handle_outliers", True):
        cleaned_df, outlier_logs = handle_outliers(
            cleaned_df,
            method=cfg.get("outlier_method", "iqr"),
            factor=cfg.get("outlier_factor", 3.0),
            action=cfg.get("outlier_action", "flag"),
        )
        audit_trail.extend(outlier_logs)

    # 9. Missing value imputation
    if cfg.get("impute_missing", True):
        cleaned_df, impute_logs = impute_missing_values(
            cleaned_df,
            numeric_strategy=cfg.get("numeric_impute_strategy", "median"),
            categorical_strategy=cfg.get("categorical_impute_strategy", "mode"),
        )
        audit_trail.extend(impute_logs)

    # 10. Audit correlation before and after cleaning for top numeric pair
    if top_pair:
        c1, c2, r_before = top_pair
        if c1 in cleaned_df.columns and c2 in cleaned_df.columns:
            try:
                sub = cleaned_df[[c1, c2]].dropna()
                if len(sub) >= 2:
                    r_after = float(sub.corr().iloc[0, 1])
                    audit_trail.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "step": "correlation_audit",
                            "column": f"{c1} vs {c2}",
                            "method": "pearson_correlation",
                            "rows_affected": 0,
                            "action": (
                                f"Audited top numeric correlation ({c1} vs {c2}): "
                                f"before r = {r_before:.4f}, after r = {r_after:.4f} "
                                f"(delta = {r_after - r_before:+.4f})"
                            ),
                            "details": {
                                "column_1": c1,
                                "column_2": c2,
                                "r_before": round(r_before, 4),
                                "r_after": round(r_after, 4),
                                "delta": round(r_after - r_before, 4),
                            },
                        }
                    )
            except Exception:
                pass

    return cleaned_df, audit_trail
