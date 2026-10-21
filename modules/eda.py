"""Exploratory Data Analysis Module for AutoAnalyst.

Computes comprehensive univariate distributions, bivariate correlations,
hypothesis tests (ANOVA, Chi-square), and time-series trend and seasonality
decomposition, driven purely by detected semantic types without hardcoded column names.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from modules.profiling import infer_column_semantic_type


def classify_columns(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Classify DataFrame columns into semantic buckets and identify excluded columns.

    Constant columns and ID-like columns are excluded from statistical analysis and charts.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (classified_dict, excluded_dict) where:
            - classified_dict maps semantic types to column lists.
            - excluded_dict maps excluded column names to the reason for exclusion.
    """
    classified: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": [],
    }
    excluded: Dict[str, str] = {}

    for col in df.columns:
        series = df[col]
        sem_type, _ = infer_column_semantic_type(series)

        if sem_type == "constant":
            excluded[col] = "Constant column (0 or 1 unique non-null value); excluded from stats and charts"
        elif sem_type == "id_like":
            excluded[col] = "Identifier/ID-like column (high-cardinality unique codes); excluded from stats and charts"
        elif sem_type == "numeric":
            classified["numeric"].append(col)
        elif sem_type == "datetime":
            classified["datetime"].append(col)
        elif sem_type in ("categorical", "boolean"):
            classified["categorical"].append(col)
        elif sem_type == "text":
            classified["text"].append(col)

    return classified, excluded


def compute_univariate_numeric(series: pd.Series) -> Dict[str, Any]:
    """Compute detailed univariate descriptive statistics and distribution metrics for a numeric series.

    Stats match Pandas and SciPy directly.

    Args:
        series: Numeric Pandas Series.

    Returns:
        Dictionary of statistical metrics.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)

    if n == 0:
        return {
            "count": 0,
            "missing_count": int(series.isna().sum()),
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
            "is_normal": False,
            "normality_pvalue": None,
            "outliers_count": 0,
            "zeros_count": 0,
        }

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if n > 1 else 0.0
    median_val = float(clean.median())
    min_val = float(clean.min())
    max_val = float(clean.max())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    iqr = float(q75 - q25)

    skewness = float(clean.skew()) if n >= 3 else 0.0
    kurtosis = float(clean.kurtosis()) if n >= 4 else 0.0

    # SciPy normality test (D'Agostino and Pearson's test) requires >= 8 observations
    is_normal = False
    normality_pvalue = None
    if n >= 8 and std_val > 0:
        try:
            stat, p_val = stats.normaltest(clean.values)
            normality_pvalue = float(p_val)
            is_normal = bool(p_val > 0.05)
        except Exception:
            is_normal = False
            normality_pvalue = None

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers_count = int(((clean < lower_bound) | (clean > upper_bound)).sum())
    zeros_count = int((clean == 0).sum())

    return {
        "count": n,
        "missing_count": int(series.isna().sum()),
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_normal": is_normal,
        "normality_pvalue": normality_pvalue,
        "outliers_count": outliers_count,
        "zeros_count": zeros_count,
    }


def compute_univariate_categorical(series: pd.Series, top_n: int = 10) -> Dict[str, Any]:
    """Compute univariate summary statistics and frequencies for a categorical series.

    Args:
        series: Categorical Pandas Series.
        top_n: Maximum frequency levels to return.

    Returns:
        Dictionary of categorical metrics.
    """
    clean = series.dropna().astype(str)
    n = len(clean)

    if n == 0:
        return {
            "count": 0,
            "missing_count": int(series.isna().sum()),
            "cardinality": 0,
            "top_value": None,
            "top_freq": 0,
            "top_percentage": 0.0,
            "entropy": 0.0,
            "frequency_table": {},
        }

    val_counts = clean.value_counts()
    cardinality = len(val_counts)
    top_value = str(val_counts.index[0])
    top_freq = int(val_counts.iloc[0])
    top_percentage = float((top_freq / n) * 100.0)

    # Shannon entropy
    probs = val_counts.values / n
    entropy_val = float(stats.entropy(probs, base=2))

    top_dict = {str(k): int(v) for k, v in val_counts.head(top_n).items()}

    return {
        "count": n,
        "missing_count": int(series.isna().sum()),
        "cardinality": cardinality,
        "top_value": top_value,
        "top_freq": top_freq,
        "top_percentage": top_percentage,
        "entropy": entropy_val,
        "frequency_table": top_dict,
    }


def compute_univariate_datetime(series: pd.Series) -> Dict[str, Any]:
    """Compute univariate summary statistics for a datetime series, safely skipping NaT values.

    Args:
        series: Datetime Pandas Series.

    Returns:
        Dictionary of datetime metrics.
    """
    dt_series = pd.to_datetime(series, errors="coerce")
    valid_dates = dt_series.dropna()
    n = len(valid_dates)
    missing_count = int(series.isna().sum()) + int(dt_series.isna().sum() - series.isna().sum())

    if n == 0:
        return {
            "count": 0,
            "missing_count": missing_count,
            "min_date": None,
            "max_date": None,
            "range_days": 0,
        }

    min_date = valid_dates.min()
    max_date = valid_dates.max()
    range_days = int((max_date - min_date).days) if pd.notna(min_date) and pd.notna(max_date) else 0

    return {
        "count": n,
        "missing_count": int(dt_series.isna().sum()),
        "min_date": str(min_date.date()) if pd.notna(min_date) else None,
        "max_date": str(max_date.date()) if pd.notna(max_date) else None,
        "range_days": range_days,
    }


def compute_correlations(
    df: pd.DataFrame, numeric_cols: List[str]
) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlation matrices and pairwise significance.

    Args:
        df: DataFrame.
        numeric_cols: List of numeric column names.

    Returns:
        Dictionary containing pearson, spearman matrices, p-values, and ranked strong pairs.
    """
    if len(numeric_cols) < 2:
        return {
            "pearson": pd.DataFrame(),
            "spearman": pd.DataFrame(),
            "pairs": [],
        }

    sub_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    pearson_mat = sub_df.corr(method="pearson")
    spearman_mat = sub_df.corr(method="spearman")

    pairs: List[Dict[str, Any]] = []
    cols = list(numeric_cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            valid = sub_df[[col1, col2]].dropna()
            if len(valid) >= 3:
                try:
                    p_coeff, p_pval = stats.pearsonr(valid[col1], valid[col2])
                    s_coeff, s_pval = stats.spearmanr(valid[col1], valid[col2])
                except Exception:
                    p_coeff, p_pval = float(pearson_mat.loc[col1, col2]), 1.0
                    s_coeff, s_pval = float(spearman_mat.loc[col1, col2]), 1.0

                pairs.append(
                    {
                        "col1": col1,
                        "col2": col2,
                        "pearson_r": float(p_coeff) if not np.isnan(p_coeff) else 0.0,
                        "pearson_p": float(p_pval) if not np.isnan(p_pval) else 1.0,
                        "spearman_rho": float(s_coeff) if not np.isnan(s_coeff) else 0.0,
                        "spearman_p": float(s_pval) if not np.isnan(s_pval) else 1.0,
                    }
                )

    # Sort pairs by absolute Pearson correlation descending
    pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    return {
        "pearson": pearson_mat,
        "spearman": spearman_mat,
        "pairs": pairs,
    }


def compute_group_comparisons(
    df: pd.DataFrame, cat_col: str, num_col: str
) -> Dict[str, Any]:
    """Compute ANOVA F-test / Kruskal-Wallis test and group summary statistics for categorical vs numeric.

    Args:
        df: Input DataFrame.
        cat_col: Categorical grouping column.
        num_col: Numeric measurement column.

    Returns:
        Dictionary containing group summaries and hypothesis test results.
    """
    valid = df[[cat_col, num_col]].dropna()
    valid[num_col] = pd.to_numeric(valid[num_col], errors="coerce")
    valid = valid.dropna()

    if len(valid) == 0:
        return {"groups": {}, "anova_f": None, "anova_p": None, "significant": False}

    grouped = valid.groupby(cat_col)[num_col]
    group_stats: Dict[str, Dict[str, float]] = {}
    group_arrays = []

    for name, grp in grouped:
        vals = grp.values
        if len(vals) > 0:
            group_stats[str(name)] = {
                "count": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
            }
            if len(vals) >= 2:
                group_arrays.append(vals)

    anova_f, anova_p = None, None
    significant = False
    if len(group_arrays) >= 2:
        try:
            f_stat, p_val = stats.f_oneway(*group_arrays)
            if not np.isnan(f_stat) and not np.isnan(p_val):
                anova_f = float(f_stat)
                anova_p = float(p_val)
                significant = bool(p_val < 0.05)
        except Exception:
            anova_f, anova_p = None, None

    return {
        "cat_col": cat_col,
        "num_col": num_col,
        "groups": group_stats,
        "anova_f": anova_f,
        "anova_p": anova_p,
        "significant": significant,
    }


def compute_categorical_association(
    df: pd.DataFrame, cat1: str, cat2: str
) -> Dict[str, Any]:
    """Compute Chi-Square test of independence and Cramér's V between two categorical columns.

    Args:
        df: Input DataFrame.
        cat1: First categorical column.
        cat2: Second categorical column.

    Returns:
        Dictionary with crosstab, chi2 statistic, p-value, and Cramér's V.
    """
    valid = df[[cat1, cat2]].dropna()
    if len(valid) == 0 or valid[cat1].nunique() < 2 or valid[cat2].nunique() < 2:
        return {"chi2": None, "p_value": None, "cramers_v": None, "significant": False}

    contingency = pd.crosstab(valid[cat1], valid[cat2])
    try:
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        n = len(valid)
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0
        return {
            "cat1": cat1,
            "cat2": cat2,
            "chi2": float(chi2),
            "p_value": float(p_val),
            "dof": int(dof),
            "cramers_v": float(cramers_v),
            "significant": bool(p_val < 0.05),
        }
    except Exception:
        return {"chi2": None, "p_value": None, "cramers_v": None, "significant": False}


def get_timeseries_aggregation(metric_name: str) -> Tuple[str, str]:
    """Determine whether a time-series metric should be aggregated by mean or sum.

    Semantic rules:
    - 'mean' for rate, average, intensity, or bounded columns: age, rating, discount, margin, price, percentage.
    - 'sum' for volumetric throughput / accumulator columns: revenue, profit, quantity, sales, cost, units, orders.

    Args:
        metric_name: Name of the numeric metric column.

    Returns:
        Tuple of (agg_function, agg_display_label), e.g. ('mean', 'Average') or ('sum', 'Daily Total').
    """
    name_lower = metric_name.lower()
    mean_keywords = [
        "age", "rating", "discount", "margin", "price", "unit_price",
        "score", "ratio", "pct", "percent", "percentage", "average", "avg", "rate"
    ]
    sum_keywords = [
        "revenue", "profit", "quantity", "sales", "cost", "unit", "units",
        "order", "orders", "volume", "total", "count", "amount"
    ]
    for kw in mean_keywords:
        if kw in name_lower:
            return "mean", "Average"
    for kw in sum_keywords:
        if kw in name_lower:
            return "sum", "Daily Total"
    return "sum", "Daily Total"


def compute_timeseries_analysis(
    df: pd.DataFrame, dt_col: str, num_col: str
) -> Dict[str, Any]:
    """Decompose and evaluate time-series trends, seasonality, and rolling statistics.

    Safely skips NaT date entries. Aggregates daily by mean or sum based on metric semantics.

    Args:
        df: Input DataFrame.
        dt_col: Datetime column name.
        num_col: Numeric metric column name.

    Returns:
        Dictionary containing trend slope, R2, p-value, seasonality indicators, and resampled series.
    """
    sub_df = df[[dt_col, num_col]].copy()
    sub_df[dt_col] = pd.to_datetime(sub_df[dt_col], errors="coerce")
    sub_df[num_col] = pd.to_numeric(sub_df[num_col], errors="coerce")

    # Drop NaT and NaN rows safely
    valid = sub_df.dropna().sort_values(dt_col)
    n = len(valid)

    agg_func, agg_label = get_timeseries_aggregation(num_col)

    if n < 5:
        return {
            "dt_col": dt_col,
            "num_col": num_col,
            "agg_func": agg_func,
            "agg_label": agg_label,
            "count": n,
            "has_trend": False,
            "trend_direction": "flat",
            "trend_strength": "flat",
            "trend_card_label": "Flat / No trend (R² = 0.00)",
            "trend_summary_label": "no clear trend (R² = 0.00)",
            "trend_slope": 0.0,
            "trend_r2": 0.0,
            "trend_pvalue": 1.0,
            "has_seasonality": False,
            "seasonal_period": None,
            "resampled_daily": pd.DataFrame(),
        }

    # Aggregate by date according to metric semantics (mean for rates/averages; sum for volumes)
    if agg_func == "mean":
        daily = valid.set_index(dt_col).resample("D")[num_col].mean().dropna().reset_index()
    else:
        daily = valid.set_index(dt_col).resample("D")[num_col].sum().dropna().reset_index()

    if len(daily) < 5:
        return {
            "dt_col": dt_col,
            "num_col": num_col,
            "agg_func": agg_func,
            "agg_label": agg_label,
            "count": n,
            "has_trend": False,
            "trend_direction": "flat",
            "trend_strength": "flat",
            "trend_card_label": "Flat / No trend (R² = 0.00)",
            "trend_summary_label": "no clear trend (R² = 0.00)",
            "trend_slope": 0.0,
            "trend_r2": 0.0,
            "trend_pvalue": 1.0,
            "has_seasonality": False,
            "seasonal_period": None,
            "resampled_daily": daily,
        }

    # Linear trend estimation via OLS
    x_ordinals = (daily[dt_col] - daily[dt_col].iloc[0]).dt.days.values
    y_values = daily[num_col].values

    slope, intercept, r_val, p_val, _ = stats.linregress(x_ordinals, y_values)
    r2 = float(r_val ** 2) if not np.isnan(r_val) else 0.0
    p_val_flt = float(p_val) if not np.isnan(p_val) else 1.0
    slope_flt = float(slope) if not np.isnan(slope) else 0.0

    # Trend strength classification based on R²:
    # R² < 0.3: weak
    # 0.3 ≤ R² ≤ 0.6: moderate
    # R² > 0.6: strong
    if r2 > 0.6:
        trend_strength = "strong"
    elif r2 >= 0.3:
        trend_strength = "moderate"
    else:
        trend_strength = "weak"

    has_trend = bool(p_val_flt < 0.05 and abs(slope_flt) > 1e-5 and r2 >= 0.02)
    if has_trend:
        direction = "upward" if slope_flt > 0 else "downward"
        trend_direction = f"{trend_strength} {direction}"
        trend_card_label = f"{trend_strength.capitalize()} {direction} trend (R² = {r2:.2f})"
        trend_summary_label = f"{trend_strength} {direction} trend (R² = {r2:.2f})"
    else:
        trend_direction = "flat"
        trend_strength = "flat"
        trend_card_label = f"Flat / No trend (R² = {r2:.2f})"
        trend_summary_label = f"no clear trend (R² = {r2:.2f})"

    # Multi-frequency seasonality testing (Weekly, Monthly, Yearly)
    detected_seasonalities: List[str] = []

    if len(daily) >= 14:
        # 1. Weekly seasonality (lag 7 autocorrelation or day-of-week ANOVA)
        autocorr_7 = daily[num_col].autocorr(lag=7)
        daily["_dow"] = daily[dt_col].dt.dayofweek
        dow_groups = [grp[num_col].values for _, grp in daily.groupby("_dow") if len(grp) >= 2]
        dow_p = 1.0
        if len(dow_groups) >= 2:
            try:
                _, dow_p = stats.f_oneway(*dow_groups)
            except Exception:
                dow_p = 1.0

        if (not np.isnan(autocorr_7) and autocorr_7 > 0.12) or dow_p < 0.05:
            detected_seasonalities.append("weekly (7 days)")

        # 2. Monthly seasonality (lag 28-31 autocorrelation or day-of-month ANOVA)
        if len(daily) >= 60:
            autocorr_30 = daily[num_col].autocorr(lag=30)
            daily["_dom"] = daily[dt_col].dt.day
            dom_groups = [grp[num_col].values for _, grp in daily.groupby("_dom") if len(grp) >= 2]
            dom_p = 1.0
            if len(dom_groups) >= 2:
                try:
                    _, dom_p = stats.f_oneway(*dom_groups)
                except Exception:
                    dom_p = 1.0

            lag_m_max = max(
                [
                    daily[num_col].autocorr(lag=lag)
                    for lag in [28, 29, 30, 31]
                    if not np.isnan(daily[num_col].autocorr(lag=lag))
                ]
                or [0.0]
            )

            if lag_m_max > 0.10 or dom_p < 0.05 or (not np.isnan(autocorr_30) and autocorr_30 > 0.10):
                detected_seasonalities.append("monthly (~30 days)")

        # 3. Yearly / Annual seasonality (month-of-year averages, ANOVA, or lag 365 autocorrelation)
        if len(daily) >= 365:
            daily["_month"] = daily[dt_col].dt.month
            month_groups = [grp[num_col].values for _, grp in daily.groupby("_month") if len(grp) >= 2]
            month_p = 1.0
            if len(month_groups) >= 2:
                try:
                    _, month_p = stats.f_oneway(*month_groups)
                except Exception:
                    month_p = 1.0

            # Month-of-year averages variation
            monthly_means = daily.groupby("_month")[num_col].mean()
            has_month_spread = False
            if len(monthly_means) >= 6 and monthly_means.mean() > 0:
                cv = float(monthly_means.std() / monthly_means.mean())
                if cv > 0.10 and month_p < 0.10:
                    has_month_spread = True

            autocorr_365 = daily[num_col].autocorr(lag=365) if len(daily) >= 730 else np.nan
            if month_p < 0.05 or has_month_spread or (not np.isnan(autocorr_365) and autocorr_365 > 0.10):
                detected_seasonalities.append("yearly (365 days)")

    has_seasonality = len(detected_seasonalities) > 0
    seasonal_period = " & ".join(detected_seasonalities) if detected_seasonalities else None

    return {
        "dt_col": dt_col,
        "num_col": num_col,
        "agg_func": agg_func,
        "agg_label": agg_label,
        "count": n,
        "has_trend": has_trend,
        "trend_direction": trend_direction,
        "trend_strength": trend_strength,
        "trend_card_label": trend_card_label,
        "trend_summary_label": trend_summary_label,
        "trend_slope": slope_flt,
        "trend_r2": r2,
        "trend_pvalue": p_val_flt,
        "has_seasonality": has_seasonality,
        "seasonal_period": seasonal_period,
        "resampled_daily": daily[[dt_col, num_col]],
    }


def run_eda(df_clean: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """Execute complete automated exploratory data analysis on a cleaned dataset.

    Excludes constant and ID-like columns automatically. Driven purely by detected column types.

    Args:
        df_clean: Cleaned DataFrame.
        target: Optional target column name for supervised feature analysis.

    Returns:
        Comprehensive dictionary of statistical results and metadata.
    """
    classified, excluded = classify_columns(df_clean)

    numeric_cols = classified["numeric"]
    cat_cols = classified["categorical"]
    dt_cols = classified["datetime"]

    # 1. Univariate distributions
    univariate_num: Dict[str, Dict[str, Any]] = {}
    for col in numeric_cols:
        univariate_num[col] = compute_univariate_numeric(df_clean[col])

    univariate_cat: Dict[str, Dict[str, Any]] = {}
    for col in cat_cols:
        univariate_cat[col] = compute_univariate_categorical(df_clean[col])

    univariate_dt: Dict[str, Dict[str, Any]] = {}
    for col in dt_cols:
        univariate_dt[col] = compute_univariate_datetime(df_clean[col])

    # 2. Correlations (numeric vs numeric)
    correlations = compute_correlations(df_clean, numeric_cols)

    # 3. Group comparisons (categorical vs numeric)
    group_comparisons: List[Dict[str, Any]] = []
    for c_col in cat_cols[:4]:  # Limit pairs to top 4 categories to maintain efficiency
        for n_col in numeric_cols[:4]:
            comp = compute_group_comparisons(df_clean, c_col, n_col)
            if comp.get("anova_f") is not None:
                group_comparisons.append(comp)

    # 4. Categorical associations (categorical vs categorical)
    cat_associations: List[Dict[str, Any]] = []
    if len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                assoc = compute_categorical_association(df_clean, cat_cols[i], cat_cols[j])
                if assoc.get("chi2") is not None:
                    cat_associations.append(assoc)

    # 5. Time series analysis (if datetime + numeric exist)
    timeseries_results: List[Dict[str, Any]] = []
    if len(dt_cols) > 0 and len(numeric_cols) > 0:
        primary_dt = dt_cols[0]
        for n_col in numeric_cols:
            ts_res = compute_timeseries_analysis(df_clean, primary_dt, n_col)
            timeseries_results.append(ts_res)

    # 6. Optional PCA Dimensionality Reduction
    pca_results = compute_pca(df_clean, numeric_cols)

    # 7. Optional Target Analysis
    target_results = None
    if target and target in df_clean.columns:
        target_results = compute_target_analysis(df_clean, target, classified)

    return {
        "classified_columns": classified,
        "excluded_columns": excluded,
        "univariate_numeric": univariate_num,
        "univariate_categorical": univariate_cat,
        "univariate_datetime": univariate_dt,
        "correlations": correlations,
        "group_comparisons": group_comparisons,
        "categorical_associations": cat_associations,
        "timeseries_analysis": timeseries_results,
        "pca": pca_results,
        "target": target,
        "target_analysis": target_results,
    }


def compute_pca(
    df: pd.DataFrame, numeric_cols: List[str], n_components: int = 2
) -> Optional[Dict[str, Any]]:
    """Compute Principal Component Analysis on standardized numeric features.

    Returns explained variance ratios, loadings table, and 2D projections.
    """
    if len(numeric_cols) < 2:
        return None

    valid = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 5:
        return None

    # Standardize features (Z-scores)
    means = valid.mean()
    stds = valid.std().replace(0, 1.0)
    scaled = (valid - means) / stds

    try:
        from sklearn.decomposition import PCA

        n_comp = min(n_components, len(numeric_cols), len(valid))
        pca = PCA(n_components=n_comp)
        transformed = pca.fit_transform(scaled)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=numeric_cols,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        )

        return {
            "n_components": n_comp,
            "explained_variance_ratio": [float(r) for r in pca.explained_variance_ratio_],
            "total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "loadings": loadings,
            "projections": pd.DataFrame(
                transformed,
                columns=[f"PC{i+1}" for i in range(n_comp)],
                index=valid.index,
            ),
            "columns": numeric_cols,
        }
    except Exception:
        return None


def compute_target_analysis(
    df: pd.DataFrame, target_col: str, classified: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Perform supervised bivariate profiling against a designated target column."""
    numeric_cols = [c for c in classified.get("numeric", []) if c != target_col]
    cat_cols = [c for c in classified.get("categorical", []) if c != target_col]

    is_numeric_target = target_col in classified.get("numeric", [])

    if is_numeric_target:
        corrs = []
        for n_col in numeric_cols:
            valid = df[[target_col, n_col]].dropna()
            if len(valid) >= 3:
                r_val, p_val = stats.pearsonr(valid[target_col], valid[n_col])
                corrs.append({
                    "feature": n_col,
                    "pearson_r": float(r_val) if not np.isnan(r_val) else 0.0,
                    "pearson_p": float(p_val) if not np.isnan(p_val) else 1.0,
                })
        corrs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

        cat_diffs = []
        for c_col in cat_cols:
            comp = compute_group_comparisons(df, c_col, target_col)
            if comp.get("anova_f") is not None:
                cat_diffs.append(comp)

        return {
            "target_type": "numeric",
            "target_col": target_col,
            "correlations": corrs,
            "category_differences": cat_diffs,
        }
    else:
        num_diffs = []
        for n_col in numeric_cols:
            comp = compute_group_comparisons(df, target_col, n_col)
            if comp.get("anova_f") is not None:
                num_diffs.append(comp)

        cat_assocs = []
        for c_col in cat_cols:
            assoc = compute_categorical_association(df, target_col, c_col)
            if assoc.get("chi2") is not None:
                cat_assocs.append(assoc)

        return {
            "target_type": "categorical",
            "target_col": target_col,
            "numeric_differences": num_diffs,
            "categorical_associations": cat_assocs,
        }
