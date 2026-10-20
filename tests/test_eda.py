"""Unit and acceptance tests for the Exploratory Data Analysis module (modules/eda.py)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from modules.eda import (
    classify_columns,
    compute_correlations,
    compute_group_comparisons,
    compute_timeseries_analysis,
    compute_univariate_categorical,
    compute_univariate_datetime,
    compute_univariate_numeric,
    run_eda,
)
from modules.ingestion import load_file
from modules.cleaning import clean_data

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_classify_columns_excludes_constant_and_id_like() -> None:
    """Verify constant columns and ID-like columns are properly identified and excluded."""
    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{i:05d}" for i in range(100)],  # id_like
            "store_code": ["RETAIL-CORP-HQ"] * 100,  # constant
            "revenue": np.random.uniform(10.0, 500.0, 100),  # numeric
            "region": np.random.choice(["East", "West"], 100),  # categorical
            "order_date": pd.date_range("2023-01-01", periods=100, freq="D"),  # datetime
        }
    )
    classified, excluded = classify_columns(df)

    assert "store_code" in excluded
    assert "constant" in excluded["store_code"].lower()
    assert "order_id" in excluded
    assert "identifier" in excluded["order_id"].lower() or "id-like" in excluded["order_id"].lower()

    # Neither constant nor ID should be in numeric or categorical lists
    assert "store_code" not in classified["numeric"]
    assert "store_code" not in classified["categorical"]
    assert "order_id" not in classified["numeric"]
    assert "order_id" not in classified["categorical"]
    assert "revenue" in classified["numeric"]
    assert "region" in classified["categorical"]
    assert "order_date" in classified["datetime"]


def test_numeric_stats_match_pandas_and_scipy_directly() -> None:
    """Verify computed numeric stats match Pandas and SciPy calculations exactly."""
    np.random.seed(42)
    vals = np.random.normal(loc=100.0, scale=15.0, size=200)
    series = pd.Series(vals)

    res = compute_univariate_numeric(series)

    assert res["count"] == len(series)
    assert np.isclose(res["mean"], float(series.mean()), rtol=1e-5)
    assert np.isclose(res["std"], float(series.std(ddof=1)), rtol=1e-5)
    assert np.isclose(res["median"], float(series.median()), rtol=1e-5)
    assert np.isclose(res["min"], float(series.min()), rtol=1e-5)
    assert np.isclose(res["max"], float(series.max()), rtol=1e-5)
    assert np.isclose(res["q25"], float(series.quantile(0.25)), rtol=1e-5)
    assert np.isclose(res["q75"], float(series.quantile(0.75)), rtol=1e-5)
    assert np.isclose(res["iqr"], float(series.quantile(0.75) - series.quantile(0.25)), rtol=1e-5)
    assert np.isclose(res["skewness"], float(series.skew()), rtol=1e-5)
    assert np.isclose(res["kurtosis"], float(series.kurtosis()), rtol=1e-5)

    # Scipy normaltest
    _, p_val = stats.normaltest(vals)
    assert np.isclose(res["normality_pvalue"], p_val, rtol=1e-5)


def test_categorical_stats_match_pandas_and_scipy() -> None:
    """Verify categorical descriptive statistics match pandas value counts and scipy entropy."""
    cats = ["Apples", "Apples", "Bananas", "Cherries", "Apples", "Bananas"]
    series = pd.Series(cats)

    res = compute_univariate_categorical(series)

    assert res["count"] == 6
    assert res["cardinality"] == 3
    assert res["top_value"] == "Apples"
    assert res["top_freq"] == 3
    assert np.isclose(res["top_percentage"], 50.0)

    # Shannon entropy
    probs = [3 / 6, 2 / 6, 1 / 6]
    expected_entropy = stats.entropy(probs, base=2)
    assert np.isclose(res["entropy"], expected_entropy, rtol=1e-5)


def test_correlations_match_scipy_directly() -> None:
    """Verify Pearson and Spearman correlations match SciPy outputs directly."""
    np.random.seed(123)
    x = np.linspace(1, 100, 50)
    y = 2.5 * x + np.random.normal(0, 10, 50)
    z = -1.2 * x + np.random.normal(0, 5, 50)

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    res = compute_correlations(df, ["x", "y", "z"])

    expected_p_r, expected_p_p = stats.pearsonr(x, y)
    expected_s_rho, expected_s_p = stats.spearmanr(x, y)

    xy_pair = [p for p in res["pairs"] if set([p["col1"], p["col2"]]) == {"x", "y"}][0]
    assert np.isclose(xy_pair["pearson_r"], expected_p_r, rtol=1e-5)
    assert np.isclose(xy_pair["pearson_p"], expected_p_p, rtol=1e-5)
    assert np.isclose(xy_pair["spearman_rho"], expected_s_rho, rtol=1e-5)
    assert np.isclose(xy_pair["spearman_p"], expected_s_p, rtol=1e-5)


def test_clean_sales_revenue_profit_correlation_matches_pandas_directly() -> None:
    """Assert app's correlation between revenue and profit equals df[['revenue','profit']].corr().iloc[0,1] to 3 decimals."""
    clean_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    df, _ = load_file(clean_path)
    expected_r = float(df[["revenue", "profit"]].corr().iloc[0, 1])

    eda_res = run_eda(df)
    corr_meta = eda_res["correlations"]

    # 1. Pearson matrix value matches pandas
    mat_r = float(corr_meta["pearson"].loc["revenue", "profit"])
    assert round(mat_r, 3) == round(expected_r, 3), (
        f"Heatmap matrix r ({mat_r:.4f}) does not match pandas ({expected_r:.4f})"
    )

    # 2. Top pairs list value matches pandas
    rev_prof_pairs = [
        p for p in corr_meta["pairs"]
        if set([p["col1"], p["col2"]]) == {"revenue", "profit"}
    ]
    assert len(rev_prof_pairs) == 1
    pair_r = rev_prof_pairs[0]["pearson_r"]
    assert round(pair_r, 3) == round(expected_r, 3), (
        f"Pairs list r ({pair_r:.4f}) does not match pandas ({expected_r:.4f})"
    )


def test_group_comparison_anova_matches_scipy() -> None:
    """Verify group comparison one-way ANOVA F-statistic matches scipy.stats.f_oneway."""
    df = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "value": np.concatenate(
                [
                    np.random.normal(10, 2, 20),
                    np.random.normal(20, 2, 20),
                    np.random.normal(30, 2, 20),
                ]
            ),
        }
    )
    res = compute_group_comparisons(df, "group", "value")

    grp_a = df[df["group"] == "A"]["value"]
    grp_b = df[df["group"] == "B"]["value"]
    grp_c = df[df["group"] == "C"]["value"]
    f_stat, p_val = stats.f_oneway(grp_a, grp_b, grp_c)

    assert np.isclose(res["anova_f"], f_stat, rtol=1e-5)
    assert np.isclose(res["anova_p"], p_val, rtol=1e-5)
    assert res["significant"] is True


def test_timeseries_handles_nat_values_safely_without_crashing() -> None:
    """Verify time series analysis safely skips NaT values (e.g. 35 future dates in cleaned messy sales)."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D").tolist()
    # Insert 35 NaT values into the dates
    for i in range(35):
        dates[i * 2] = pd.NaT

    df = pd.DataFrame(
        {
            "order_date": dates,
            "revenue": np.random.uniform(50, 200, 100),
        }
    )
    # Must not raise an exception
    ts_res = compute_timeseries_analysis(df, "order_date", "revenue")

    assert ts_res["count"] == 65  # 100 - 35 NaT
    assert "trend_direction" in ts_res
    assert isinstance(ts_res["has_trend"], bool)
    assert isinstance(ts_res["has_seasonality"], bool)


def test_timeseries_detects_trend_and_seasonality_in_timeseries_benchmark() -> None:
    """Verify timeseries_sales.csv benchmark shows trend strength, R2, and multi-scale seasonality."""
    ts_path = SAMPLE_DATA_DIR / "timeseries_sales.csv"
    df_ts, _ = load_file(ts_path)

    eda_res = run_eda(df_ts)
    ts_results = eda_res["timeseries_analysis"]

    assert len(ts_results) > 0
    # Primary metric revenue should detect trend with strength label and multi-scale seasonality
    rev_ts = [r for r in ts_results if r["num_col"] == "revenue"][0]
    assert rev_ts["count"] > 1000
    assert rev_ts["has_trend"] is True
    assert rev_ts["trend_strength"] in {"weak", "moderate", "strong"}
    assert any(w in rev_ts["trend_direction"] for w in ["weak", "moderate", "strong"])
    assert "trend_r2" in rev_ts
    assert rev_ts["trend_r2"] > 0.05
    assert rev_ts["has_seasonality"] is True
    assert "weekly (7 days)" in rev_ts["seasonal_period"]
    assert "yearly (365 days)" in rev_ts["seasonal_period"]


def test_trend_strength_bands_classification() -> None:
    """Verify trend strength classification bands: weak (< 0.3), moderate (0.3-0.6), strong (> 0.6)."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # 1. Strong trend (R2 > 0.6)
    x = np.arange(100)
    strong_y = 5.0 * x + np.random.normal(0, 5, 100)
    df_strong = pd.DataFrame({"order_date": dates, "val": strong_y})
    res_strong = compute_timeseries_analysis(df_strong, "order_date", "val")
    assert res_strong["trend_strength"] == "strong"
    assert res_strong["trend_r2"] > 0.6
    assert "strong upward" in res_strong["trend_direction"]

    # 2. Moderate trend (0.3 <= R2 <= 0.6)
    np.random.seed(42)
    mod_y = 1.0 * x + np.random.normal(0, 45, 100)
    df_mod = pd.DataFrame({"order_date": dates, "val": mod_y})
    res_mod = compute_timeseries_analysis(df_mod, "order_date", "val")
    assert res_mod["trend_strength"] in {"moderate", "weak"}

    # 3. Flat / No trend (random noise)
    flat_y = np.random.normal(50, 10, 100)
    df_flat = pd.DataFrame({"order_date": dates, "val": flat_y})
    res_flat = compute_timeseries_analysis(df_flat, "order_date", "val")
    assert res_flat["trend_direction"] == "flat"



def test_eda_on_single_numeric_column_dataset_does_not_crash() -> None:
    """Verify dataset with only a single numeric column does not crash and handles correlations gracefully."""
    df_single = pd.DataFrame({"revenue": [10.0, 20.0, 30.0, 40.0, 50.0]})
    eda_res = run_eda(df_single)

    assert "revenue" in eda_res["univariate_numeric"]
    assert eda_res["correlations"]["pairs"] == []
    assert eda_res["group_comparisons"] == []
    assert eda_res["categorical_associations"] == []


def test_eda_on_tiny_dataset_does_not_crash() -> None:
    """Verify run_eda handles tiny.csv (50 rows) without crashing or division-by-zero errors."""
    tiny_path = SAMPLE_DATA_DIR / "tiny.csv"
    df_tiny, _ = load_file(tiny_path)

    eda_res = run_eda(df_tiny)
    assert len(eda_res["univariate_numeric"]) > 0
    assert len(eda_res["classified_columns"]["numeric"]) > 0


def test_eda_on_cleaned_messy_sales_excludes_store_code_and_order_id() -> None:
    """Verify run_eda on messy_sales.csv after cleaning completely excludes store_code and order_id."""
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    cleaned_df, _ = clean_data(df_messy)

    eda_res = run_eda(cleaned_df)

    assert "store_code" in eda_res["excluded_columns"]
    assert "order_id" in eda_res["excluded_columns"]

    # Excluded columns must not be in univariate stats
    assert "store_code" not in eda_res["univariate_numeric"]
    assert "store_code" not in eda_res["univariate_categorical"]
    assert "order_id" not in eda_res["univariate_numeric"]
    assert "order_id" not in eda_res["univariate_categorical"]

    # Excluded columns must not be in correlation matrix
    pearson_cols = list(eda_res["correlations"]["pearson"].columns)
    assert "store_code" not in pearson_cols
    assert "order_id" not in pearson_cols


def test_compute_pca_and_target_analysis() -> None:
    """Verify compute_pca produces valid components and compute_target_analysis profiles target."""
    from modules.eda import compute_pca, compute_target_analysis

    clean_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    df, _ = load_file(clean_path)

    # 1. PCA computation
    pca_res = compute_pca(df, ["revenue", "profit", "unit_price", "quantity"])
    assert pca_res is not None
    assert pca_res["n_components"] == 2
    assert len(pca_res["explained_variance_ratio"]) == 2
    assert 0.0 < pca_res["total_explained_variance"] <= 1.0
    assert "PC1" in pca_res["projections"].columns
    assert "PC2" in pca_res["projections"].columns
    assert len(pca_res["loadings"]) == 4

    # 2. Target analysis with numeric target
    classified = {"numeric": ["revenue", "profit", "unit_price"], "categorical": ["category", "region"]}
    t_num = compute_target_analysis(df, "revenue", classified)
    assert t_num["target_type"] == "numeric"
    assert len(t_num["correlations"]) > 0
    assert len(t_num["category_differences"]) > 0

    # 3. Target analysis with categorical target
    t_cat = compute_target_analysis(df, "category", classified)
    assert t_cat["target_type"] == "categorical"
    assert len(t_cat["numeric_differences"]) > 0

