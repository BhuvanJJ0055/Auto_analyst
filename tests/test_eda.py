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
    x = np.arange(100)

    # 1. Strong trend (R2 > 0.6)
    np.random.seed(42)
    strong_y = 5.0 * x + np.random.normal(0, 5, 100)
    df_strong = pd.DataFrame({"order_date": dates, "val": strong_y})
    res_strong = compute_timeseries_analysis(df_strong, "order_date", "val")
    assert res_strong["trend_strength"] == "strong"
    assert res_strong["trend_r2"] > 0.6
    assert "strong upward" in res_strong["trend_direction"]
    assert "strong upward trend" in res_strong["trend_summary_label"]
    assert "Strong upward trend" in res_strong["trend_card_label"]

    # 2. Moderate trend (0.3 <= R2 <= 0.6)
    np.random.seed(42)
    # Calibrate noise so R2 is within [0.35, 0.55]
    mod_y = 1.0 * x + np.random.normal(0, 38, 100)
    df_mod = pd.DataFrame({"order_date": dates, "val": mod_y})
    res_mod = compute_timeseries_analysis(df_mod, "order_date", "val")
    assert 0.30 <= res_mod["trend_r2"] <= 0.60
    assert res_mod["trend_strength"] == "moderate"
    assert "moderate upward" in res_mod["trend_direction"]
    assert "moderate upward trend" in res_mod["trend_summary_label"]
    assert "Moderate upward trend" in res_mod["trend_card_label"]

    # 3. Weak trend (R2 < 0.3, with statistically significant slope)
    np.random.seed(42)
    # Calibrate noise so R2 is within [0.05, 0.25] with p < 0.05
    weak_y = 0.35 * x + np.random.normal(0, 30, 100)
    df_weak = pd.DataFrame({"order_date": dates, "val": weak_y})
    res_weak = compute_timeseries_analysis(df_weak, "order_date", "val")
    assert res_weak["trend_r2"] < 0.30
    assert res_weak["trend_strength"] == "weak"
    assert "weak upward" in res_weak["trend_direction"]
    assert "weak upward trend" in res_weak["trend_summary_label"]
    assert "Weak upward trend" in res_weak["trend_card_label"]

    # 4. Flat / No trend (pure random noise)
    np.random.seed(42)
    flat_y = np.random.normal(50, 10, 100)
    df_flat = pd.DataFrame({"order_date": dates, "val": flat_y})
    res_flat = compute_timeseries_analysis(df_flat, "order_date", "val")
    assert res_flat["trend_direction"] == "flat"
    assert res_flat["trend_strength"] == "flat"
    assert "no clear trend" in res_flat["trend_summary_label"]


def test_timeseries_aggregation_per_column_type() -> None:
    """Verify daily aggregation selects mean for rate/age/rating/discount and sum for volumetric metrics."""
    from modules.eda import get_timeseries_aggregation

    # Mean for rate/average metrics
    assert get_timeseries_aggregation("age") == ("mean", "Average")
    assert get_timeseries_aggregation("customer_age") == ("mean", "Average")
    assert get_timeseries_aggregation("rating") == ("mean", "Average")
    assert get_timeseries_aggregation("discount") == ("mean", "Average")
    assert get_timeseries_aggregation("discount_percent") == ("mean", "Average")
    assert get_timeseries_aggregation("unit_price") == ("mean", "Average")

    # Sum for throughput / volume metrics
    assert get_timeseries_aggregation("revenue") == ("sum", "Daily Total")
    assert get_timeseries_aggregation("profit") == ("sum", "Daily Total")
    assert get_timeseries_aggregation("quantity") == ("sum", "Daily Total")
    assert get_timeseries_aggregation("sales") == ("sum", "Daily Total")
    assert get_timeseries_aggregation("orders") == ("sum", "Daily Total")


def test_timeseries_detects_yearly_seasonality_on_timeseries_sales() -> None:
    """Verify yearly and weekly seasonality are detected on timeseries_sales.csv, without monthly false positive."""
    ts_path = SAMPLE_DATA_DIR / "timeseries_sales.csv"
    df_ts, _ = load_file(ts_path)

    res = compute_timeseries_analysis(df_ts, "order_date", "revenue")
    assert res["has_seasonality"] is True
    assert res["seasonal_period"] is not None
    assert "yearly (365 days)" in res["seasonal_period"]
    assert "weekly (7 days)" in res["seasonal_period"]
    assert "monthly (~30 days)" not in res["seasonal_period"]
    assert res["seasonal_period"] == "weekly (7 days) & yearly (365 days)"


def test_controlled_synthetic_timeseries_detects_weekly_and_yearly_only() -> None:
    """Controlled synthetic series with linear trend, 7-day, and 365-day cycles only.

    Asserts that exactly weekly and yearly seasonality are detected, and monthly is NOT.
    """
    rng = np.random.RandomState(42)
    dates = pd.date_range("2022-01-01", periods=1000, freq="D")
    t = np.arange(len(dates))
    known_slope = 12.0
    weekly_cycle = 40.0 * np.sin(2 * np.pi * t / 7)
    yearly_cycle = 80.0 * np.cos(2 * np.pi * t / 365)
    noise = rng.normal(0, 8.0, len(dates))
    y = 500.0 + known_slope * t + weekly_cycle + yearly_cycle + noise

    df_synth = pd.DataFrame({"order_date": dates, "revenue": y})
    res = compute_timeseries_analysis(df_synth, "order_date", "revenue")

    assert res["has_trend"] is True
    assert "upward" in res["trend_direction"]
    assert res["has_seasonality"] is True
    assert "weekly (7 days)" in res["seasonal_period"]
    assert "yearly (365 days)" in res["seasonal_period"]
    assert "monthly (~30 days)" not in res["seasonal_period"]
    assert res["seasonal_period"] == "weekly (7 days) & yearly (365 days)"


def test_negative_timeseries_no_trend_no_seasonality_on_noise_and_non_seasonal_datasets() -> None:
    """Verify negative cases: random noise reports no trend and no seasonality, and clean/messy sales do not report yearly seasonality."""
    # 1. Pure random Gaussian noise across 400 days
    np.random.seed(42)
    dates_noise = pd.date_range("2023-01-01", periods=400, freq="D")
    df_noise = pd.DataFrame({
        "order_date": dates_noise,
        "metric": np.random.normal(loc=100.0, scale=15.0, size=400),
    })
    res_noise = compute_timeseries_analysis(df_noise, "order_date", "metric")
    assert res_noise["has_trend"] is False
    assert res_noise["trend_direction"] == "flat"
    assert res_noise["has_seasonality"] is False
    assert res_noise["seasonal_period"] is None

    # 2. clean_sales.csv does not have yearly seasonality
    df_clean, _ = load_file(SAMPLE_DATA_DIR / "clean_sales.csv")
    res_clean = compute_timeseries_analysis(df_clean, "order_date", "revenue")
    if res_clean["has_seasonality"] and res_clean["seasonal_period"]:
        assert "yearly (365 days)" not in res_clean["seasonal_period"]
    else:
        assert res_clean["seasonal_period"] is None

    # 3. messy_sales.csv (cleaned) does not have yearly seasonality
    df_messy, _ = load_file(SAMPLE_DATA_DIR / "messy_sales.csv")
    cleaned_messy, _ = clean_data(df_messy)
    res_messy = compute_timeseries_analysis(cleaned_messy, "order_date", "revenue")
    if res_messy["has_seasonality"] and res_messy["seasonal_period"]:
        assert "yearly (365 days)" not in res_messy["seasonal_period"]
    else:
        assert res_messy["seasonal_period"] is None



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

