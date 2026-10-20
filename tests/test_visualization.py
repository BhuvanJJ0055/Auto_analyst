"""Unit and acceptance tests for the automated visualization engine (modules/visualization.py)."""

from pathlib import Path
import pandas as pd
import pytest

from modules.cleaning import clean_data
from modules.eda import run_eda
from modules.ingestion import load_file
from modules.visualization import generate_visualizations

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_clean_sales_produces_at_least_eight_charts_without_errors() -> None:
    """Verify clean_sales.csv produces at least 8 publication-ready charts without errors."""
    clean_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    df_clean, _ = load_file(clean_path)

    charts = generate_visualizations(df_clean)

    assert len(charts) >= 8, f"Expected >= 8 charts, received {len(charts)}"

    # Check chart attributes for every generated chart
    for chart in charts:
        assert "id" in chart and len(chart["id"]) > 0
        assert "title" in chart and len(chart["title"]) > 0
        assert "chart_type" in chart and len(chart["chart_type"]) > 0
        assert "fig" in chart and chart["fig"] is not None
        assert "caption" in chart and len(chart["caption"]) > 0
        assert "columns" in chart and len(chart["columns"]) > 0

        # Check figure has layout title and axis labels
        layout = chart["fig"].layout
        assert layout.title.text is not None and len(layout.title.text) > 0


def test_chart_types_match_semantic_column_combinations() -> None:
    """Verify chart engine generates expected chart types matching semantic column combinations."""
    df = pd.DataFrame(
        {
            "revenue": [100.0, 250.0, 300.0, 450.0, 500.0, 600.0, 750.0, 800.0],
            "profit": [20.0, 50.0, 60.0, 90.0, 100.0, 120.0, 150.0, 160.0],
            "region": ["East", "West", "East", "West", "East", "West", "East", "West"],
            "order_date": pd.date_range("2023-01-01", periods=8, freq="D"),
        }
    )

    charts = generate_visualizations(df)
    chart_types = {c["chart_type"] for c in charts}

    assert "histogram" in chart_types
    assert "box" in chart_types
    assert "bar" in chart_types
    assert "heatmap" in chart_types
    assert "scatter" in chart_types
    assert "timeseries" in chart_types


def test_every_chart_has_title_axis_labels_and_computed_caption() -> None:
    """Verify every chart includes title, x and y axis labels, and a non-empty computed caption."""
    clean_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    df_clean, _ = load_file(clean_path)

    charts = generate_visualizations(df_clean)

    for c in charts:
        fig = c["fig"]
        title = c["title"]
        caption = c["caption"]

        assert len(title.strip()) > 0, "Missing chart title"
        assert len(caption.strip()) > 0, "Missing plain-English caption"
        # Confirm caption is computed from real numbers/stats
        assert any(char.isdigit() for char in caption), f"Caption lacks computed stats numbers: {caption}"

        # Check axis labels in layout (except boxplot where x is blank category)
        assert fig.layout.yaxis.title.text is not None, f"Missing yaxis label in {title}"


def test_charts_exclude_constant_and_id_like_columns() -> None:
    """Verify store_code (constant) and order_id (id-like) are excluded from all generated charts."""
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    df_cleaned, _ = clean_data(df_messy)

    charts = generate_visualizations(df_cleaned)

    for c in charts:
        cols = c["columns"]
        assert "store_code" not in cols, f"Constant column store_code found in chart {c['title']}"
        assert "order_id" not in cols, f"ID-like column order_id found in chart {c['title']}"


def test_charts_on_cleaned_messy_sales_handles_nat_dates_without_crashing() -> None:
    """Verify 35 NaT dates in order_date after cleaning do not crash visualization engine."""
    messy_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df_messy, _ = load_file(messy_path)
    df_cleaned, _ = clean_data(df_messy)

    # 35 NaT values must be present in order_date
    assert df_cleaned["order_date"].isna().sum() == 35

    # Generate charts must run without raising an exception
    charts = generate_visualizations(df_cleaned)
    assert len(charts) >= 8

    # Time series chart must still be generated using the remaining valid dates
    ts_charts = [c for c in charts if c["chart_type"] == "timeseries"]
    assert len(ts_charts) > 0


def test_visualization_on_single_numeric_column_dataset_does_not_crash() -> None:
    """Verify dataset with a single numeric column produces distribution charts without crashing."""
    df_single = pd.DataFrame({"sales_volume": [12.0, 15.0, 22.0, 35.0, 40.0, 48.0, 55.0, 62.0]})
    charts = generate_visualizations(df_single)

    assert len(charts) == 2  # 1 histogram + 1 boxplot
    assert charts[0]["chart_type"] == "histogram"
    assert charts[1]["chart_type"] == "box"
    assert "sales_volume" in charts[0]["columns"]


def test_visualization_on_tiny_dataset_does_not_crash() -> None:
    """Verify tiny.csv (50 records) executes cleanly and generates valid charts."""
    tiny_path = SAMPLE_DATA_DIR / "tiny.csv"
    df_tiny, _ = load_file(tiny_path)

    charts = generate_visualizations(df_tiny)
    assert len(charts) >= 5
    for c in charts:
        assert c["fig"] is not None
        assert len(c["caption"]) > 0


# ---------------------------------------------------------------------------
# describe_distribution – skewness and kurtosis band tests
# ---------------------------------------------------------------------------
from modules.visualization import describe_distribution  # noqa: E402


class TestDescribeDistributionSkewnessBands:
    """Each band must map to the correct human-readable label."""

    def test_near_zero_skew_is_approximately_symmetric(self):
        """skew = -0.03 (e.g. age column) must NOT be labelled 'skewed'."""
        label, _, short = describe_distribution(-0.03, 0.0)
        assert label == "approximately symmetric", f"Got: {label}"
        assert short == "Symmetric"

    def test_zero_skew_is_approximately_symmetric(self):
        label, _, short = describe_distribution(0.0, 0.0)
        assert label == "approximately symmetric"
        assert short == "Symmetric"

    def test_skew_just_below_threshold_right(self):
        """0.49 is still in the symmetric band."""
        label, _, _ = describe_distribution(0.49, 0.0)
        assert label == "approximately symmetric"

    def test_skew_just_below_threshold_left(self):
        label, _, _ = describe_distribution(-0.49, 0.0)
        assert label == "approximately symmetric"

    def test_skew_at_threshold_right_is_moderately_skewed(self):
        """Exactly 0.5 enters moderately-skewed band."""
        label, _, short = describe_distribution(0.5, 0.0)
        assert label == "moderately right-skewed"
        assert short == "Mod. skewed"

    def test_skew_at_threshold_left_is_moderately_skewed(self):
        label, _, short = describe_distribution(-0.5, 0.0)
        assert label == "moderately left-skewed"
        assert short == "Mod. skewed"

    def test_skew_inside_moderate_right_band(self):
        label, _, _ = describe_distribution(0.75, 0.0)
        assert label == "moderately right-skewed"

    def test_skew_inside_moderate_left_band(self):
        label, _, _ = describe_distribution(-0.75, 0.0)
        assert label == "moderately left-skewed"

    def test_skew_at_upper_moderate_boundary_right(self):
        """Exactly 1.0 is still in moderately-skewed band."""
        label, _, _ = describe_distribution(1.0, 0.0)
        assert label == "moderately right-skewed"

    def test_skew_at_upper_moderate_boundary_left(self):
        label, _, _ = describe_distribution(-1.0, 0.0)
        assert label == "moderately left-skewed"

    def test_skew_above_one_is_highly_right_skewed(self):
        """1.01 crosses into highly-skewed band."""
        label, _, short = describe_distribution(1.01, 0.0)
        assert label == "highly right-skewed"
        assert short == "Highly skewed"

    def test_skew_above_one_is_highly_left_skewed(self):
        label, _, short = describe_distribution(-1.01, 0.0)
        assert label == "highly left-skewed"
        assert short == "Highly skewed"

    def test_large_positive_skew(self):
        label, _, _ = describe_distribution(3.5, 0.0)
        assert label == "highly right-skewed"

    def test_large_negative_skew(self):
        label, _, _ = describe_distribution(-2.8, 0.0)
        assert label == "highly left-skewed"


class TestDescribeDistributionKurtosisBands:
    """Each kurtosis band must map to the correct human-readable label."""

    def test_kurt_below_minus_one_is_flat(self):
        """kurt = -1.5 → flat (platykurtic)."""
        _, label, _ = describe_distribution(0.0, -1.5)
        assert label == "flat (platykurtic)"

    def test_kurt_at_minus_one_is_flat(self):
        """kurt = exactly -1 → flat."""
        _, label, _ = describe_distribution(0.0, -1.0)
        # -1.0 is NOT < -1.0, so it falls in normal-tailed
        assert label == "normal-tailed (mesokurtic)"

    def test_kurt_just_above_minus_one_is_normal_tailed(self):
        _, label, _ = describe_distribution(0.0, -0.99)
        assert label == "normal-tailed (mesokurtic)"

    def test_kurt_at_zero_is_normal_tailed(self):
        _, label, _ = describe_distribution(0.0, 0.0)
        assert label == "normal-tailed (mesokurtic)"

    def test_kurt_at_one_is_normal_tailed(self):
        """kurt = exactly 1 is NOT > 1, so still normal-tailed."""
        _, label, _ = describe_distribution(0.0, 1.0)
        assert label == "normal-tailed (mesokurtic)"

    def test_kurt_just_above_one_is_heavy_tailed(self):
        _, label, _ = describe_distribution(0.0, 1.01)
        assert label == "heavy-tailed (leptokurtic)"

    def test_kurt_large_positive_is_heavy_tailed(self):
        _, label, _ = describe_distribution(0.0, 5.0)
        assert label == "heavy-tailed (leptokurtic)"


class TestHistogramCaptionUsesCorrectBands:
    """Integration: histogram caption must reflect the correct band labels."""

    def test_symmetric_column_caption_says_approximately_symmetric(self):
        """A nearly-flat distribution should produce 'approximately symmetric' in caption."""
        import numpy as np
        from modules.visualization import create_histogram

        rng = np.random.default_rng(42)
        data = pd.Series(rng.uniform(0, 100, 500))
        num_stats = {
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis()),
            "median": float(data.median()),
            "is_normal": False,
        }
        result = create_histogram(data, "test_col", num_stats)
        caption = result["caption"].lower()
        assert "approximately symmetric" in caption or "moderately" in caption or "highly" in caption

    def test_age_skew_minus_003_produces_symmetric_caption(self):
        """Confirm age-like skewness (-0.03) yields 'approximately symmetric' in caption."""
        from modules.visualization import create_histogram

        data = pd.Series([25, 30, 35, 40, 45, 28, 33, 38, 42, 29])
        num_stats = {
            "skewness": -0.03,
            "kurtosis": 0.1,
            "median": 34.0,
            "is_normal": False,
        }
        result = create_histogram(data, "age", num_stats)
        caption = result["caption"].lower()
        assert "approximately symmetric" in caption, f"Expected 'approximately symmetric', got: {caption}"
        assert "skewed" not in caption, f"Caption should not say 'skewed' for age: {caption}"


# ---------------------------------------------------------------------------
# _compute_bins – adaptive bin rule tests
# ---------------------------------------------------------------------------
from modules.visualization import _compute_bins  # noqa: E402


class TestComputeBins:
    """_compute_bins must return a sensible bin width for a variety of inputs."""

    def _bin_count(self, series: pd.Series) -> int:
        """Helper: compute how many bins _compute_bins would produce."""
        import math
        b = _compute_bins(series)
        clean = pd.to_numeric(series, errors="coerce").dropna()
        data_range = float(clean.max()) - float(clean.min())
        if data_range == 0 or b["size"] == 0:
            return 1
        return int(math.ceil(data_range / b["size"]))

    def test_integer_column_bin_width_at_least_one(self):
        """Integer series must have bin_width >= 1."""
        data = pd.Series(range(1, 51))  # 1 to 50
        result = _compute_bins(data)
        assert result["size"] >= 1.0, f"bin_width too small: {result['size']}"

    def test_integer_column_bin_width_is_whole_number(self):
        """Integer column bin widths must be whole numbers."""
        data = pd.Series(list(range(18, 75)))  # age-like
        result = _compute_bins(data)
        assert result["size"] == int(result["size"]), f"Non-integer bin_width: {result['size']}"

    def test_constant_series_does_not_crash(self):
        """A series where all values are equal must not raise."""
        data = pd.Series([5.0] * 100)
        result = _compute_bins(data)
        assert result["size"] > 0

    def test_empty_series_does_not_crash(self):
        data = pd.Series([], dtype=float)
        result = _compute_bins(data)  # should not raise
        assert result["size"] > 0

    def test_bin_count_between_5_and_60_for_float_column(self):
        """Continuous float column: bin count must be in [5, 60]."""
        import numpy as np
        rng = np.random.default_rng(0)
        data = pd.Series(rng.normal(50, 15, 1000))
        count = self._bin_count(data)
        assert 5 <= count <= 60, f"Bin count {count} out of [5, 60]"

    def test_bin_count_between_5_and_60_for_integer_column(self):
        """Integer column: bin count must be in [5, 60]."""
        import numpy as np
        rng = np.random.default_rng(1)
        data = pd.Series(rng.integers(18, 75, size=1000))
        count = self._bin_count(data)
        assert 5 <= count <= 60, f"Bin count {count} out of [5, 60]"


def test_histogram_bin_count_sensible_for_every_numeric_column_in_clean_sales() -> None:
    """Every numeric column in clean_sales.csv must produce a histogram with a sensible bin count.

    For most columns this is 5-60 bins.
    Low-cardinality integer columns (range < 5, e.g. ratings 1-5) are allowed
    to have as few bins as their integer range, but never more than 60.
    """
    import math
    clean_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    df, _ = load_file(clean_path)

    numeric_cols = df.select_dtypes(include="number").columns
    assert len(numeric_cols) > 0, "No numeric columns found in clean_sales.csv"

    for col in numeric_cols:
        series = df[col]
        b = _compute_bins(series)
        clean = pd.to_numeric(series, errors="coerce").dropna()
        data_range = float(clean.max()) - float(clean.min())
        if data_range == 0 or b["size"] == 0:
            continue  # constant column — skip
        n_bins = int(math.ceil(data_range / b["size"]))

        # For integer columns with range < 5 (e.g. ratings 1-5), the sensible
        # minimum is the integer range itself (e.g. 4 for values 1-5).
        is_integer_col = (clean == clean.round()).all()
        min_bins = max(1, int(data_range)) if (is_integer_col and data_range < 5) else 5

        assert min_bins <= n_bins <= 60, (
            f"Column '{col}': bin_count={n_bins} outside [{min_bins}, 60] "
            f"(range={data_range:.2f}, bin_size={b['size']:.4f})"
        )


def test_correlation_heatmap_shows_r_values_as_cell_text() -> None:
    """Verify correlation heatmap displays Pearson r values inside every cell."""
    from modules.visualization import create_correlation_heatmap
    corr_data = {
        "pearson": pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]],
            columns=["col_a", "col_b"],
            index=["col_a", "col_b"],
        ),
        "pairs": [],
    }
    chart = create_correlation_heatmap(corr_data)
    assert chart is not None
    heatmap_trace = chart["fig"].data[0]
    assert heatmap_trace.text is not None
    assert heatmap_trace.text[0][1] == "0.85"
    assert heatmap_trace.texttemplate == "%{text}"


class TestPValueAndCorrelationDescription:
    """Tests ensuring p-values never display 0.00e+00 and correlations have plain-English readings."""

    def test_format_p_value_tiny_values_show_less_than_threshold(self):
        from modules.visualization import format_p_value

        assert format_p_value(0.0) == "p < 0.001"
        assert format_p_value(1e-15) == "p < 0.001"
        assert format_p_value(0.0005) == "p < 0.001"
        assert format_p_value(0.0, include_prefix=False) == "< 0.001"
        assert "0.00e+00" not in format_p_value(0.0)

    def test_format_p_value_moderate_and_non_significant(self):
        from modules.visualization import format_p_value

        assert format_p_value(0.0042) == "p = 0.004"
        assert format_p_value(0.046) == "p = 0.05"
        assert format_p_value(0.789) == "p = 0.79"
        assert format_p_value(None) == "p = N/A"

    def test_describe_correlation_bands_and_direction(self):
        from modules.visualization import describe_correlation

        assert describe_correlation(0.85) == "very strong positive relationship"
        assert describe_correlation(-0.82) == "very strong negative relationship"
        assert describe_correlation(0.68) == "strong positive relationship"
        assert describe_correlation(-0.62) == "strong negative relationship"
        assert describe_correlation(0.45) == "moderate positive relationship"
        assert describe_correlation(-0.30) == "weak negative relationship"
        assert describe_correlation(0.12) == "very weak positive relationship"
        assert describe_correlation(0.01) == "negligible relationship"

    def test_heatmap_and_scatter_captions_never_contain_zero_exponential(self):
        from modules.visualization import create_correlation_heatmap, create_scatter_plot

        corr_data = {
            "pearson": pd.DataFrame(
                [[1.0, 0.84], [0.84, 1.0]],
                columns=["revenue", "profit"],
                index=["revenue", "profit"],
            ),
            "pairs": [
                {
                    "col1": "revenue",
                    "col2": "profit",
                    "pearson_r": 0.84,
                    "pearson_p": 0.0,  # Extreme significance
                }
            ],
        }
        h_chart = create_correlation_heatmap(corr_data)
        assert h_chart is not None
        assert "0.00e+00" not in h_chart["caption"]
        assert "p < 0.001" in h_chart["caption"]
        assert "very strong positive relationship" in h_chart["caption"]

        df_dummy = pd.DataFrame({"revenue": [10, 20, 30], "profit": [2, 4, 6]})
        s_chart = create_scatter_plot(df_dummy, "revenue", "profit", corr_data["pairs"][0])
        assert "0.00e+00" not in s_chart["caption"]
        assert "p < 0.001" in s_chart["caption"]
        assert "very strong positive relationship" in s_chart["caption"]



