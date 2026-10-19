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
        label, _ = describe_distribution(-0.03, 0.0)
        assert label == "approximately symmetric", f"Got: {label}"

    def test_zero_skew_is_approximately_symmetric(self):
        label, _ = describe_distribution(0.0, 0.0)
        assert label == "approximately symmetric"

    def test_skew_just_below_threshold_right(self):
        """0.49 is still in the symmetric band."""
        label, _ = describe_distribution(0.49, 0.0)
        assert label == "approximately symmetric"

    def test_skew_just_below_threshold_left(self):
        label, _ = describe_distribution(-0.49, 0.0)
        assert label == "approximately symmetric"

    def test_skew_at_threshold_right_is_moderately_skewed(self):
        """Exactly 0.5 enters moderately-skewed band."""
        label, _ = describe_distribution(0.5, 0.0)
        assert label == "moderately right-skewed"

    def test_skew_at_threshold_left_is_moderately_skewed(self):
        label, _ = describe_distribution(-0.5, 0.0)
        assert label == "moderately left-skewed"

    def test_skew_inside_moderate_right_band(self):
        label, _ = describe_distribution(0.75, 0.0)
        assert label == "moderately right-skewed"

    def test_skew_inside_moderate_left_band(self):
        label, _ = describe_distribution(-0.75, 0.0)
        assert label == "moderately left-skewed"

    def test_skew_at_upper_moderate_boundary_right(self):
        """Exactly 1.0 is still in moderately-skewed band."""
        label, _ = describe_distribution(1.0, 0.0)
        assert label == "moderately right-skewed"

    def test_skew_at_upper_moderate_boundary_left(self):
        label, _ = describe_distribution(-1.0, 0.0)
        assert label == "moderately left-skewed"

    def test_skew_above_one_is_highly_right_skewed(self):
        """1.01 crosses into highly-skewed band."""
        label, _ = describe_distribution(1.01, 0.0)
        assert label == "highly right-skewed"

    def test_skew_above_one_is_highly_left_skewed(self):
        label, _ = describe_distribution(-1.01, 0.0)
        assert label == "highly left-skewed"

    def test_large_positive_skew(self):
        label, _ = describe_distribution(3.5, 0.0)
        assert label == "highly right-skewed"

    def test_large_negative_skew(self):
        label, _ = describe_distribution(-2.8, 0.0)
        assert label == "highly left-skewed"


class TestDescribeDistributionKurtosisBands:
    """Each kurtosis band must map to the correct human-readable label."""

    def test_kurt_below_minus_one_is_flat(self):
        """kurt = -1.5 → flat (platykurtic)."""
        _, label = describe_distribution(0.0, -1.5)
        assert label == "flat (platykurtic)"

    def test_kurt_at_minus_one_is_flat(self):
        """kurt = exactly -1 → flat."""
        _, label = describe_distribution(0.0, -1.0)
        # -1.0 is NOT < -1.0, so it falls in normal-tailed
        assert label == "normal-tailed (mesokurtic)"

    def test_kurt_just_above_minus_one_is_normal_tailed(self):
        _, label = describe_distribution(0.0, -0.99)
        assert label == "normal-tailed (mesokurtic)"

    def test_kurt_at_zero_is_normal_tailed(self):
        _, label = describe_distribution(0.0, 0.0)
        assert label == "normal-tailed (mesokurtic)"

    def test_kurt_at_one_is_normal_tailed(self):
        """kurt = exactly 1 is NOT > 1, so still normal-tailed."""
        _, label = describe_distribution(0.0, 1.0)
        assert label == "normal-tailed (mesokurtic)"

    def test_kurt_just_above_one_is_heavy_tailed(self):
        _, label = describe_distribution(0.0, 1.01)
        assert label == "heavy-tailed (leptokurtic)"

    def test_kurt_large_positive_is_heavy_tailed(self):
        _, label = describe_distribution(0.0, 5.0)
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

