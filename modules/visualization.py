"""Automated Visualization Engine for AutoAnalyst.

Selects, configures, and renders publication-ready Plotly figures tailored strictly
to detected semantic column types, distributions, correlations, and trends.
Excludes constant and ID-like columns and computes descriptive one-line captions from real stats.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.eda import run_eda
from modules.utils import get_color_palette

# Plotly dark theme styling tokens
PLOTLY_DARK_LAYOUT = {
    "paper_bgcolor": "#161B22",
    "plot_bgcolor": "#0D1117",
    "font": {"color": "#C9D1D9", "family": "Inter, Segoe UI, sans-serif"},
    "xaxis": {
        "gridcolor": "#30363D",
        "zerolinecolor": "#30363D",
        "showline": True,
        "linecolor": "#30363D",
    },
    "yaxis": {
        "gridcolor": "#30363D",
        "zerolinecolor": "#30363D",
        "showline": True,
        "linecolor": "#30363D",
    },
    "margin": {"l": 50, "r": 30, "t": 60, "b": 50},
}


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent high-contrast dark theme styling to a Plotly figure."""
    fig.update_layout(**PLOTLY_DARK_LAYOUT)
    return fig


def describe_distribution(skew: float, kurt: float) -> tuple:
    """Return (long_skew_label, kurt_label, short_skew_label) from raw stats.

    Skewness bands
    --------------
    |skew| < 0.5          → long: 'approximately symmetric',  short: 'Symmetric'
    0.5 ≤ |skew| ≤ 1.0   → long: 'moderately right/left-skewed', short: 'Mod. skewed'
    |skew| > 1.0          → long: 'highly right/left-skewed',  short: 'Highly skewed'

    Kurtosis bands (excess kurtosis)
    ---------------------------------
    kurt < -1  → 'flat (platykurtic)'
    kurt > 1   → 'heavy-tailed (leptokurtic)'
    otherwise  → 'normal-tailed (mesokurtic)'
    """
    abs_skew = abs(skew)

    if abs_skew < 0.5:
        skew_label = "approximately symmetric"
        short_label = "Symmetric"
    elif abs_skew <= 1.0:
        direction = "right" if skew > 0 else "left"
        skew_label = f"moderately {direction}-skewed"
        short_label = "Mod. skewed"
    else:
        direction = "right" if skew > 0 else "left"
        skew_label = f"highly {direction}-skewed"
        short_label = "Highly skewed"

    if kurt < -1.0:
        kurt_label = "flat (platykurtic)"
    elif kurt > 1.0:
        kurt_label = "heavy-tailed (leptokurtic)"
    else:
        kurt_label = "normal-tailed (mesokurtic)"

    return skew_label, kurt_label, short_label


def _compute_bins(series: pd.Series) -> Dict[str, float]:
    """Compute adaptive histogram bin edges using Freedman-Diaconis, falling back to Sturges.

    - Freedman-Diaconis: h = 2 * IQR * n^(-1/3)
    - Sturges fallback when IQR = 0 or the FD bin count would be > 60
    - Integer columns: bin width is rounded up to the nearest integer ≥ 1
    - Result bin count is clamped to [5, 60]
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    if n < 2:
        return {"start": float(clean.min()) if n else 0.0, "end": float(clean.max()) if n else 1.0, "size": 1.0}

    data_min = float(clean.min())
    data_max = float(clean.max())
    data_range = data_max - data_min
    if data_range == 0:
        return {"start": data_min - 0.5, "end": data_max + 0.5, "size": 1.0}

    # Detect integer-valued column
    is_integer_col = (clean == clean.round()).all()

    # Freedman-Diaconis
    q75, q25 = float(clean.quantile(0.75)), float(clean.quantile(0.25))
    iqr = q75 - q25
    if iqr > 0:
        fd_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
        fd_bins = int(np.ceil(data_range / fd_width))
    else:
        fd_bins = 0  # trigger Sturges fallback

    # Sturges fallback
    sturges_bins = int(np.ceil(np.log2(n))) + 1

    # Choose: use FD if it gives a sensible count, else Sturges
    if 5 <= fd_bins <= 60:
        n_bins = fd_bins
    else:
        n_bins = max(5, min(60, sturges_bins))

    bin_width = data_range / n_bins

    # Integer columns: bin width must be ≥ 1 and a whole number
    if is_integer_col:
        bin_width = max(1.0, np.ceil(bin_width))
        n_bins = int(np.ceil(data_range / bin_width))
        # For low-cardinality integers (e.g. ratings 1-5, range=4),
        # the minimum sensible bin count is the range itself, not 5.
        min_bins = max(1, int(data_range))
        n_bins = max(min_bins, min(60, n_bins))

    # Align start to a clean boundary
    start = data_min - (bin_width * 0.05)  # tiny left pad so first bar is full
    end = data_max + (bin_width * 0.05)

    return {"start": start, "end": end, "size": bin_width}


def create_histogram(
    series: pd.Series, col_name: str, num_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a distribution histogram with descriptive axes and calculated caption."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean_title = col_name.replace("_", " ").title()

    xbins = _compute_bins(series)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=clean,
            xbins=dict(
                start=xbins["start"],
                end=xbins["end"],
                size=xbins["size"],
            ),
            marker_color="#58A6FF",
            opacity=0.85,
            name="Count",
        )
    )

    # Median line
    med = num_stats.get("median")
    if med is not None:
        fig.add_vline(
            x=med,
            line_width=2,
            line_dash="dash",
            line_color="#3FB950",
            annotation_text=f"Median: {med:,.1f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"{clean_title} Distribution",
        xaxis_title=clean_title,
        yaxis_title="Count",
        showlegend=False,
    )
    _apply_theme(fig)

    # Generate one-line plain-English caption from stats
    skew = num_stats.get("skewness", 0.0) or 0.0
    kurt = num_stats.get("kurtosis", 0.0) or 0.0
    is_norm = num_stats.get("is_normal", False)

    if is_norm:
        shape_desc = "follows a normal distribution"
        kurt_desc = ""
    else:
        skew_label, kurt_label, _ = describe_distribution(skew, kurt)
        shape_desc = f"is {skew_label}"
        kurt_desc = f"; tail shape is {kurt_label}"

    caption = (
        f"{clean_title} {shape_desc}{kurt_desc} "
        f"(skewness = {skew:.2f}, kurtosis = {kurt:.2f}) "
        f"with a median of {med:,.2f}."
    )

    return {
        "id": f"hist_{col_name}",
        "title": f"{clean_title} Distribution",
        "chart_type": "histogram",
        "fig": fig,
        "caption": caption,
        "columns": [col_name],
    }


def create_boxplot(
    series: pd.Series, col_name: str, num_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a box plot showing spread, IQR, and outliers."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean_title = col_name.replace("_", " ").title()

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=clean,
            name=clean_title,
            marker_color="#A371F7",
            boxpoints="outliers",
            jitter=0.2,
        )
    )

    fig.update_layout(
        title=f"{clean_title} Spread & Outlier Analysis",
        xaxis_title="",
        yaxis_title=clean_title,
        showlegend=False,
    )
    _apply_theme(fig)

    min_v = num_stats.get("min", 0.0)
    max_v = num_stats.get("max", 0.0)
    iqr_v = num_stats.get("iqr", 0.0)
    outliers_c = num_stats.get("outliers_count", 0)

    caption = (
        f"{clean_title} ranges from {min_v:,.2f} to {max_v:,.2f} with an IQR of {iqr_v:,.2f} "
        f"and {outliers_c} statistical outlier values identified outside 1.5×IQR."
    )

    return {
        "id": f"box_{col_name}",
        "title": f"{clean_title} Spread & Outlier Analysis",
        "chart_type": "box",
        "fig": fig,
        "caption": caption,
        "columns": [col_name],
    }


def create_categorical_bar(
    series: pd.Series, col_name: str, cat_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a category frequency bar chart with tooltips and percentages."""
    clean_title = col_name.replace("_", " ").title()
    freq_table = cat_stats.get("frequency_table", {})

    if not freq_table:
        freq_table = {str(k): int(v) for k, v in series.dropna().value_counts().head(10).items()}

    cats = list(freq_table.keys())
    counts = list(freq_table.values())
    total = sum(counts) if sum(counts) > 0 else 1
    pcts = [(c / total) * 100.0 for c in counts]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cats,
            y=counts,
            text=[f"{p:.1f}%" for p in pcts],
            textposition="auto",
            marker_color="#2EA043",
        )
    )

    fig.update_layout(
        title=f"{clean_title} Breakdown by Frequency",
        xaxis_title=clean_title,
        yaxis_title="Count",
        showlegend=False,
    )
    _apply_theme(fig)

    top_val = cat_stats.get("top_value", cats[0] if cats else "N/A")
    top_pct = cat_stats.get("top_percentage", pcts[0] if pcts else 0.0)
    cardinality = cat_stats.get("cardinality", len(cats))

    caption = (
        f"'{top_val}' is the leading category representing {top_pct:.1f}% of observations "
        f"across {cardinality} distinct groups."
    )

    return {
        "id": f"bar_{col_name}",
        "title": f"{clean_title} Breakdown by Frequency",
        "chart_type": "bar",
        "fig": fig,
        "caption": caption,
        "columns": [col_name],
    }


def create_correlation_heatmap(corr_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate a correlation heatmap for numerical features."""
    pearson_mat = corr_data.get("pearson")
    if pearson_mat is None or pearson_mat.empty or len(pearson_mat.columns) < 2:
        return None

    cols = list(pearson_mat.columns)
    z_vals = pearson_mat.values

    text_vals = [[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in z_vals]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=[c.replace("_", " ").title() for c in cols],
            y=[c.replace("_", " ").title() for c in cols],
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="Viridis",
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "Pearson r"},
        )
    )

    fig.update_layout(
        title="Bivariate Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features",
    )
    _apply_theme(fig)

    pairs = corr_data.get("pairs", [])
    if pairs:
        top_p = pairs[0]
        c1 = top_p["col1"].replace("_", " ").title()
        c2 = top_p["col2"].replace("_", " ").title()
        r = top_p["pearson_r"]
        p_val = top_p["pearson_p"]
        caption = (
            f"Strongest linear association detected between {c1} and {c2} "
            f"(r = {r:+.2f}, p = {p_val:.2e})."
        )
    else:
        caption = "Correlation matrix displaying pairwise linear associations between numerical variables."

    return {
        "id": "corr_heatmap",
        "title": "Bivariate Correlation Heatmap",
        "chart_type": "heatmap",
        "fig": fig,
        "caption": caption,
        "columns": cols,
    }


def create_scatter_plot(
    df: pd.DataFrame, col1: str, col2: str, pair_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a bivariate scatter plot with linear trendline."""
    c1_title = col1.replace("_", " ").title()
    c2_title = col2.replace("_", " ").title()

    valid = df[[col1, col2]].dropna()
    x = valid[col1].astype(float)
    y = valid[col2].astype(float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"color": "#58A6FF", "opacity": 0.6, "size": 6},
            name="Data Points",
        )
    )

    # Add linear trendline
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        x_trend = np.linspace(x.min(), x.max(), 100)
        y_trend = m * x_trend + b
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                line={"color": "#F85149", "width": 2},
                name="Trendline (OLS)",
            )
        )

    fig.update_layout(
        title=f"{c1_title} vs. {c2_title} Relationship",
        xaxis_title=c1_title,
        yaxis_title=c2_title,
        showlegend=True,
    )
    _apply_theme(fig)

    r = pair_meta.get("pearson_r", 0.0)
    p = pair_meta.get("pearson_p", 1.0)
    strength = "strong" if abs(r) > 0.7 else ("moderate" if abs(r) > 0.3 else "weak")
    direction = "positive" if r > 0 else "negative"

    caption = (
        f"{c1_title} and {c2_title} exhibit a {strength} {direction} linear relationship "
        f"(r = {r:+.2f}, p = {p:.2e})."
    )

    return {
        "id": f"scatter_{col1}_{col2}",
        "title": f"{c1_title} vs. {c2_title} Relationship",
        "chart_type": "scatter",
        "fig": fig,
        "caption": caption,
        "columns": [col1, col2],
    }


def create_grouped_bar(
    group_comp: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a grouped category mean comparison bar chart with ANOVA annotation."""
    cat_col = group_comp["cat_col"]
    num_col = group_comp["num_col"]
    groups = group_comp["groups"]

    c_title = cat_col.replace("_", " ").title()
    n_title = num_col.replace("_", " ").title()

    cats = list(groups.keys())
    means = [groups[c]["mean"] for c in cats]
    stds = [groups[c]["std"] for c in cats]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cats,
            y=means,
            error_y={"type": "data", "array": stds, "visible": True},
            marker_color="#D29922",
        )
    )

    fig.update_layout(
        title=f"Mean {n_title} by {c_title}",
        xaxis_title=c_title,
        yaxis_title=f"Mean {n_title}",
        showlegend=False,
    )
    _apply_theme(fig)

    anova_f = group_comp.get("anova_f")
    anova_p = group_comp.get("anova_p")
    sig = group_comp.get("significant", False)

    if anova_f is not None and anova_p is not None:
        caption = (
            f"Variations in mean {n_title} across {c_title} are "
            f"{'statistically significant' if sig else 'not statistically significant'} "
            f"(one-way ANOVA F = {anova_f:.2f}, p = {anova_p:.2e})."
        )
    else:
        caption = f"Group means and standard deviations of {n_title} across {c_title} categories."

    return {
        "id": f"group_{cat_col}_{num_col}",
        "title": f"Mean {n_title} by {c_title}",
        "chart_type": "bar",
        "fig": fig,
        "caption": caption,
        "columns": [cat_col, num_col],
    }


def create_timeseries_chart(
    ts_meta: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Generate a time-series trend line with 7-day rolling average, safely skipping NaT values."""
    daily_df = ts_meta.get("resampled_daily")
    if daily_df is None or daily_df.empty or len(daily_df) < 5:
        return None

    dt_col = ts_meta["dt_col"]
    num_col = ts_meta["num_col"]
    d_title = dt_col.replace("_", " ").title()
    n_title = num_col.replace("_", " ").title()

    # Drop NaT safely
    valid = daily_df.dropna(subset=[dt_col, num_col]).sort_values(dt_col)
    if len(valid) < 5:
        return None

    # Compute 7-day rolling average
    valid["_rolling_7"] = valid[num_col].rolling(window=7, min_periods=1).mean()

    fig = go.Figure()
    # Daily points / line
    fig.add_trace(
        go.Scatter(
            x=valid[dt_col],
            y=valid[num_col],
            mode="lines",
            line={"color": "#58A6FF", "width": 1.5},
            opacity=0.6,
            name="Daily Total",
        )
    )
    # 7-day rolling average
    fig.add_trace(
        go.Scatter(
            x=valid[dt_col],
            y=valid["_rolling_7"],
            mode="lines",
            line={"color": "#3FB950", "width": 2.5},
            name="7-Day Rolling Trend",
        )
    )

    fig.update_layout(
        title=f"{n_title} Trend Over Time",
        xaxis_title=d_title,
        yaxis_title=f"Daily Total {n_title}",
        showlegend=True,
    )
    _apply_theme(fig)

    direction = ts_meta.get("trend_direction", "flat")
    slope = ts_meta.get("trend_slope", 0.0)
    r2 = ts_meta.get("trend_r2", 0.0)
    has_seasonality = ts_meta.get("has_seasonality", False)
    period = ts_meta.get("seasonal_period")

    seasonality_str = f" with identified {period} seasonality" if has_seasonality else ""
    caption = (
        f"{n_title} demonstrates a {direction} trend over time "
        f"(slope = {slope:+.2f}/day, R² = {r2:.3f}){seasonality_str}."
    )

    return {
        "id": f"ts_{dt_col}_{num_col}",
        "title": f"{n_title} Trend Over Time",
        "chart_type": "timeseries",
        "fig": fig,
        "caption": caption,
        "columns": [dt_col, num_col],
    }


def generate_visualizations(
    df_clean: pd.DataFrame,
    eda_results: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate a comprehensive list of structured visualization objects with metadata and captions.

    Driven strictly by detected column semantic types; excludes constant and ID-like columns.

    Args:
        df_clean: Cleaned DataFrame.
        eda_results: Optional precomputed EDA results from run_eda.

    Returns:
        List of visualization dictionaries with keys (id, title, chart_type, fig, caption, columns).
    """
    if eda_results is None:
        eda_results = run_eda(df_clean)

    classified = eda_results.get("classified_columns", {})
    num_cols = classified.get("numeric", [])
    cat_cols = classified.get("categorical", [])
    dt_cols = classified.get("datetime", [])

    charts: List[Dict[str, Any]] = []

    # 1. Univariate Numerical: Histogram & Boxplot for each numeric column (up to top 5)
    u_num = eda_results.get("univariate_numeric", {})
    for col in num_cols[:5]:
        stats_meta = u_num.get(col, {})
        hist_chart = create_histogram(df_clean[col], col, stats_meta)
        charts.append(hist_chart)

        box_chart = create_boxplot(df_clean[col], col, stats_meta)
        charts.append(box_chart)

    # 2. Univariate Categorical: Bar Chart for each categorical column (up to top 5)
    u_cat = eda_results.get("univariate_categorical", {})
    for col in cat_cols[:5]:
        stats_meta = u_cat.get(col, {})
        bar_chart = create_categorical_bar(df_clean[col], col, stats_meta)
        charts.append(bar_chart)

    # 3. Bivariate Numerical: Correlation Heatmap
    corr_meta = eda_results.get("correlations", {})
    if len(num_cols) >= 2:
        heatmap_chart = create_correlation_heatmap(corr_meta)
        if heatmap_chart is not None:
            charts.append(heatmap_chart)

        # Scatter plots with OLS trendline for top correlated pairs (up to 3 pairs)
        pairs = corr_meta.get("pairs", [])
        for pair in pairs[:3]:
            scatter_chart = create_scatter_plot(df_clean, pair["col1"], pair["col2"], pair)
            charts.append(scatter_chart)

    # 4. Bivariate Categorical x Numerical: Grouped Mean Bar Chart
    group_comps = eda_results.get("group_comparisons", [])
    for comp in group_comps[:3]:
        group_chart = create_grouped_bar(comp)
        charts.append(group_chart)

    # 5. Time-Series Trend Charts
    ts_results = eda_results.get("timeseries_analysis", [])
    for ts in ts_results[:2]:
        ts_chart = create_timeseries_chart(ts)
        if ts_chart is not None:
            charts.append(ts_chart)

    return charts
