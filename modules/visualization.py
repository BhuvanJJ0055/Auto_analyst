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

PLOTLY_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "scale": 2,  # Crisp 2x publication resolution
    },
    "displayModeBar": True,
    "displaylogo": False,
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


def format_p_value(p: Optional[float], include_prefix: bool = True) -> str:
    """Format a p-value cleanly, never printing 0.00e+00.

    Returns '< 0.001' (or 'p < 0.001') for very small values,
    and standard decimal formatting for higher p-values.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = N/A" if include_prefix else "N/A"
    if p < 0.001:
        return "p < 0.001" if include_prefix else "< 0.001"
    if p < 0.01:
        return f"p = {p:.3f}" if include_prefix else f"{p:.3f}"
    return f"p = {p:.2f}" if include_prefix else f"{p:.2f}"


def describe_correlation(r: float) -> str:
    """Return a plain-English interpretation of a Pearson correlation r value.

    Examples:
        0.84  -> 'very strong positive relationship'
       -0.65  -> 'strong negative relationship'
        0.45  -> 'moderate positive relationship'
       -0.25  -> 'weak negative relationship'
        0.05  -> 'very weak positive relationship'
        0.01  -> 'negligible relationship'
    """
    abs_r = abs(r)
    if abs_r < 0.05:
        return "negligible relationship"
    if abs_r >= 0.8:
        strength = "very strong"
    elif abs_r >= 0.6:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"

    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction} relationship"


def create_correlation_heatmap(
    corr_data: Dict[str, Any], lower_only: bool = True
) -> Optional[Dict[str, Any]]:
    """Generate a correlation heatmap for numerical features.

    Parameters
    ----------
    corr_data : Dict[str, Any]
        Dictionary containing 'pearson' DataFrame and 'pairs' list.
    lower_only : bool, default True
        If True, display only the lower triangle (including diagonal),
        masking the redundant upper triangle to reduce visual clutter.
    """
    pearson_mat = corr_data.get("pearson")
    if pearson_mat is None or pearson_mat.empty or len(pearson_mat.columns) < 2:
        return None

    cols = list(pearson_mat.columns)
    z_vals = np.array(pearson_mat.values, dtype=float)
    n_features = len(cols)

    if lower_only:
        z_vals = z_vals.copy()
        for i in range(n_features):
            for j in range(n_features):
                if j > i:
                    z_vals[i, j] = np.nan

    text_vals = [
        [f"{val:.2f}" if not np.isnan(val) else "" for val in row]
        for row in z_vals
    ]

    clean_labels = [c.replace("_", " ").title() for c in cols]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=clean_labels,
            y=clean_labels,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="Viridis",
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "Pearson r"},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Bivariate Correlation Heatmap" + (" (Lower Triangle)" if lower_only else ""),
        xaxis_title="Features",
        yaxis_title="Features",
        yaxis={"autorange": "reversed"},
    )
    _apply_theme(fig)

    pairs = corr_data.get("pairs", [])
    if pairs:
        top_p = pairs[0]
        c1 = top_p["col1"].replace("_", " ").title()
        c2 = top_p["col2"].replace("_", " ").title()
        r = top_p["pearson_r"]
        p_val = top_p["pearson_p"]
        reading = describe_correlation(r)
        p_str = format_p_value(p_val)
        caption = (
            f"Strongest linear association detected between {c1} and {c2}: "
            f"{reading} (r = {r:+.2f}, {p_str})."
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
    reading = describe_correlation(r)
    p_str = format_p_value(p)

    caption = (
        f"{c1_title} and {c2_title} exhibit a {reading} "
        f"(r = {r:+.2f}, {p_str})."
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
        p_str = format_p_value(anova_p)
        caption = (
            f"Variations in mean {n_title} across {c_title} are "
            f"{'statistically significant' if sig else 'not statistically significant'} "
            f"(one-way ANOVA F = {anova_f:.2f}, {p_str})."
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


def _format_slope(slope: float, col_name: str) -> str:
    """Format time-series slope with appropriate sign and currency/unit."""
    col_lower = col_name.lower()
    monetary_keywords = [
        "revenue", "profit", "sales", "price", "cost", "income", "amount", "spend", "fee"
    ]
    is_monetary = any(kw in col_lower for kw in monetary_keywords)
    sign = "+" if slope >= 0 else "-"
    abs_slope = abs(slope)

    if is_monetary:
        return f"{sign}₹{abs_slope:,.2f}"
    elif "age" in col_lower:
        return f"{sign}{abs_slope:.2f} yrs"
    elif any(kw in col_lower for kw in ["quantity", "unit", "orders", "volume"]):
        return f"{sign}{abs_slope:,.2f} units"
    else:
        return f"{sign}{abs_slope:.2f}"


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
    agg_label = ts_meta.get("agg_label", "Daily Total")
    agg_func = ts_meta.get("agg_func", "sum")

    # Drop NaT safely
    valid = daily_df.dropna(subset=[dt_col, num_col]).sort_values(dt_col)
    if len(valid) < 5:
        return None

    # Compute 7-day rolling average
    valid["_rolling_7"] = valid[num_col].rolling(window=7, min_periods=1).mean()

    # Determine axis title and trace label matching aggregation semantics
    if agg_func == "mean":
        yaxis_title = f"{agg_label} {n_title}"
        trace_name = f"Daily {agg_label}"
        chart_title = f"{agg_label} {n_title} Over Time"
    else:
        yaxis_title = f"Daily Total {n_title}"
        trace_name = "Daily Total"
        chart_title = f"{n_title} Trend Over Time"

    fig = go.Figure()
    # Daily points / line
    fig.add_trace(
        go.Scatter(
            x=valid[dt_col],
            y=valid[num_col],
            mode="lines",
            line={"color": "#58A6FF", "width": 1.5},
            opacity=0.6,
            name=trace_name,
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
        title=chart_title,
        xaxis_title=d_title,
        yaxis_title=yaxis_title,
        showlegend=True,
    )
    _apply_theme(fig)

    direction = ts_meta.get("trend_direction", "flat")
    strength = ts_meta.get("trend_strength", "flat")
    slope = ts_meta.get("trend_slope", 0.0)
    r2 = ts_meta.get("trend_r2", 0.0)
    has_seasonality = ts_meta.get("has_seasonality", False)
    period = ts_meta.get("seasonal_period")

    slope_str = _format_slope(slope, num_col)
    seasonality_str = f" with identified {period} seasonality" if has_seasonality else ""

    if ts_meta.get("has_trend", False):
        dir_clean = "upward" if slope > 0 else "downward"
        if strength in ["weak", "moderate", "strong"]:
            trend_phrase = f"a {strength} {dir_clean} trend"
        else:
            article = "an" if dir_clean == "upward" else "a"
            trend_phrase = f"{article} {dir_clean} trend"
        caption = (
            f"{yaxis_title} demonstrates {trend_phrase} over time "
            f"(slope = {slope_str}/day, R² = {r2:.2f}){seasonality_str}."
        )
    else:
        caption = (
            f"{yaxis_title} demonstrates no clear trend over time "
            f"(slope = {slope_str}/day, R² = {r2:.2f}){seasonality_str}."
        )

    return {
        "id": f"ts_{dt_col}_{num_col}",
        "title": chart_title,
        "chart_type": "timeseries",
        "fig": fig,
        "caption": caption,
        "columns": [dt_col, num_col],
    }


def create_pca_chart(
    pca_res: Dict[str, Any],
    hue_series: Optional[pd.Series] = None,
    hue_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate 2D Principal Component Analysis projection scatter plot."""
    if pca_res is None or "projections" not in pca_res:
        return None

    projections = pca_res["projections"]
    if "PC1" not in projections.columns or "PC2" not in projections.columns:
        return None

    ev = pca_res.get("explained_variance_ratio", [0.0, 0.0])
    pc1_pct = ev[0] * 100 if len(ev) > 0 else 0.0
    pc2_pct = ev[1] * 100 if len(ev) > 1 else 0.0
    total_ev = pca_res.get("total_explained_variance", 0.0) * 100

    plot_df = projections[["PC1", "PC2"]].copy()
    if hue_series is not None:
        plot_df["Category"] = hue_series.loc[plot_df.index].astype(str)

    fig = go.Figure()
    if hue_series is not None and "Category" in plot_df.columns:
        for cat, grp in plot_df.groupby("Category"):
            fig.add_trace(
                go.Scatter(
                    x=grp["PC1"],
                    y=grp["PC2"],
                    mode="markers",
                    name=str(cat),
                    marker={"size": 6, "opacity": 0.75},
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df["PC1"],
                y=plot_df["PC2"],
                mode="markers",
                name="Observations",
                marker={"color": "#58A6FF", "size": 6, "opacity": 0.75},
            )
        )

    legend_title = hue_name.replace("_", " ").title() if hue_name else "Category"
    fig.update_layout(
        title="2D PCA Dimensionality Reduction",
        xaxis_title=f"Principal Component 1 ({pc1_pct:.1f}% Variance)",
        yaxis_title=f"Principal Component 2 ({pc2_pct:.1f}% Variance)",
        showlegend=(hue_series is not None),
        legend={"title": legend_title} if hue_series is not None else {},
    )
    _apply_theme(fig)

    caption = (
        f"2D PCA captures {total_ev:.1f}% of cumulative variance across {len(pca_res['columns'])} "
        f"numerical features (PC1: {pc1_pct:.1f}%, PC2: {pc2_pct:.1f}%)."
    )

    return {
        "id": "pca_2d",
        "title": "2D PCA Dimensionality Reduction",
        "chart_type": "scatter",
        "fig": fig,
        "caption": caption,
        "columns": pca_res["columns"],
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

    # 6. PCA 2D Dimensionality Reduction (if >= 2 numeric features)
    if eda_results.get("pca") is not None:
        hue_s = df_clean[cat_cols[0]] if cat_cols else None
        hue_n = cat_cols[0] if cat_cols else None
        pca_chart = create_pca_chart(eda_results["pca"], hue_series=hue_s, hue_name=hue_n)
        if pca_chart is not None:
            charts.append(pca_chart)

    return charts
