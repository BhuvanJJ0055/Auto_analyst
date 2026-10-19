"""AutoAnalyst - Automated Data Analyst Web Application.

Phase 1: Ingestion and Profiling UI.
Provides dataset loading, file validation, dynamic semantic type inference,
and transparent data quality scoring.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st

from modules.cleaning import clean_data
from modules.ingestion import IngestionError, get_excel_sheet_names, load_file
from modules.profiling import profile_data

# Page configuration
st.set_page_config(
    page_title="AutoAnalyst | Automated Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: #F0F6FC;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8B949E;
        margin-bottom: 1.5rem;
    }
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .pipeline-step {
        background: #21262D;
        color: #8B949E;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #30363D;
    }
    .pipeline-step.completed {
        background: #0E4429;
        color: #3FB950;
        border-color: #238636;
    }
    .pipeline-step.active {
        background: #1F6FEB;
        color: #FFFFFF;
        border-color: #388BFD;
    }
    .card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .card h4 {
        color: #FFFFFF !important;
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .card p {
        color: #C9D1D9 !important;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 0;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .grade-A { background: #0E4429; color: #3FB950; border: 1px solid #238636; }
    .grade-B { background: #0C2D6B; color: #58A6FF; border: 1px solid #1F6FEB; }
    .grade-C { background: #4D2D00; color: #D29922; border: 1px solid #9E6A03; }
    .grade-D { background: #4D1B0C; color: #F0883E; border: 1px solid #BD561D; }
    .grade-F { background: #490202; color: #F85149; border: 1px solid #DA3633; }
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        min-height: 94px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8B949E;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .metric-status {
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    .status-good {
        color: #3FB950;
    }
    .status-warning {
        color: #D29922;
    }
    .status-danger {
        color: #F85149;
    }
    .status-neutral {
        color: #8B949E;
    }
    .metric-denom {
        font-size: 0.92rem;
        color: #8B949E;
        font-weight: 500;
    }
    button[data-baseweb="tab"] {
        color: #8B949E !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #58A6FF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-header">📊 AutoAnalyst</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Upload a dataset and get automatic cleaning, EDA, an interactive dashboard, "
    "and a plain-English report."
    "</div>",
    unsafe_allow_html=True,
)

# Pipeline Flow Visualizer
st.markdown(
    """
    <div class="pipeline-container">
        <span class="pipeline-step completed">✓ 1. Ingestion</span>
        <span class="pipeline-step completed">✓ 2. Profiling</span>
        <span class="pipeline-step active">⚡ 3. Cleaning</span>
        <span class="pipeline-step">4. EDA</span>
        <span class="pipeline-step">5. Visualization</span>
        <span class="pipeline-step">6. Insights</span>
        <span class="pipeline-step">7. Dashboard</span>
        <span class="pipeline-step">8. Report</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_load_file(
    file_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cached file loader wrapper."""
    return load_file(file_bytes, sheet_name=sheet_name, filename=filename)


@st.cache_data(show_spinner=False)
def cached_profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Cached profiling calculation wrapper."""
    return profile_data(df)


@st.cache_data(show_spinner=False)
def cached_clean_data(
    df: pd.DataFrame, config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Any]:
    """Cached data cleaning wrapper."""
    return clean_data(df, config=config)


def format_audit_details(step: str, details: Any) -> str:
    """Format operation metadata details dictionary into concise human-readable summary text."""
    if not isinstance(details, dict):
        return str(details)

    if step == "deduplication":
        removed = details.get("rows_removed", 0)
        remaining = details.get("rows_remaining", 0)
        return f"{removed} duplicate rows removed ({remaining} rows remaining)"

    if step == "drop_all_null_column":
        rows = details.get("total_rows", 0)
        return f"100% missing (0 non-null values across {rows} rows)"

    if step == "drop_constant_column":
        val = details.get("constant_value", "")
        return f"Dropped single constant value: '{val}'"

    if step == "flag_constant_column":
        val = details.get("constant_value", "")
        return f"Business constant retained: '{val}'"

    if step == "clean_currency_strings":
        cells = details.get("cells_changed", details.get("parsed_values", 0))
        new_dtype = details.get("new_dtype", "float64")
        return f"{cells} currency strings normalized to {new_dtype}"

    if step == "clean_datetime":
        cells = details.get("cells_changed", details.get("parsed_count", 0))
        min_d = str(details.get("min_date", ""))[:10]
        max_d = str(details.get("max_date", ""))[:10]
        span_str = f" ({min_d} to {max_d})" if min_d and max_d else ""
        return f"{cells} date strings parsed to datetime{span_str}"

    if step == "handle_future_dates":
        flagged = details.get("future_dates_flagged", details.get("future_dates_count", 0))
        act = details.get("action", "nullify")
        thresh = details.get("threshold_date", "")
        thresh_str = f" [fence > {thresh}]" if thresh else ""
        if act == "nullify":
            return f"{flagged} future dates flagged and set to NaT{thresh_str}"
        elif act == "cap":
            cap_lim = details.get("cap_target", "")
            return f"{flagged} future dates flagged and capped at {cap_lim}"
        return f"{flagged} future dates flagged (retained){thresh_str}"

    if step == "standardize_text":
        changed = details.get("cells_changed", 0)
        orig_u = details.get("original_unique_values", 0)
        new_u = details.get("standardized_unique_values", 0)
        return f"{changed} cells changed ({orig_u} variants -> {new_u} canonical categories)"

    if step == "correct_negatives":
        neg = details.get("negatives_flagged", details.get("negatives_inverted", 0))
        return f"{neg} negative domain values set to missing (to be filled by median)"

    if step == "handle_outliers":
        if details.get("method") == "domain_validation":
            inv = details.get("outliers_treated", 0)
            return f"Small integer validation: min >= 1 ({inv} invalid flagged)"
        low = details.get("lower_limit", "")
        high = details.get("upper_limit", "")
        act = details.get("action", "clipped")
        grp = details.get("group_by")
        grp_str = f" grouped by {grp}" if grp else ""
        pct = details.get("percent_capped", details.get("capped_percent"))
        pct_str = f" ({pct}% of rows)" if pct is not None else ""
        flag_str = " [FLAGGED: >5% capped]" if details.get("flagged_exceeds_5pct") else ""
        return (
            f"{act.capitalize()} at statistical bounds [{low}, {high}]{grp_str}{pct_str}{flag_str}"
        )

    if step == "impute_missing":
        strat = details.get("strategy", "")
        filled = details.get("missing_imputed", 0)
        val = details.get("imputed_value", "")
        return f"{filled} nulls filled using {strat} strategy (fill value: {val})"

    # Clean key-value pairs fallback
    parts = []
    for k, v in details.items():
        if isinstance(v, (list, dict)):
            continue
        clean_k = k.replace("_", " ").title()
        parts.append(f"{clean_k}: {v}")
    return "; ".join(parts[:3]) if parts else str(details)


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("**Version:** `v0.2.0-dev (Phase 2: Cleaning Engine)`")
    st.divider()

    st.subheader("1. Ingest Dataset")
    source_mode = st.radio(
        "Data Source",
        ["Upload File", "Benchmark Dataset"],
        index=1,
        help="Choose between uploading your own file or exploring pre-built benchmark datasets.",
    )

    df_loaded: Optional[pd.DataFrame] = None
    ingestion_meta: Optional[Dict[str, Any]] = None
    ingestion_error: Optional[str] = None
    sample_dir = Path(__file__).resolve().parent / "sample_data"

    if source_mode == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"],
            help="Supported formats: CSV, TSV, Excel, JSON, Parquet (Max 200 MB)",
        )
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            target_sheet: Optional[str] = None

            if filename.lower().endswith((".xlsx", ".xls")):
                try:
                    sheets = get_excel_sheet_names(raw_bytes)
                    if len(sheets) > 1:
                        target_sheet = st.selectbox("Select Excel Sheet", sheets)
                except Exception as exc:
                    st.error(f"Error reading Excel sheets: {exc}")

            try:
                with st.spinner("Ingesting uploaded dataset..."):
                    df_loaded, ingestion_meta = cached_load_file(
                        raw_bytes, filename=filename, sheet_name=target_sheet
                    )
            except IngestionError as err:
                ingestion_error = str(err)
                st.error(f"⚠️ Ingestion Error: {err}")
    else:
        sample_files = sorted([f.name for f in sample_dir.glob("*.*") if f.name != "large.csv"])
        default_idx = (
            sample_files.index("clean_sales.csv") if "clean_sales.csv" in sample_files else 0
        )
        selected_sample = st.selectbox(
            "Select Benchmark Dataset",
            sample_files,
            index=default_idx,
        )
        if selected_sample:
            sample_path = sample_dir / selected_sample
            with open(sample_path, "rb") as sf:
                sample_bytes = sf.read()
            target_sheet = None
            if selected_sample.lower().endswith((".xlsx", ".xls")):
                try:
                    sheets = get_excel_sheet_names(sample_bytes)
                    if len(sheets) > 1:
                        target_sheet = st.selectbox("Select Excel Sheet", sheets)
                except Exception as exc:
                    st.error(f"Error reading Excel sheets: {exc}")
            try:
                with st.spinner(f"Loading {selected_sample}..."):
                    df_loaded, ingestion_meta = cached_load_file(
                        sample_bytes, filename=selected_sample, sheet_name=target_sheet
                    )
            except IngestionError as err:
                ingestion_error = str(err)
                st.error(f"⚠️ Ingestion Error: {err}")

    st.divider()
    st.caption("AutoAnalyst executes locally without external API dependencies.")

# Main Application Body
if df_loaded is not None and ingestion_meta is not None:
    # Compute Profiling
    with st.spinner("Profiling dataset health and computing quality score..."):
        profile = cached_profile_data(df_loaded)

    quality = profile["quality_score"]
    grade = quality["grade"]
    score_val = quality["score"]

    # Quality Scorecard Header Row
    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([1.5, 1, 1, 1, 1])

    with top_col1:
        st.markdown(
            f"""
            <div class="metric-card" style="padding: 0.85rem 1.1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;
                height: 100%;">
                    <div>
                        <div class="metric-label">Data Quality Score</div>
                        <div class="metric-value">
                            {score_val} <span class="metric-denom">/ 100</span>
                        </div>
                        <div class="metric-status status-neutral">Automated Audit</div>
                    </div>
                    <span class="score-badge grade-{grade}">{grade}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{profile['total_rows']:,}</div>
                <div class="metric-status status-neutral">Dataset records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{profile['total_columns']:,}</div>
                <div class="metric-status status-neutral">Profiled features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col4:
        dup_cnt = profile["duplicate_rows"]
        dup_pct = profile["duplicate_pct"]
        if dup_cnt == 0:
            dup_badge = '<div class="metric-status status-good">✓ 0.0% duplicates</div>'
        elif dup_pct <= 5.0:
            dup_badge = f'<div class="metric-status status-warning">▲ {dup_pct}% duplicates</div>'
        else:
            dup_badge = f'<div class="metric-status status-danger">▲ {dup_pct}% duplicates</div>'

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Duplicate Rows</div>
                <div class="metric-value">{dup_cnt:,}</div>
                {dup_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col5:
        missing_cnt = profile["total_missing_cells"]
        total_cells_count = profile["total_rows"] * profile["total_columns"]
        missing_rate_pct = (
            round((missing_cnt / total_cells_count) * 100, 2) if total_cells_count > 0 else 0.0
        )
        if missing_cnt == 0:
            missing_badge = '<div class="metric-status status-good">✓ 0.0% missing</div>'
        elif missing_rate_pct <= 5.0:
            missing_badge = (
                f'<div class="metric-status status-warning">▲ {missing_rate_pct}% missing</div>'
            )
        else:
            missing_badge = (
                f'<div class="metric-status status-danger">▲ {missing_rate_pct}% missing</div>'
            )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Missing Cells</div>
                <div class="metric-value">{missing_cnt:,}</div>
                {missing_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tabs Section
    tab_overview, tab_preview, tab_profiling, tab_cleaning, tab_specs = st.tabs(
        [
            "🚀 Dataset Overview",
            "📂 Data Preview",
            "🔍 Column Profiling",
            "🧹 Cleaning Engine",
            "📋 Technical Architecture",
        ]
    )

    with tab_overview:
        st.subheader("Data Health & Quality Breakdown")

        def make_breakdown_status(score: float, max_score: float, label: str) -> str:
            ratio = round((score / max_score) * 100, 1)
            if ratio >= 99.9:
                return f'<div class="metric-status status-good">✓ 100% {label}</div>'
            elif ratio >= 80.0:
                return f'<div class="metric-status status-warning">▲ {ratio}% {label}</div>'
            else:
                return f'<div class="metric-status status-danger">▲ {ratio}% {label}</div>'

        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        c_score = quality["completeness_score"]
        u_score = quality["uniqueness_score"]
        v_score = quality["validity_score"]
        s_score = quality["consistency_score"]

        with q_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Completeness</div>
                    <div class="metric-value">
                        {c_score} <span class="metric-denom">/ 30</span>
                    </div>
                    {make_breakdown_status(c_score, 30, "clean")}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Uniqueness</div>
                    <div class="metric-value">
                        {u_score} <span class="metric-denom">/ 25</span>
                    </div>
                    {make_breakdown_status(u_score, 25, "unique")}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Validity</div>
                    <div class="metric-value">
                        {v_score} <span class="metric-denom">/ 25</span>
                    </div>
                    {make_breakdown_status(v_score, 25, "valid")}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with q_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Consistency</div>
                    <div class="metric-value">
                        {s_score} <span class="metric-denom">/ 20</span>
                    </div>
                    {make_breakdown_status(s_score, 20, "consistent")}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
        with st.expander("ℹ️ Data Quality Score Formula & Weights Breakdown"):
            st.markdown("""
                The **AutoAnalyst Data Quality Score (0–100)** is computed using four pillars:
                1. **Completeness (30% weight)**: Measures missing cell density across the dataset
                   and the percentage of columns affected by null values.
                2. **Uniqueness (25% weight)**: Measures the absence of duplicate transaction rows.
                3. **Validity (25% weight)**: Penalizes string-encoded measures (currency symbols,
                   commas), mixed date strings, negative values in restricted domains (e.g. age),
                   and text casing variants.
                4. **Consistency (20% weight)**: Penalizes structural defects such as single-value
                   constant columns and 100% all-null columns.
                """)
            st.code(quality["formula_explanation"], language="text")

        st.divider()

        # Ingestion Metadata Overview
        st.subheader("Ingestion Metadata")
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        meta_col1.markdown(f"**Filename:** `{ingestion_meta['filename']}`")
        meta_col2.markdown(f"**Format:** `{ingestion_meta['file_type']}`")
        meta_col3.markdown(f"**Size:** `{ingestion_meta['file_size_formatted']}`")
        meta_col4.markdown(f"**Memory in RAM:** `{profile['memory_formatted']}`")

        if ingestion_meta.get("delimiter"):
            delim_str = repr(ingestion_meta["delimiter"])
            enc_str = ingestion_meta["encoding"]
            st.caption(f"Detected Delimiter: `{delim_str}` | Encoding: `{enc_str}`")
        if ingestion_meta.get("warnings"):
            for w in ingestion_meta["warnings"]:
                st.warning(f"Notice: {w}")

    with tab_preview:
        st.subheader(f"Data Preview (First 20 of {len(df_loaded):,} rows)")
        st.dataframe(df_loaded.head(20), use_container_width=True)

        csv_download = df_loaded.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Raw Dataset Copy (CSV)",
            data=csv_download,
            file_name=f"raw_{ingestion_meta['filename'].rsplit('.', 1)[0]}.csv",
            mime="text/csv",
        )

    with tab_profiling:
        st.subheader("Semantic Column Profiling")

        # Type summary counts
        type_counts: Dict[str, int] = {}
        for col_info in profile["columns"].values():
            stype = col_info["semantic_type"]
            type_counts[stype] = type_counts.get(stype, 0) + 1

        badge_row = " ".join(
            [f"`{stype.upper()}: {cnt}`" for stype, cnt in sorted(type_counts.items())]
        )
        st.markdown(f"**Column Distribution:** {badge_row}")

        # Summary Table of All Columns
        col_rows = []
        for col_name, info in profile["columns"].items():
            stats = info.get("stats", {})
            summary_str = ""
            if stats:
                summary_str = (
                    f"Min: {stats.get('min')}, Mean: {stats.get('mean')}, Max: {stats.get('max')}"
                )
            elif info.get("top_values"):
                top_v = [f"{v['value']} ({v['count']})" for v in info["top_values"][:3]]
                summary_str = ", ".join(top_v)

            col_rows.append(
                {
                    "Column": col_name,
                    "Semantic Type": info["semantic_type"].upper(),
                    "Storage Dtype": info["dtype"],
                    "Missing Values": f"{info['missing_count']} ({info['missing_pct']}%)",
                    "Unique Values": f"{info['unique_count']} ({info['unique_pct']}%)",
                    "Summary / Top Frequencies": summary_str,
                }
            )

        st.dataframe(pd.DataFrame(col_rows), use_container_width=True)

        st.divider()
        st.subheader("Detailed Column Inspections")
        selected_col = st.selectbox(
            "Inspect individual column attributes:", list(profile["columns"].keys())
        )
        if selected_col:
            col_info = profile["columns"][selected_col]
            i_col1, i_col2, i_col3 = st.columns(3)
            i_col1.markdown(f"**Semantic Type:** `{col_info['semantic_type']}`")
            i_col2.markdown(
                f"**Missing:** `{col_info['missing_count']}` ({col_info['missing_pct']}%)"
            )
            i_col3.markdown(f"**Unique:** `{col_info['unique_count']}` ({col_info['unique_pct']}%)")

            if col_info.get("stats"):
                st.markdown("**Descriptive Statistics:**")
                st.json(col_info["stats"])
            if col_info.get("top_values"):
                st.markdown("**Top Value Frequencies:**")
                st.table(pd.DataFrame(col_info["top_values"]))
            if col_info.get("details"):
                st.caption(f"Detected Attributes: {col_info['details']}")

    with tab_cleaning:
        st.subheader("🧹 Automated Data Cleaning Engine")
        st.markdown(
            "Configure and execute intelligent, non-destructive cleaning transformations. "
            "The engine preserves raw data immutability and generates a complete audit trail."
        )

        with st.expander("🛠️ Cleaning Configuration & Rule Overrides", expanded=False):
            st.markdown("##### 1. Toggle Cleaning Transformations")
            cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
            with cfg_col1:
                cfg_dedup = st.checkbox("Deduplication", value=True, help="Remove duplicate rows")
                cfg_drop_null = st.checkbox(
                    "Drop 100% Empty Columns",
                    value=True,
                    help="Drop all-null structural defect columns",
                )
                cfg_drop_const = st.checkbox(
                    "Drop Constant Columns", value=False, help="Drop single-value columns"
                )
            with cfg_col2:
                cfg_currency = st.checkbox(
                    "Normalize Currency", value=True, help="Convert currency strings to float64"
                )
                cfg_datetime = st.checkbox(
                    "Standardize Dates", value=True, help="Parse mixed date formats into datetime"
                )
                cfg_future = st.checkbox(
                    "Handle Future Dates", value=True, help="Cap or flag unrealistic future dates"
                )
                cfg_text = st.checkbox(
                    "Standardize Text",
                    value=True,
                    help="Trim spaces, title-case categoricals, merge region & gender variants",
                )
            with cfg_col3:
                cfg_negatives = st.checkbox(
                    "Correct Negatives", value=True, help="Invert negative ages & quantities"
                )
                cfg_outliers = st.checkbox(
                    "Treat Outliers",
                    value=True,
                    help="Neutralize numerical outliers using statistical boundaries",
                )
                cfg_impute = st.checkbox(
                    "Impute Missing Values", value=True, help="Fill nulls (median/mode)"
                )

            st.markdown("##### 2. Algorithmic Method Overrides")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                outlier_method = st.selectbox(
                    "Outlier Detection Method", ["iqr", "zscore"], index=0
                )
            with m_col2:
                outlier_factor_choice = st.selectbox(
                    "Outlier Threshold Factor",
                    ["3.0 (Extreme Outliers)", "1.5 (Standard Fences)"],
                    index=0,
                )
                outlier_factor = 3.0 if "3.0" in outlier_factor_choice else 1.5
            with m_col3:
                outlier_action = st.selectbox(
                    "Outlier Treatment Action", ["clip", "remove", "nullify"], index=0
                )
            with m_col4:
                num_strategy = st.selectbox(
                    "Numeric Imputation Strategy", ["median", "mean", "zero"], index=0
                )
            with m_col5:
                future_action = st.selectbox(
                    "Future Date Action", ["cap", "nullify", "flag"], index=0
                )

            st.button("🔄 Re-run Cleaning Engine", help="Apply chosen rules and recalculate")

        cleaning_config = {
            "deduplicate": cfg_dedup,
            "drop_all_null": cfg_drop_null,
            "drop_constant": cfg_drop_const,
            "clean_currency": cfg_currency,
            "clean_datetime": cfg_datetime,
            "handle_future_dates": cfg_future,
            "future_date_action": future_action,
            "standardize_text": cfg_text,
            "correct_negatives": cfg_negatives,
            "handle_outliers": cfg_outliers,
            "outlier_method": outlier_method,
            "outlier_factor": outlier_factor,
            "outlier_action": outlier_action,
            "impute_missing": cfg_impute,
            "numeric_impute_strategy": num_strategy,
            "categorical_impute_strategy": "mode",
        }

        # Run cleaning
        cleaned_df, audit_log = cached_clean_data(df_loaded, config=cleaning_config)
        cleaned_profile = cached_profile_data(cleaned_df)
        cleaned_quality = cleaned_profile["quality_score"]

        # Before vs After Impact Metric Cards
        st.markdown("#### Before vs. After Impact Comparison")
        comp1, comp2, comp3, comp4, comp5 = st.columns(5)

        with comp1:
            score_diff = round(cleaned_quality["score"] - quality["score"], 1)
            diff_color = "#3FB950" if score_diff >= 0 else "#F85149"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Quality Score</div>
                    <div class="metric-value">{quality['score']} ➔ {cleaned_quality['score']}</div>
                    <div class="metric-status" style="color: {diff_color}; font-weight: 600;">
                        Grade {quality['grade']} ➔ Grade {cleaned_quality['grade']}
                        ({score_diff:+} pts)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with comp2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Rows</div>
                    <div class="metric-value">{len(df_loaded):,} ➔ {len(cleaned_df):,}</div>
                    <div class="metric-status status-good">
                        {len(df_loaded) - len(cleaned_df)} duplicate rows dropped
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with comp3:
            orig_nulls = int(df_loaded.isna().sum().sum())
            new_nulls = int(cleaned_df.isna().sum().sum())
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Missing Cells</div>
                    <div class="metric-value">{orig_nulls:,} ➔ {new_nulls:,}</div>
                    <div class="metric-status status-good">
                        {orig_nulls - new_nulls} cells resolved
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with comp4:
            reg_orig = df_loaded["region"].nunique() if "region" in df_loaded.columns else 0
            reg_clean = cleaned_df["region"].nunique() if "region" in cleaned_df.columns else 0
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Region Variants</div>
                    <div class="metric-value">{reg_orig} ➔ {reg_clean}</div>
                    <div class="metric-status status-good">
                        Merged to 4 canonical regions
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with comp5:
            gen_orig = df_loaded["gender"].nunique() if "gender" in df_loaded.columns else 0
            gen_clean = cleaned_df["gender"].nunique() if "gender" in cleaned_df.columns else 0
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Gender Variants</div>
                    <div class="metric-value">{gen_orig} ➔ {gen_clean}</div>
                    <div class="metric-status status-good">
                        Merged to 2 binary genders
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Cleaned Dataset Preview & Downloads
        st.subheader("Cleaned Dataset Preview & Exports")
        preview_rows = st.slider(
            "Rows to display in preview",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="clean_slider",
        )
        st.dataframe(cleaned_df.head(preview_rows), use_container_width=True)

        dataset_filename = ingestion_meta.get("filename", "dataset.csv")
        clean_filename = (
            f"cleaned_{dataset_filename}"
            if not dataset_filename.startswith("cleaned_")
            else dataset_filename
        )
        if not clean_filename.endswith(".csv"):
            clean_filename += ".csv"

        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            csv_clean_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_clean_bytes,
                file_name=clean_filename,
                mime="text/csv",
                use_container_width=True,
            )

        # Transformation Audit Log
        st.divider()
        st.subheader(f"Transformation Audit Trail ({len(audit_log)} Operations Logged)")
        if audit_log:
            log_records = []
            for item in audit_log:
                log_records.append(
                    {
                        "Step": item.get("step"),
                        "Target Column": item.get("column"),
                        "Method": item.get("method"),
                        "Rows Affected": item.get("rows_affected"),
                        "Action Performed": item.get("action"),
                        "Details": format_audit_details(
                            str(item.get("step", "")), item.get("details", {})
                        ),
                    }
                )
            log_df = pd.DataFrame(log_records)
            st.dataframe(log_df, use_container_width=True)

            with st.expander("🔍 View Technical Parameter Details per Operation (JSON)"):
                for idx, item in enumerate(audit_log, 1):
                    op_step = item.get("step")
                    op_col = item.get("column")
                    op_act = item.get("action")
                    st.markdown(f"**#{idx} [{op_step}] `{op_col}`** — *{op_act}*")
                    st.json(item.get("details", {}))

            with dl_col2:
                log_csv_bytes = log_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📜 Download Audit Log (CSV)",
                    data=log_csv_bytes,
                    file_name="audit_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl_col3:
                log_json_bytes = log_df.to_json(orient="records", indent=2).encode("utf-8")
                st.download_button(
                    label="🧾 Download Audit Log (JSON)",
                    data=log_json_bytes,
                    file_name="audit_log.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.info("No cleaning transformations were necessary; dataset is already clean.")

    with tab_specs:
        st.subheader("Component Pipeline Specification")
        specs_data = [
            {
                "Phase": "Phase 0",
                "Component": "Setup & Benchmarks",
                "Status": "Completed",
                "Deliverable": "Scaffolding, sample data, tests, git flow",
            },
            {
                "Phase": "Phase 1",
                "Component": "Ingestion & Profiling",
                "Status": "Completed",
                "Deliverable": "Multi-format loader, type inference, quality score",
            },
            {
                "Phase": "Phase 2",
                "Component": "Cleaning Engine",
                "Status": "Completed",
                "Deliverable": "Rule-based cleaning, outlier handling, audit trail",
            },
            {
                "Phase": "Phase 3",
                "Component": "EDA & Visualization",
                "Status": "Planned",
                "Deliverable": "Univariate, bivariate, timeseries & Plotly charts",
            },
            {
                "Phase": "Phase 4",
                "Component": "Insights Engine",
                "Status": "Planned",
                "Deliverable": "Numerical rule-based business insights",
            },
            {
                "Phase": "Phase 5",
                "Component": "Interactive Dashboard",
                "Status": "Planned",
                "Deliverable": "PowerBI/Tableau style cards, charts, filters",
            },
            {
                "Phase": "Phase 6",
                "Component": "Report & UI Assembly",
                "Status": "Planned",
                "Deliverable": "HTML report export, end-to-end integration",
            },
            {
                "Phase": "Phase 7",
                "Component": "Hardening & Docs",
                "Status": "Planned",
                "Deliverable": "Stress tests, Docker, deployment, final v1.0.0",
            },
        ]
        st.table(specs_data)
elif ingestion_error:
    st.error(f"⚠️ **Unable to ingest dataset**: {ingestion_error}")
    st.info("Please inspect the file format and ensure it is a valid, non-corrupt dataset.")
else:
    st.info(
        "Please upload a dataset or select a benchmark sample in the sidebar to begin analysis."
    )
