"""AutoAnalyst - Automated Data Analyst Web Application.

Phase 1: Ingestion and Profiling UI.
Provides dataset loading, file validation, dynamic semantic type inference,
and transparent data quality scoring.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st

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
    .grade-D { background: #4D1B0C; color: #DB6D28; border: 1px solid #BD561D; }
    .grade-F { background: #490202; color: #F85149; border: 1px solid #DA3633; }
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
        <span class="pipeline-step active">⚡ 2. Profiling</span>
        <span class="pipeline-step">3. Cleaning</span>
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


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("**Version:** `v0.1.0 (Phase 1: Ingestion & Profiling)`")
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
                st.error(f"⚠️ Ingestion Error: {err}")
    else:
        sample_files = sorted([f.name for f in sample_dir.glob("*.*") if f.name != "large.csv"])
        selected_sample = st.selectbox(
            "Select Benchmark Dataset",
            sample_files,
            index=sample_files.index("clean_sales.csv") if "clean_sales.csv" in sample_files else 0,
        )
        if selected_sample:
            sample_path = sample_dir / selected_sample
            with open(sample_path, "rb") as sf:
                sample_bytes = sf.read()
            with st.spinner(f"Loading {selected_sample}..."):
                df_loaded, ingestion_meta = cached_load_file(sample_bytes, filename=selected_sample)

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
            <div class="card" style="padding: 1rem 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.85rem; color: #8B949E; text-transform: uppercase;
                        font-weight: 600;">Data Quality Score</div>
                        <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">
                            {score_val} <span style="font-size: 1rem; color: #8B949E;">/ 100</span>
                        </div>
                    </div>
                    <span class="score-badge grade-{grade}">{grade}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col2:
        st.metric("Total Rows", f"{profile['total_rows']:,}")
    with top_col3:
        st.metric("Total Columns", f"{profile['total_columns']:,}")
    with top_col4:
        st.metric(
            "Duplicate Rows", f"{profile['duplicate_rows']:,}", f"{profile['duplicate_pct']}%"
        )
    with top_col5:
        total_cells_count = profile["total_rows"] * profile["total_columns"]
        missing_rate_pct = round((profile["total_missing_cells"] / total_cells_count) * 100, 2)
        st.metric(
            "Missing Cells",
            f"{profile['total_missing_cells']:,}",
            f"{missing_rate_pct}%",
            delta_color="inverse",
        )

    # Tabs Section
    tab_overview, tab_preview, tab_profiling, tab_specs = st.tabs(
        [
            "🚀 Dataset Overview",
            "📂 Data Preview",
            "🔍 Column Profiling",
            "📋 Technical Architecture",
        ]
    )

    with tab_overview:
        st.subheader("Data Health & Quality Breakdown")

        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            st.metric(
                "Completeness",
                f"{quality['completeness_score']} / 30",
                f"{quality['completeness_ratio']}% clean",
            )
        with q_col2:
            st.metric(
                "Uniqueness",
                f"{quality['uniqueness_score']} / 25",
                f"{quality['uniqueness_ratio']}% unique",
            )
        with q_col3:
            st.metric(
                "Validity",
                f"{quality['validity_score']} / 25",
                f"{quality['validity_ratio']}% valid",
            )
        with q_col4:
            st.metric(
                "Consistency",
                f"{quality['consistency_score']} / 20",
                f"{quality['consistency_ratio']}% consistent",
            )

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
                "Status": "Next",
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
else:
    st.info(
        "Please upload a dataset or select a benchmark sample in the sidebar to begin analysis."
    )
