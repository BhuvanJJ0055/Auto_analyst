"""AutoAnalyst - Automated Data Analyst Web Application.

Phase 0: Application Shell and Benchmark Explorer.
Provides the main application layout, pipeline architecture overview,
and interactive sample data preview.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

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
        color: #1E293B;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .pipeline-step {
        background: #F1F5F9;
        color: #334155;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #CBD5E1;
    }
    .pipeline-step.active {
        background: #2563EB;
        color: #FFFFFF;
        border-color: #1D4ED8;
    }
    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-header">📊 AutoAnalyst</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Autonomous end-to-end data analysis engine: ingestion, profiling, automated cleaning, "
    "EDA, interactive dashboards, plain-English insights, and publication reports."
    "</div>",
    unsafe_allow_html=True,
)

# Pipeline Flow Visualizer
st.markdown(
    """
    <div class="pipeline-container">
        <span class="pipeline-step active">1. Ingestion</span>
        <span class="pipeline-step">2. Profiling</span>
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

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("**Version:** `v0.0.0 (Phase 0 Setup)`")
    st.markdown("**Status:** Ready for Phase 1 Ingestion & Profiling")
    st.divider()

    st.subheader("Sample Datasets")
    sample_dir = Path(__file__).resolve().parent / "sample_data"
    sample_files = sorted([f.name for f in sample_dir.glob("*.*") if f.name != "large.csv"])
    selected_sample = st.selectbox("Explore Sample Benchmark Data", sample_files, index=0 if sample_files else None)

# Main Content Tabs
tab_overview, tab_preview, tab_specs = st.tabs(["🚀 Project Overview", "📂 Sample Data Preview", "📋 Technical Architecture"])

with tab_overview:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card">
                <h4>🛡️ Non-Destructive Engine</h4>
                <p>The original uploaded dataset is never modified. All transformations produce explicit copies accompanied by a forensic audit log.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card">
                <h4>🤖 Zero-Hardcoding EDA</h4>
                <p>Fully dynamic semantic type inference adapts to any tabular dataset (CSV, Excel, JSON, Parquet) without predetermined schemas.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card">
                <h4>⚡ Interactive Dashboards</h4>
                <p>Sub-second filtering across 100k+ rows with KPI metrics, drill-downs, calculated fields, and offline HTML export.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("System setup is complete. To begin analyzing real-world datasets, proceed with Phase 1: Ingestion & Profiling.")

with tab_preview:
    if selected_sample:
        file_path = sample_dir / selected_sample
        st.subheader(f"Dataset: `{selected_sample}`")

        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_path.suffix == ".json":
            df = pd.read_json(file_path)
        else:
            df = pd.DataFrame()

        if not df.empty:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Rows", f"{len(df):,}")
            m_col2.metric("Columns", f"{len(df.columns):,}")
            m_col3.metric("Duplicates", f"{df.duplicated().sum():,}")
            m_col4.metric("Total Nulls", f"{df.isna().sum().sum():,}")

            st.dataframe(df.head(20), use_container_width=True)
            with st.expander("Column Data Types"):
                st.write(df.dtypes.astype(str))
    else:
        st.warning("No sample datasets found. Run `python scripts/make_sample_data.py` to generate.")

with tab_specs:
    st.subheader("Component Pipeline Specification")
    specs_data = [
        {"Phase": "Phase 0", "Component": "Setup & Benchmarks", "Status": "Completed", "Deliverable": "Scaffolding, sample data, tests, git flow"},
        {"Phase": "Phase 1", "Component": "Ingestion & Profiling", "Status": "Next", "Deliverable": "Multi-format loader, type inference, quality score"},
        {"Phase": "Phase 2", "Component": "Cleaning Engine", "Status": "Planned", "Deliverable": "Rule-based cleaning, outlier handling, audit trail"},
        {"Phase": "Phase 3", "Component": "EDA & Visualization", "Status": "Planned", "Deliverable": "Univariate, bivariate, timeseries & Plotly charts"},
        {"Phase": "Phase 4", "Component": "Insights Engine", "Status": "Planned", "Deliverable": "Numerical rule-based business insights"},
        {"Phase": "Phase 5", "Component": "Interactive Dashboard", "Status": "Planned", "Deliverable": "PowerBI/Tableau style cards, charts, filters"},
        {"Phase": "Phase 6", "Component": "Report & UI Assembly", "Status": "Planned", "Deliverable": "HTML report export, end-to-end integration"},
        {"Phase": "Phase 7", "Component": "Hardening & Docs", "Status": "Planned", "Deliverable": "Stress tests, Docker, deployment, final v1.0.0"},
    ]
    st.table(specs_data)
