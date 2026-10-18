# Changelog

All notable changes to the AutoAnalyst project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Automated, non-destructive data cleaning engine in `modules/cleaning.py` with full provenance audit logging.
- Deduplication with logged duplicate indices (drops 140 duplicate rows on benchmark data).
- Automatic dropping of 100% all-null structural defect columns (`cancellation_notes`).
- String-encoded currency parsing (`₹`, `$`, `€`, comma separators, parenthesized negatives) to native `float64`.
- Mixed datetime parser standardizing varied date formats into `datetime64[ns]`.
- Text standardization engine trimming whitespace, compressing internal spaces, and canonicalizing categories:
  - Merges region variants (east/East/EAST/spaces) down to exactly 4 canonical regions (`East`, `North`, `South`, `West`).
  - Merges gender variants (M/Male/male/F/Female/female) down to exactly 2 binary genders (`Female`, `Male`).
- Negative value anomaly correction for strictly positive domains (e.g., negative ages).
- Numerical outlier treatment with IQR boundary clipping (`1.5 * IQR`) neutralizing extreme revenue spikes.
- Missing value imputation using median for skewed numeric columns and mode for categoricals.
- Dedicated "🧹 Cleaning Engine" Streamlit UI tab with interactive configuration toggles, before/after impact comparison cards, 20-row preview, CSV export, and step-by-step audit log table.
- Comprehensive unit and acceptance test suite in `tests/test_cleaning.py` expanding test coverage to 32 tests (100% pass rate).

## [0.1.0] - 2026-10-18
### Added
- Multi-format ingestion engine in `modules/ingestion.py` supporting CSV, TSV, Excel (`.xlsx`, `.xls`), JSON, and Apache Parquet.
- Character encoding auto-detection via `chardet` with intelligent fallbacks.
- Delimiter sniffer for automated separator detection (commas, tabs, semicolons, pipes).
- Multi-sheet Excel inspector (`get_excel_sheet_names`) and selective sheet loader.
- File size guard (200MB default limit) and empty file validation with user-friendly errors.
- Dynamic semantic type inference engine in `modules/profiling.py` categorizing columns into `numeric`, `categorical`, `datetime`, `boolean`, `id_like`, `constant`, and `text`.
- Currency string detection and mixed date format identifier.
- Transparent 0–100 Data Quality Score algorithm factoring completeness, uniqueness, validity, and consistency.
- Streamlit UI integration featuring dynamic file uploader, benchmark selector, Data Quality Score card, 20-row preview, and detailed column profiling tables.
- Comprehensive unit tests in `tests/test_ingestion.py` and `tests/test_profiling.py` bringing test suite to 21 passing tests with explicit quality score threshold assertions.
- UI screenshot documentation in `docs/profiling_overview.png`, `docs/clean_sales_scorecard.png`, `docs/messy_sales_scorecard.png`, and `docs/messy_sales_profiling.png`.

## [0.0.0] - 2026-10-17
### Added
- Git repository initialization with `main` branch and `.gitignore` rules.
- MIT License and `.env.example` template with secure configuration defaults.
- Modular architecture package layout across ingestion, profiling, cleaning, EDA, visualization, insights, dashboard, and reporting.
- Pinned `requirements.txt` covering Streamlit, Pandas, NumPy, SciPy, Statsmodels, Scikit-learn, Plotly, Pytest, and file format libraries.
- Synthetic benchmark generator (`scripts/make_sample_data.py`) creating clean, messy, timeseries, tiny, large, Excel, and JSON test datasets.
- Interactive Streamlit application shell (`app.py`) featuring pipeline step visualizer and sample dataset inspector.
- Automated unit test suite with 100% pass rate validating utils, datasets, and app compilation.
- Project documentation including architecture diagrams (`docs/architecture.md`), `README.md`, and `PROGRESS.md`.
