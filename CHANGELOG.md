# Changelog

All notable changes to the AutoAnalyst project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
