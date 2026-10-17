# Changelog

All notable changes to the AutoAnalyst project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
