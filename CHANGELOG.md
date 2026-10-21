# Changelog

All notable changes to the AutoAnalyst project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-21
### Added
- Automated statistical Exploratory Data Analysis (EDA) engine in `modules/eda.py` driven dynamically by inferred column types (excluding constant and ID-like columns).
- Univariate statistical profiling with mean, median, std, IQR, variance, min, max, skewness, kurtosis, and normality testing (Shapiro-Wilk / D'Agostino-Pearson).
- Distribution classification using standard skewness bands (|skew| < 0.5: approximately symmetric, 0.5–1: moderately skewed, > 1: highly skewed) and kurtosis descriptors (below -1: flat, above 1: heavy-tailed).
- Adaptive-bin histogram computation using Freedman-Diaconis with Sturges fallback and integer-aware minimum bin width (5 to 60 bins).
- Shared correlation matrix engine supporting Pearson and Spearman methods, with p-value computation, correlation direction/strength classification, text annotations inside heatmap cells, and optional lower-triangle rendering.
- Bivariate hexbin/scatter plots with OLS regression lines and category mean comparisons with one-way ANOVA F-tests.
- Categorical independence testing using Chi-Square test of independence with Cramér's V effect size.
- 2D Principal Component Analysis (PCA) projection engine and chart mapping with explained variance ratio annotations.
- Time-series decomposition engine with column-type-aware daily aggregation (sums for volumetric metrics; means for rates/intensities).
- Linear trend strength classification graded by R² (< 0.3: weak, 0.3–0.6: moderate, > 0.6: strong) with grammatical article agreement and slope units.
- Detrended multi-frequency seasonality detection (weekly 7-day and annual 365-day cycles) requiring local autocorrelation peaks and ANOVA on residuals, eliminating monthly false positives on trending series.
- Interactive Plotly chart engine (`modules/visualization.py`) producing colorblind-safe, publication-grade figures with computed plain-English captions.
- "📈 Exploratory Data Analysis" tab integrated into Streamlit application shell with interactive sub-tabs (Univariate, Correlations, Time-Series, and Visualizations Gallery).
- Outlier treatment updated to default to flagging without mutating series values, preserving clean dataset values and auditing correlation deltas.
- Unit and acceptance test suites in `tests/test_eda.py` and `tests/test_visualization.py` bringing the test suite to 107 passing tests.

## [0.2.0] - 2026-10-19
### Added
- Automated, non-destructive data cleaning engine (`modules/cleaning.py`) with 19-step provenance audit log covering every transformation applied.
- Deduplication step with logged duplicate indices (removes 140 duplicate rows on benchmark data).
- Automatic dropping of 100 % all-null structural defect columns (e.g. `cancellation_notes`).
- String-encoded currency parser (`₹`, `$`, `€`, comma separators, parenthesized negatives) coercing to `float64`.
- Mixed-format datetime parser standardising varied date strings to `datetime64[ns]`.
- Future-date guard: flags and sets dates after today to `NaT` (default); optional `cap` mode available via config.
- Text standardisation engine: trims whitespace, compresses internal spaces, canonicalises categories:
  - Region variants (east/East/EAST/…) → 4 canonical values; gender variants (M/Male/male/…) → 2.
  - Constant columns correctly report **0 cells changed** (previously over-counted).
- Negative-value anomaly correction for strictly-positive domains: sets values to missing then lets median imputation handle them (no `abs()` distortion); count logged as "set to missing".
- Small-integer column validation instead of IQR capping (e.g. `quantity < 1` → flagged as invalid).
- IQR outlier capping with non-negative lower bounds for revenue, quantity, and unit_price; profit may remain negative.
- Outlier capping percentage shown in audit log; columns where capping exceeds 5 % of rows are flagged.
- Missing-value imputation using median for numeric columns and mode for categoricals; median fill for `age` runs **after** negative values are set to missing.
- "🧹 Cleaning Engine" Streamlit tab: configuration toggles, before/after impact comparison cards, 20-row preview, CSV export, and full audit log table.
- 45-test suite in `tests/test_cleaning.py` (100 % pass rate) covering all cleaning steps, edge cases, and reconciliation checks.

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
