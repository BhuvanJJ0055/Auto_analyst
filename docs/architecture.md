# AutoAnalyst Architecture & Pipeline Design

AutoAnalyst is an autonomous, non-destructive data analytics application designed to convert raw datasets (CSV, Excel, JSON, Parquet) into actionable business intelligence.

## Pipeline Flow

```mermaid
flowchart TD
    A[Input Data: CSV / XLSX / JSON / Parquet] --> B[Phase 1: Ingestion Engine]
    B --> C[Phase 1: Profiling & Quality Scoring]
    C --> D[Phase 2: Automated Cleaning Engine]
    D --> E[Phase 3: Exploratory Data Analysis EDA]
    E --> F[Phase 3: Automated Visualization Engine]
    E --> G[Phase 4: Insights & Recommendations Engine]
    F --> H[Phase 5: Interactive BI Dashboard]
    G --> H
    H --> I[Phase 6: HTML & PDF Report Generation]
    D -. Audit Trail .-> I
```

## Core Design Principles

1. **Non-Destructive Guarantee**: The original uploaded dataset is never modified in place. All transformations yield immutable copies.
2. **Comprehensive Audit Trail**: Every data cleaning step is recorded with the exact rule, column, rows affected, and before/after distribution.
3. **Zero-Hardcoding**: Dynamic type inference ensures compatibility with arbitrary tabular structures.
4. **Local Execution**: No mandatory external cloud services or databases; completely privacy-preserving.
