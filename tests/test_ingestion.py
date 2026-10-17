"""Unit tests for the data ingestion engine."""

import io
from pathlib import Path
import pytest

from modules.ingestion import IngestionError, get_excel_sheet_names, load_file

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_load_clean_csv() -> None:
    """Verify loading standard clean CSV file."""
    csv_path = SAMPLE_DATA_DIR / "clean_sales.csv"
    df, meta = load_file(csv_path)
    assert not df.empty
    assert meta["row_count"] == 1000
    assert meta["col_count"] == 14
    assert meta["delimiter"] == ","
    assert meta["file_type"] == "CSV"


def test_load_messy_csv() -> None:
    """Verify loading messy CSV file with deliberate anomalies."""
    csv_path = SAMPLE_DATA_DIR / "messy_sales.csv"
    df, meta = load_file(csv_path)
    assert meta["row_count"] == 5000
    assert "store_code" in df.columns
    assert "cancellation_notes" in df.columns


def test_load_excel_and_sheets() -> None:
    """Verify Excel parsing and sheet name introspection."""
    excel_path = SAMPLE_DATA_DIR / "sales.xlsx"
    sheets = get_excel_sheet_names(excel_path)
    assert len(sheets) >= 1

    df, meta = load_file(excel_path)
    assert meta["file_type"] == "XLSX"
    assert meta["selected_sheet"] == sheets[0]
    assert len(df) == 100


def test_load_json_and_parquet() -> None:
    """Verify JSON and Parquet formats load identically formatted data."""
    json_path = SAMPLE_DATA_DIR / "sales.json"
    df_json, meta_json = load_file(json_path)
    assert meta_json["file_type"] == "JSON"
    assert len(df_json) == 100

    parquet_path = SAMPLE_DATA_DIR / "sales.parquet"
    df_parquet, meta_parquet = load_file(parquet_path)
    assert meta_parquet["file_type"] == "PARQUET"
    assert len(df_parquet) == 100


def test_delimiter_auto_detection() -> None:
    """Verify delimiter sniffing across semicolon and tab separated files."""
    # Semicolon CSV
    tsv_content = "id;name;score\n1;Alice;95\n2;Bob;88\n"
    buffer = io.BytesIO(tsv_content.encode("utf-8"))
    df, meta = load_file(buffer, filename="test.csv")
    assert meta["delimiter"] == ";"
    assert list(df.columns) == ["id", "name", "score"]
    assert len(df) == 2

    # Tab-separated TSV
    tab_content = "id\tmetric\tvalue\n101\tA\t10.5\n102\tB\t20.2\n"
    tab_buffer = io.BytesIO(tab_content.encode("utf-8"))
    df_tab, meta_tab = load_file(tab_buffer, filename="test.tsv")
    assert meta_tab["delimiter"] == "\t"
    assert len(df_tab) == 2


def test_empty_file_handling() -> None:
    """Verify empty file raises IngestionError with friendly message."""
    empty_buffer = io.BytesIO(b"")
    with pytest.raises(IngestionError, match="empty"):
        load_file(empty_buffer, filename="empty.csv")


def test_oversized_file_handling() -> None:
    """Verify files exceeding size limits raise IngestionError."""
    sample_content = b"a,b,c\n1,2,3\n" * 100
    buffer = io.BytesIO(sample_content)
    with pytest.raises(IngestionError, match="exceeds maximum allowed threshold"):
        load_file(buffer, max_size_mb=0.0001, filename="data.csv")


def test_corrupt_file_handling() -> None:
    """Verify corrupt binary files raise clear IngestionError."""
    corrupt_buffer = io.BytesIO(b"PK\x03\x04CorruptedNonZipDataGarbage")
    with pytest.raises(IngestionError):
        load_file(corrupt_buffer, filename="corrupt.xlsx")


def test_immutable_copy() -> None:
    """Verify load_file returns an independent DataFrame copy."""
    csv_path = SAMPLE_DATA_DIR / "tiny.csv"
    df, _ = load_file(csv_path)
    original_val = df.iloc[0, 0]
    df.iloc[0, 0] = "MODIFIED"

    # Reload and assert original file content was not affected
    df2, _ = load_file(csv_path)
    assert df2.iloc[0, 0] == original_val
