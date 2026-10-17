"""Data Ingestion Module for AutoAnalyst.

Provides multi-format data ingestion (CSV, Excel, JSON, Parquet) with automatic
character encoding detection, delimiter sniffing, header detection, size validation,
and defensive error handling.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chardet
import openpyxl
import pandas as pd

from modules.utils import format_bytes


class IngestionError(Exception):
    """Custom exception raised when file ingestion fails."""

    pass


def get_excel_sheet_names(file_path_or_buffer: Any) -> List[str]:
    """Retrieve sheet names from an Excel file without loading data into memory.

    Args:
        file_path_or_buffer: Path to Excel workbook or file-like binary buffer.

    Returns:
        List of sheet names.
    """
    try:
        if isinstance(file_path_or_buffer, bytes):
            file_path_or_buffer = io.BytesIO(file_path_or_buffer)

        if isinstance(file_path_or_buffer, (str, Path)):
            wb = openpyxl.load_workbook(file_path_or_buffer, read_only=True)
            return wb.sheetnames
        else:
            # Buffer object
            pos = file_path_or_buffer.tell() if hasattr(file_path_or_buffer, "tell") else 0
            wb = openpyxl.load_workbook(file_path_or_buffer, read_only=True)
            sheetnames = wb.sheetnames
            if hasattr(file_path_or_buffer, "seek"):
                file_path_or_buffer.seek(pos)
            return sheetnames
    except Exception as exc:
        raise IngestionError(f"Could not read Excel workbook sheets: {str(exc)}") from exc


def _detect_encoding_and_delimiter(
    sample_bytes: bytes,
) -> Tuple[str, str]:
    """Detect text encoding and delimiter from raw file bytes.

    Args:
        sample_bytes: Initial byte slice of the file.

    Returns:
        Tuple of (encoding, delimiter).
    """
    detect_res = chardet.detect(sample_bytes)
    encoding = detect_res.get("encoding") or "utf-8"

    # Normalize common chardet aliases
    if encoding.lower() in ["ascii", "iso-8859-1", "windows-1252"]:
        # Verify utf-8 compatibility first
        try:
            sample_bytes.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            pass

    try:
        sample_text = sample_bytes.decode(encoding, errors="ignore")
    except Exception:
        sample_text = sample_bytes.decode("utf-8", errors="ignore")
        encoding = "utf-8"

    # Sniff delimiter across candidate delimiters
    candidates = [",", ";", "\t", "|"]
    first_lines = [line.strip() for line in sample_text.splitlines() if line.strip()][:5]

    if not first_lines:
        return encoding, ","

    delimiter_scores: Dict[str, float] = {}
    for cand in candidates:
        counts = [line.count(cand) for line in first_lines]
        if counts and counts[0] > 0 and all(c == counts[0] for c in counts):
            delimiter_scores[cand] = float(counts[0]) + 10.0
        elif counts:
            delimiter_scores[cand] = float(sum(counts)) / len(counts)
        else:
            delimiter_scores[cand] = 0.0

    best_delimiter = max(delimiter_scores.items(), key=lambda x: x[1])[0]
    if delimiter_scores[best_delimiter] == 0:
        best_delimiter = ","

    return encoding, best_delimiter


def load_file(
    file_path_or_buffer: Any,
    sheet_name: Optional[Union[str, int]] = None,
    max_size_mb: float = 200.0,
    filename: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load an input file and return an immutable DataFrame copy with metadata.

    Supports CSV, TSV, XLSX, XLS, JSON, and Parquet.

    Args:
        file_path_or_buffer: File system path or Streamlit UploadedFile / BytesIO object.
        sheet_name: Optional Excel sheet name to load.
        max_size_mb: Maximum allowed file size in megabytes (default 200 MB).
        filename: Optional explicit filename if loading from a generic buffer.

    Returns:
        Tuple of (loaded DataFrame copy, metadata dictionary).

    Raises:
        IngestionError: If file is missing, empty, exceeds max size, corrupt, or unsupported.
    """
    file_bytes: bytes
    resolved_name: str

    if isinstance(file_path_or_buffer, (str, Path)):
        path = Path(file_path_or_buffer)
        if not path.exists():
            raise IngestionError(f"File not found: {path}")
        file_size = path.stat().st_size
        resolved_name = filename or path.name
        with open(path, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_path_or_buffer, bytes):
        file_bytes = file_path_or_buffer
        file_size = len(file_bytes)
        resolved_name = filename or "dataset"
    else:
        # Buffer / UploadedFile
        if hasattr(file_path_or_buffer, "name"):
            resolved_name = filename or getattr(file_path_or_buffer, "name")
        else:
            resolved_name = filename or "uploaded_data"

        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        file_bytes = file_path_or_buffer.read()
        file_size = len(file_bytes)

    # 1. Size Validation
    if file_size == 0:
        raise IngestionError("The uploaded file is empty (0 bytes). Please upload a valid dataset.")

    max_bytes = int(max_size_mb * 1024 * 1024)
    if file_size > max_bytes:
        raise IngestionError(
            f"File size ({format_bytes(file_size)}) exceeds maximum allowed threshold of "
            f"{max_size_mb:.1f} MB."
        )

    # 2. Extension determination
    ext = Path(resolved_name).suffix.lower()
    if not ext:
        # Fallback inspection by content header
        if file_bytes.startswith(b"PAR1"):
            ext = ".parquet"
        elif file_bytes.startswith(b"PK\x03\x04"):
            ext = ".xlsx"
        elif file_bytes.strip().startswith((b"{", b"[")):
            ext = ".json"
        else:
            ext = ".csv"

    metadata: Dict[str, Any] = {
        "filename": resolved_name,
        "file_size_bytes": file_size,
        "file_size_formatted": format_bytes(file_size),
        "file_type": ext.lstrip(".").upper(),
        "encoding": "binary",
        "delimiter": None,
        "sheet_names": [],
        "selected_sheet": None,
        "warnings": [],
    }

    df: pd.DataFrame

    # 3. Format-specific parsing
    try:
        if ext in [".csv", ".tsv", ".txt"]:
            sample_slice = file_bytes[:100000]
            encoding, delimiter = _detect_encoding_and_delimiter(sample_slice)
            metadata["encoding"] = encoding
            metadata["delimiter"] = delimiter

            buffer = io.BytesIO(file_bytes)
            # Inspect first 20 rows to confirm headers
            preview = pd.read_csv(
                buffer,
                sep=delimiter,
                encoding=encoding,
                nrows=20,
                on_bad_lines="skip",
            )
            buffer.seek(0)

            # Check if headers look like data (all numeric or unnamed)
            has_numeric_headers = all(
                isinstance(c, (int, float)) or str(c).isdigit() for c in preview.columns
            )
            header_setting = None if has_numeric_headers else 0

            df = pd.read_csv(
                buffer,
                sep=delimiter,
                encoding=encoding,
                header=header_setting,
                on_bad_lines="skip",
            )
            if header_setting is None:
                metadata["warnings"].append(
                    "Header row not detected; generated numeric column names."
                )

        elif ext in [".xlsx", ".xls"]:
            buffer = io.BytesIO(file_bytes)
            sheets = get_excel_sheet_names(buffer)
            metadata["sheet_names"] = sheets
            buffer.seek(0)

            target_sheet = sheet_name if sheet_name in sheets else sheets[0]
            metadata["selected_sheet"] = target_sheet

            df = pd.read_excel(buffer, sheet_name=target_sheet, engine="openpyxl")

        elif ext == ".parquet":
            buffer = io.BytesIO(file_bytes)
            df = pd.read_parquet(buffer, engine="pyarrow")

        elif ext == ".json":
            buffer = io.BytesIO(file_bytes)
            try:
                # First try standard records / table orientation
                df = pd.read_json(buffer)
            except ValueError:
                buffer.seek(0)
                # Fallback to lines=True
                df = pd.read_json(buffer, lines=True)

        else:
            raise IngestionError(
                f"Unsupported file format '{ext}'. AutoAnalyst supports CSV, XLSX, XLS, "
                "JSON, and Parquet."
            )

    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Failed to parse {metadata['file_type']} file: {str(exc)}") from exc

    # 4. DataFrame Sanity Check
    if df.empty:
        raise IngestionError(
            f"The parsed {metadata['file_type']} dataset contains zero rows or columns."
        )

    # Ensure column names are strings
    df.columns = [str(c) for c in df.columns]

    metadata["row_count"] = int(df.shape[0])
    metadata["col_count"] = int(df.shape[1])

    # Rule 1: Always return a fresh copy, original untouched
    return df.copy(), metadata
