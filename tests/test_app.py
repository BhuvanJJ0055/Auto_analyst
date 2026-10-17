"""Smoke tests for the Streamlit application."""

from pathlib import Path
import py_compile


def test_app_compiles() -> None:
    """Verify that app.py compiles cleanly with zero syntax errors."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists(), "app.py must exist at project root"

    # Compile python bytecode to verify syntax correctness
    compiled_path = py_compile.compile(str(app_path), doraise=True)
    assert compiled_path is not None
