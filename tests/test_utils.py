"""Unit tests for utility helpers."""

from modules.utils import format_bytes, get_color_palette


def test_format_bytes() -> None:
    """Verify byte formatting handles various orders of magnitude."""
    assert format_bytes(500) == "500.00 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1024 * 1024) == "1.00 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"


def test_color_palette() -> None:
    """Verify color palette includes required semantic tokens."""
    palette = get_color_palette()
    assert isinstance(palette, dict)
    for token in ["primary", "secondary", "accent", "danger", "neutral_dark", "neutral_light"]:
        assert token in palette
        assert palette[token].startswith("#")
