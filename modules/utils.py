"""Utility functions and common helpers for AutoAnalyst.

Provides formatting, color palettes, and memory estimation helpers.
"""

from typing import Dict


def get_color_palette() -> Dict[str, str]:
    """Return a colorblind-safe, high-contrast palette for visualizations."""
    return {
        "primary": "#2563EB",
        "secondary": "#0D9488",
        "accent": "#F59E0B",
        "danger": "#EF4444",
        "neutral_dark": "#1F2937",
        "neutral_light": "#F3F4F6",
        "background": "#FFFFFF",
    }


def format_bytes(num_bytes: int) -> str:
    """Format byte count into human-readable representation."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"
