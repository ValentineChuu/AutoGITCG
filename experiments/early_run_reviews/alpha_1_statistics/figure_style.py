"""Shared matplotlib look for report figures: fixed categorical colors, recessive grid, PNG output."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL_COLOR = "#8a8984"
TEXT_COLOR = "#0b0b0b"
SECONDARY_TEXT_COLOR = "#52514e"

plt.rcParams.update({
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": SECONDARY_TEXT_COLOR, "axes.titlecolor": TEXT_COLOR,
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.color": SECONDARY_TEXT_COLOR, "ytick.color": SECONDARY_TEXT_COLOR,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 4, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})


def series_color(position: int) -> str:
    """Fixed order; callers keep each entity on the same position in every figure."""
    return SERIES_COLORS[position % len(SERIES_COLORS)]


def save_figure(figure, folder: Path, name: str) -> str:
    figure.tight_layout()
    figure.savefig(folder / name, dpi=120)
    plt.close(figure)
    return name
