"""Shared chart style for the report figures (static PNG, light background)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# categorical slots in fixed order (validated reference palette): blue, orange, aqua, yellow
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# one-hue ordinal ramp for ordered categories such as return buckets
BLUE_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]
NEGATIVE = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.labelsize": 10,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False, "legend.fontsize": 9, "font.family": "DejaVu Sans",
    "lines.linewidth": 2, "lines.markersize": 6,
})


def save(fig, name: str, rect=(0, 0, 1, 1)) -> Path:
    """rect leaves room at the figure edges, e.g. for a footnote under the axes."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def pct(ax, axis="y"):
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    # whole-percent steps (1, 2, 5, 10, ...) so rounded tick labels never repeat
    a.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    a.set_major_formatter(PercentFormatter(1.0, decimals=0))


def label_bars(ax, bars, fmt="{:.0%}", inside=False):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -11),
                    textcoords="offset points", ha="center", fontsize=8, color=TEXT_2)
