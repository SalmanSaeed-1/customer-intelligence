"""Shared helpers for figures and metric tables (used by every notebook).

    set_plot_style() -> one consistent look for all figures (fonts, sizes, axes)
    save_fig()       -> save a figure to reports/figures as PNG (300 dpi) + PDF
    save_table()     -> save a table to reports/results as CSV

More helpers (metric tables, ROC/PR plots) will be added as the modules are built.
"""
import matplotlib.pyplot as plt

from src import config


def set_plot_style():
    """Apply the project figure style (CLAUDE.md Section 13) to every plot that follows.

    Sans font at 7-9 pt, dark text, light grey axes, no top/right frame lines,
    and a faint horizontal grid drawn behind the data.
    """
    ink = config.COLORS["ink"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "text.color": ink,
        "axes.labelcolor": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "axes.edgecolor": "#b3b3b3",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,     # grid behind bars/lines, not on top
        "figure.dpi": 120,          # on-screen size in the notebook
        "savefig.dpi": 300,
        "pdf.fonttype": 42,         # embed real (editable) fonts in PDFs
    })


def save_fig(fig, name):
    """Save `fig` as reports/figures/<name>.png (300 dpi) and <name>.pdf."""
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def save_table(df, name, index=False):
    """Save a DataFrame as reports/results/<name>.csv and return the file path."""
    path = config.RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    return path
