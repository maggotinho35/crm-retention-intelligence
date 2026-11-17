"""Shared plotting style: one palette, takeaway-style titles, source notes, colour-blind-safe colours."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAVY = "#1F2A5C"
GOLD = "#C49A3A"
RED = "#B94E42"
BLUE = "#4C78B5"
SKY = "#8FB3DE"
GREEN = "#2E8B6E"
GREY = "#98A2B3"
INK = "#1b2a3a"
MUTED = "#6b7785"
GRID = "#e3e7ec"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"], "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def finish(fig, path, note=None):
    """Save a figure with an optional grey source/assumption note underneath."""
    if note:
        fig.text(0.0, -0.02, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
