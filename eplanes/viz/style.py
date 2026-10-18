"""Figure style, colours and caption registry (dataviz skill: validated categorical order, light surface).

Categorical slots 1-3 are validated for all-pairs use; a fourth aircraft is distinguished by
colour (violet) AND marker/linestyle, and every multi-series figure has a legend.
Colour follows the entity (aircraft / battery), never its rank.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET, MAGENTA, RED, YELLOW, GREEN = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6",
                                                         "#e87ba4", "#e34948", "#eda100", "#008300")
AIRCRAFT_STYLE = {  # entity -> (colour, marker)
    "C-GAUW": (BLUE, "o"), "C-GMUW": (ORANGE, "s"),
    "VSW1280102": (AQUA, "^"), "VSW1280107": (VIOLET, "D"),
}
BATTERY_STYLE = {1: (BLUE, "-"), 2: (ORANGE, "--")}
SEQ = ["#e8f0fb", "#b9d3f3", "#7fb0ea", "#2a78d6", "#1b4f94", "#0f2f5a"]  # single-hue sequential


def aircraft_color(label: str, i: int = 0):
    return AIRCRAFT_STYLE.get(label, ([BLUE, ORANGE, AQUA, VIOLET, MAGENTA, RED][i % 6], "o"))


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 1.6, "figure.dpi": 110, "savefig.dpi": 300,
    })


apply()


class FigureSink:
    """Saves PNG+PDF and records a caption for every figure."""

    def __init__(self, out_dir: Path, prefix: str = ""):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.captions: dict[str, str] = {}
        self.reg = self.dir / "figure_captions.json"
        if self.reg.exists():
            self.captions = json.loads(self.reg.read_text())

    def save(self, fig, name: str, caption: str):
        name = f"{self.prefix}{name}"
        fig.savefig(self.dir / f"{name}.png", bbox_inches="tight")
        fig.savefig(self.dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        self.captions[name] = caption
        self.reg.write_text(json.dumps(self.captions, indent=1))
        return name
