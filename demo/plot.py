"""Draw the demo's figure: the peak of infection against both swept parameters.

Reads the table and the manifest written by ``campaign run demo/sir.yaml`` and
writes ``docs/peak_infection.png``:

    python demo/plot.py [--results results/sir] [--out docs/peak_infection.png]

One sequential hue, light to dark, for one magnitude; a scale legend; cells
separated by the background rather than by borders; and labels only where
they carry the point. The staircase line is read off the results, not
computed separately: a cell is right of it when the run's own peak came
before the day of the cut.

Deterministic: fixed size, fixed colours, and no timestamp or version string
in the PNG, so redrawing from the same results reproduces the file byte for
byte.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sir"
FIGURE = ROOT / "docs" / "peak_infection.png"

# A single blue ramp, steps 100 to 700, light meaning little.
RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


@dataclass(frozen=True)
class Grid:
    """The campaign's results laid out as rows of R0 and columns of cut day."""

    r0: NDArray[np.float64]
    cut_days: NDArray[np.float64]
    peak_pct: NDArray[np.float64]  # NaN where a run did not succeed
    after_peak: NDArray[np.bool_]  # the cut came after the run's own peak
    still_rising: NDArray[np.bool_]  # no peak within the horizon: a lower bound
    reduction: float
    t_end: float
    runs: int


def read(results: Path) -> Grid:
    """Lay the results table out as a grid."""
    with (results / "results.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    constants = json.loads((results / "manifest.json").read_text(encoding="utf-8"))[
        "constants"
    ]
    gamma = float(constants["gamma"])

    betas = sorted({float(r["beta"]) for r in rows})
    cuts = sorted({float(r["t_intervention"]) for r in rows})
    shape = (len(betas), len(cuts))
    peak = np.full(shape, np.nan)
    after = np.zeros(shape, dtype=bool)
    rising = np.zeros(shape, dtype=bool)
    for r in rows:
        i, j = betas.index(float(r["beta"])), cuts.index(float(r["t_intervention"]))
        if r["status"] != "succeeded":
            continue
        peak[i, j] = 100.0 * float(r["peak_fraction"])
        after[i, j] = float(r["peak_day"]) < float(r["t_intervention"])
        rising[i, j] = float(r["peaked"]) == 0.0

    return Grid(
        r0=np.array(betas) / gamma,
        cut_days=np.array(cuts),
        peak_pct=peak,
        after_peak=after,
        still_rising=rising,
        reduction=float(constants["reduction"]),
        t_end=float(constants["t_end"]),
        runs=len(rows),
    )


def _label(ax: Axes, grid: Grid, i: int, j: int, *, lower_bound: bool = False) -> None:
    """Write cell (i, j)'s value in it, in ink that reads against the cell."""
    value = float(grid.peak_pct[i, j])
    text = f"≥{value:.1f}%" if lower_bound else f"{value:.0f}%"
    dark_cell = value > 0.55 * float(np.nanmax(grid.peak_pct))
    ax.text(
        j + 0.5,
        i + 0.5,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white" if dark_cell else INK,
    )


def draw(grid: Grid, path: Path) -> Path:
    """Write the figure to ``path`` and return it."""
    n_rows, n_cols = grid.peak_pct.shape
    vmax = float(np.nanmax(grid.peak_pct))
    # Cells whose run did not succeed are left blank, in the background colour.
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP).with_extremes(bad=SURFACE)

    fig, ax = plt.subplots(figsize=(10.0, 5.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(grid.peak_pct),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        edgecolors=SURFACE,
        linewidth=1.0,  # about 2 px at 150 dpi: cells separated by the background
    )

    # Where the cut came after the peak it changed nothing. The boundary is
    # the first such cell in each row that has one, drawn as a staircase and
    # closed along its lowest row; rows with no such cell get no line at all.
    rows_with_late = [i for i in range(n_rows) if grid.after_peak[i].any()]
    if rows_with_late:
        xs, ys = [float(n_cols)], [float(rows_with_late[0])]
        for i in rows_with_late:
            x = float(np.flatnonzero(grid.after_peak[i])[0])
            xs += [x, x]
            ys += [float(i), float(i + 1)]
        ax.plot(xs, ys, color=INK, linewidth=2.0, solid_capstyle="butt")

    # The cut takes the reproduction number to R0 (1 - reduction). Rows where
    # that stays above 1 can only be slowed; the line sits between the last
    # row that can be stopped and the first that cannot.
    threshold = 1.0 / (1.0 - grid.reduction)
    boundary = float(np.searchsorted(grid.r0, threshold, side="right"))
    ax.axhline(boundary, color=INK_SECONDARY, linewidth=1.0)
    ax.text(
        0.1,
        boundary - 0.1,
        # A real multiplication sign, on purpose: this is text on a figure.
        f"R0 × {1 - grid.reduction:g} = 1: below this line an early cut ends the epidemic",  # noqa: RUF001
        fontsize=8.5,
        color=INK_SECONDARY,
        va="top",
    )

    # Selective labels: the two ends of the top row, which are the headline,
    # and the one value that is only a lower bound.
    top = n_rows - 1
    _label(ax, grid, top, 0)
    _label(ax, grid, top, n_cols - 1)
    for i, j in zip(*np.nonzero(grid.still_rising), strict=True):
        _label(ax, grid, int(i), int(j), lower_bound=True)

    ax.set_xticks(np.arange(n_cols) + 0.5, [f"{d:g}" for d in grid.cut_days])
    ax.set_yticks(np.arange(n_rows) + 0.5, [f"{r:g}" for r in grid.r0])
    ax.set_xlabel(
        f"day contacts are cut by {100 * grid.reduction:.0f}%",
        color=INK_SECONDARY,
        fontsize=10,
    )
    ax.set_ylabel("R0, before the cut", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9.5, length=0)
    ax.tick_params(axis="both", labelcolor=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label(
        "largest share infected at once (%)", color=INK_SECONDARY, fontsize=9.5
    )
    colorbar.ax.tick_params(
        colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9, length=0
    )
    colorbar.outline.set_visible(False)

    fig.suptitle(
        f"At or below R0 = {threshold:g}, a {100 * grid.reduction:.0f}% cut in contacts "
        "ends the epidemic; above it, an early cut at least halves the peak",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.06,
        0.905,
        f"Peak of infection in {grid.runs} SIR runs over {grid.t_end:g} days. Right of the "
        "black line the cut came after the peak and changed nothing.",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.11)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return path


def main() -> None:
    """Draw the figure from the demo's results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=FIGURE)
    args = parser.parse_args()
    print(f"wrote {draw(read(args.results), args.out)}")


if __name__ == "__main__":
    main()
