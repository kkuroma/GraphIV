"""
The look every card figure shares, taken from ModernLayoutLM's README figures

Colors come from the light half of themes/theme.json, Akiba, and text from Google Sans Flex
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402
from fontTools.varLib import instancer  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# what the generators count, score and cut, kept apart from the code
DATA = HERE.parent / "data"
THEME = json.loads((ROOT / "themes" / "theme.json").read_text(encoding="utf-8"))["light"]
INK = THEME["mOnSurface"]
MUTED = THEME["mOnSurfaceVariant"]
OUTLINE = THEME["mOutline"]
PAGE = THEME["mSurface"]
PANEL = THEME["mSurfaceVariant"]
LEAD = THEME["mPrimary"]
PURPLE = THEME["mSecondary"]
GREEN = THEME["terminal"]["normal"]["green"]
OCHRE = THEME["terminal"]["normal"]["yellow"]
GRAY = THEME["terminal"]["normal"]["white"]
# the theme's blue saturated as the paper's figures draw it, so it stands apart from purple
BLUE = "#1F74A8"

# a card figure is this wide in inches, placed from its top left, with its own title size
WIDE = 12.5
LEFT, RIGHT = 0.5, 12.0
HEADING = 27.0

STANCES = ("extends", "critiques", "compares")
STANCE_COLORS = {"extends": BLUE, "compares": PURPLE, "critiques": OCHRE}
# what each intent means, condensed from the teacher prompt the labeler was distilled from
MEANINGS = {
    "background": "general context, or a pointer to related work",
    "uses": "applies its method, data or tool unchanged",
    "compares": "sets results against it, or agrees with it",
    "critiques": "names a flaw, a limitation or a disagreement",
    "extends": "builds on, generalizes or modifies it",
    "motivates": "is what inspired the citing work",
}
# Static cuts of the variable font, as matplotlib renders only a variable font's default
WEIGHTS = {"Regular": 400, "Medium": 500, "SemiBold": 600, "Bold": 700}


def register_fonts() -> None:
    """
    Cuts one static font per weight from the variable font, then registers them with matplotlib
    """
    variable = ROOT / "themes" / "fonts" / "GoogleSansFlex-VariableFont.ttf"
    out = DATA / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    for name, weight in WEIGHTS.items():
        path = out / f"GoogleSansFlex-{name}.ttf"
        if not path.is_file():
            axes = {"wght": weight, "opsz": 18, "wdth": 100, "GRAD": 0, "ROND": 0, "slnt": 0}
            cut = instancer.instantiateVariableFont(TTFont(variable), axes, updateFontNames=True)
            cut.save(path)
        font_manager.fontManager.addfont(str(path))


def apply_theme() -> None:
    """
    Registers the fonts and sets the colors and a type scale that stays legible at card width
    """
    register_fonts()
    plt.rcParams.update(
        {
            "font.family": "Google Sans Flex",
            "font.size": 12,
            # equations in Computer Modern, the face LaTeX sets them in
            "mathtext.fontset": "cm",
            "text.color": INK,
            "figure.facecolor": PAGE,
            "savefig.facecolor": PAGE,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.35,
            "axes.facecolor": PANEL,
            "axes.edgecolor": OUTLINE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": MUTED,
            "axes.labelsize": 11.5,
            "axes.labelpad": 8,
            "axes.titlesize": 14.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "axes.axisbelow": True,
            "grid.color": PAGE,
            "grid.linewidth": 1.2,
            "grid.alpha": 1.0,
            "xtick.color": MUTED,
            "xtick.labelcolor": MUTED,
            "xtick.labelsize": 11,
            "ytick.color": MUTED,
            "ytick.labelcolor": INK,
            "ytick.labelsize": 12.5,
            "ytick.major.size": 0,
            "ytick.major.pad": 10,
            "legend.frameon": False,
            "legend.fontsize": 12.5,
            "lines.solid_capstyle": "round",
        }
    )


def tint(color: str, share: float) -> str:
    """
    A color mixed toward white, for the fill behind a node's text

    Args:
        color: The full color.
        share: How much of it to keep, 0 for white and 1 for the color itself.

    Returns:
        The mixed color as hex.
    """
    r, g, b = to_rgb(color)
    return to_hex((1 - share + share * r, 1 - share + share * g, 1 - share + share * b))


# what each kind of node in a method diagram is drawn with, as (fill, edge)
KINDS = {
    "data": (PANEL, OUTLINE),
    "model": (tint(LEAD, 0.13), LEAD),
    "step": (tint(PURPLE, 0.11), PURPLE),
}


def write_runs(
    fig: Figure,
    x: float,
    y: float,
    runs: list[tuple[str, bool | str]],
    size: float,
    weight: str = "bold",
    ink: str = INK,
) -> float:
    """
    One line of text written run by run, the emphasized runs in color, joined by single spaces

    Args:
        fig: The figure to write on.
        x: Left edge in figure coordinates.
        y: Baseline in figure coordinates.
        runs: (text, emphasis) pairs, True for the lead red, a color for that color, or False.
        size: Font size in points.
        weight: Font weight of every run.
        ink: Color of the plain runs.

    Returns:
        The right edge of the line in figure coordinates.
    """
    renderer = fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    style = {"fontsize": size, "fontweight": weight, "va": "baseline"}
    figure_width = fig.get_figwidth() * fig.dpi
    left = x
    for index, (text, emphasized) in enumerate(runs):
        piece = text if index == len(runs) - 1 else text + " "
        color = emphasized if isinstance(emphasized, str) else LEAD if emphasized else ink
        drawn = fig.text(left, y, piece, color=color, **style)
        # a trailing space is not measured, so it is measured on a probe ending in x
        probe = fig.text(0, 0, piece + "x", **style)
        bare = fig.text(0, 0, "x", **style)
        left += (
            probe.get_window_extent(renderer).width - bare.get_window_extent(renderer).width
        ) / figure_width
        probe.remove()
        bare.remove()
        _ = drawn
    return left


def mark(fig: Figure, artist: Any, number: int, color: str = LEAD) -> None:
    """
    A footnote number raised at the top right of a piece of text already drawn

    Args:
        fig: The figure, drawn at least once since the text was placed.
        artist: The text the footnote belongs to.
        number: The footnote's number.
        color: The number's color.
    """
    renderer = fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    box = artist.get_window_extent(renderer)
    x, y = fig.transFigure.inverted().transform((box.x1 + 1.5, box.y1))
    fig.text(x, y, str(number), fontsize=8.5, color=color, fontweight="bold", va="top")


def short(n: float) -> str:
    """
    A count in thousands or millions, as the bars label it

    Args:
        n: The count.

    Returns:
        The label.
    """
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def compact(n: int) -> str:
    """
    A count in thousands from ten thousand up, and in full below that

    Args:
        n: The count.

    Returns:
        The label.
    """
    return f"{n / 1e3:.0f}k" if n >= 10_000 else f"{n:,}"


def tick(n: float) -> str:
    """
    A round count for an axis, with no trailing zero

    Args:
        n: The count.

    Returns:
        The label.
    """
    return f"{n / 1e6:g}M" if n >= 1e6 else f"{n / 1e3:g}k"


class Page:
    """
    One card figure, laid out in inches from its top left corner on an overlay that spans it
    """

    def __init__(self, high: float, wide: float = WIDE) -> None:
        """
        A blank page, as wide as a card unless told otherwise

        Args:
            high: The page's height in inches.
            wide: The page's width in inches, its right margin matching a card's.
        """
        self.high = high
        self.wide = wide
        self.right = wide - (WIDE - RIGHT)
        self.fig = plt.figure(figsize=(wide, high))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, wide)
        self.ax.set_ylim(high, 0)
        self.ax.axis("off")
        self.ax.patch.set_visible(False)
        self.marked: list[tuple[Any, int]] = []

    def at(self, x: float, top: float) -> tuple[float, float]:
        """
        A point in inches from the top left, in figure coordinates

        Args:
            x: Inches from the left edge.
            top: Inches from the top edge.

        Returns:
            The point.
        """
        return x / self.wide, 1 - top / self.high

    def place(self, left: float, top: float, width: float, height: float) -> Axes:
        """
        A panel placed in inches from the page's top left corner

        Args:
            left: Left edge.
            top: Top edge.
            width: Width.
            height: Height.

        Returns:
            The panel.
        """
        x, y = self.at(left, top + height)
        return self.fig.add_axes((x, y, width / self.wide, height / self.high))

    def measure(self, text: str, size: float, **style: Any) -> float:
        """
        How wide a piece of text is drawn, in inches

        Args:
            text: The text.
            size: The font size.
            **style: Any other text property, a weight or a family.

        Returns:
            The width.
        """
        renderer = self.fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
        probe = self.ax.text(0, 0, text, fontsize=size, **style)
        width = probe.get_window_extent(renderer).width / self.fig.dpi
        probe.remove()
        return width

    def wrapped(self, text: str, width: float, size: float) -> list[str]:
        """
        Text broken into lines that each fit a width, measured in the font they are drawn in

        Args:
            text: The text.
            width: The room in inches.
            size: The font size.

        Returns:
            The lines.
        """
        lines: list[str] = []
        for word in text.split():
            trial = f"{lines[-1]} {word}" if lines else word
            if lines and self.measure(trial, size) <= width:
                lines[-1] = trial
            else:
                lines.append(word)
        return lines

    def fit(self, text: str, width: float, size: float) -> str:
        """
        Text cut at a word and closed with an ellipsis when it is wider than the room

        Args:
            text: The text.
            width: The room in inches.
            size: The font size.

        Returns:
            The text as it fits.
        """
        words = text.split()
        while len(words) > 1 and self.measure(" ".join(words), size) > width:
            words = words[:-1]
            words[-1] = words[-1].rstrip(",:;") + "..."
        return " ".join(words)

    def title(self, lines: list[list[tuple[str, bool | str]]]) -> None:
        """
        The page's title at the top, one or more lines of runs

        Args:
            lines: Each line's (text, emphasis) runs.
        """
        for index, runs in enumerate(lines):
            top = 0.95 + 0.02 * HEADING * index
            write_runs(self.fig, *self.at(LEFT, top), runs, HEADING, "bold", INK)

    def panel_title(self, x: float, top: float, text: str, note: int = 0) -> Any:
        """
        A panel's title, placed by hand so it can align with the labels rather than the plot

        Args:
            x: Left edge, in inches.
            top: Baseline, in inches from the top.
            text: The title.
            note: Its footnote, or 0.

        Returns:
            The text.
        """
        drawn = self.ax.text(
            x, top, text, fontsize=14.5, fontweight="semibold", color=INK, va="baseline"
        )
        if note:
            self.marked.append((drawn, note))
        return drawn

    def key(self, right: float, top: float, runs: list[tuple[str, bool | str]]) -> None:
        """
        Colored words ending at a right edge, which name what each color in a panel is

        Args:
            right: Right edge, in inches.
            top: Baseline, in inches from the top.
            runs: (word, color) pairs, joined by single spaces.
        """
        across = self.measure(" ".join(t for t, _ in runs), 11.5, fontweight="semibold")
        write_runs(self.fig, *self.at(right - across, top), runs, 11.5, "semibold")

    def box(
        self,
        x: float,
        top: float,
        width: float,
        high: float,
        name: str,
        detail: str,
        kind: str,
        note: int = 0,
    ) -> tuple[float, float, float, float]:
        """
        A node of a method diagram, its name over a line or two of detail

        Args:
            x: Left edge.
            top: Top edge.
            width: Width.
            high: Height.
            name: The bold first line.
            detail: The muted lines under it.
            kind: data or model, which picks its colors.
            note: The name's footnote, or 0.

        Returns:
            Its left, right, top and bottom edges, for the arrows to meet.
        """
        fill, edge = KINDS[kind]
        self.ax.add_patch(
            FancyBboxPatch(
                (x, top),
                width,
                high,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.3,
                zorder=3,
            )
        )
        lines = detail.split("\n") if detail else []
        # the name and its detail are centered on the box as one block
        start = top + (high - 0.19 - 0.205 * len(lines)) / 2
        middle = x + width / 2
        drawn = self.ax.text(
            middle,
            start + 0.165,
            name,
            ha="center",
            va="baseline",
            fontsize=13,
            fontweight="semibold",
            zorder=4,
        )
        if note:
            self.marked.append((drawn, note))
        for index, line in enumerate(lines):
            self.ax.text(
                middle,
                start + 0.38 + 0.205 * index,
                line,
                ha="center",
                va="baseline",
                fontsize=11,
                color=MUTED,
                zorder=4,
            )
        return x, x + width, top, top + high

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        bend: float = 0.0,
        dashed: bool = False,
    ) -> None:
        """
        An arrow from one point of a diagram to another

        Args:
            start: Where it leaves.
            end: Where it arrives.
            bend: Its curvature, 0 for a straight arrow.
            dashed: Whether it is drawn dashed, for a check rather than a flow.
        """
        self.ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                color=MUTED if dashed else INK,
                linewidth=1.3,
                linestyle=(0, (3, 2.5)) if dashed else "-",
                connectionstyle=f"arc3,rad={bend}",
                shrinkA=0,
                shrinkB=0,
                zorder=2,
            )
        )

    def stage(self, x: float, top: float, letter: str, name: str, note: int = 0) -> None:
        """
        A lettered stage's heading, the letter in a red disc

        Args:
            x: Left edge.
            top: Baseline, in inches from the top.
            letter: The stage's letter.
            name: What the stage is.
            note: Its footnote, or 0.
        """
        self.ax.add_patch(Circle((x + 0.16, top - 0.07), 0.16, facecolor=LEAD, zorder=3))
        self.ax.text(
            x + 0.16,
            top - 0.065,
            letter,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
            zorder=4,
        )
        drawn = self.ax.text(x + 0.45, top, name, fontsize=14, fontweight="semibold", va="baseline")
        if note:
            self.marked.append((drawn, note))

    def result(self, x: float, top: float, headline: str, detail: str, note: int = 0) -> None:
        """
        What a stage came to, a red line over a muted one, centered on a point

        Args:
            x: Center.
            top: The first line's baseline.
            headline: The result.
            detail: What it was measured on.
            note: The result's footnote, or 0.
        """
        drawn = self.ax.text(
            x, top, headline, ha="center", fontsize=12, color=LEAD, fontweight="semibold"
        )
        if note:
            self.marked.append((drawn, note))
        self.ax.text(x, top + 0.25, detail, ha="center", fontsize=11, color=MUTED)

    def kinds(self, names: list[str]) -> None:
        """
        A legend at the right of the title naming the kinds of node a diagram draws

        Args:
            names: The kinds drawn, each a key of KINDS.
        """
        labels = {"data": "data", "model": "model", "step": "procedure"}
        handles = [
            Patch(
                facecolor=KINDS[name][0],
                edgecolor=KINDS[name][1],
                linewidth=1.3,
                label=labels[name],
            )
            for name in names
        ]
        self.fig.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=self.at(self.right, 0.95),
            ncol=len(handles),
            fontsize=12,
            handlelength=1.4,
            handleheight=1.0,
            borderpad=0,
            borderaxespad=0,
            columnspacing=1.4,
        )

    def footnotes(self, top: float, notes: list[str]) -> float:
        """
        The numbered footnotes, in two columns of about equal height when there is more than one

        Args:
            top: The first line's baseline, in inches from the top.
            notes: The footnotes, the first numbered 1.

        Returns:
            The last line's baseline.
        """
        sides = 2 if len(notes) > 1 else 1
        column = (self.right - LEFT) / sides
        lines = [self.wrapped(note, column - 0.45, 11) for note in notes]
        heights = [0.205 * len(each) + 0.09 for each in lines]
        split = len(notes)
        if sides == 2:
            split = min(range(1, len(notes)), key=lambda k: max(sum(heights[:k]), sum(heights[k:])))
        bottom = top
        for side, (start, stop) in enumerate(((0, split), (split, len(notes)))[:sides]):
            y = top
            x = LEFT + side * (column + 0.15)
            for number in range(start, stop):
                self.ax.text(
                    x,
                    y,
                    str(number + 1),
                    fontsize=9,
                    color=LEAD,
                    fontweight="bold",
                    va="baseline",
                )
                for line in lines[number]:
                    self.ax.text(x + 0.2, y, line, fontsize=11, color=MUTED, va="baseline")
                    y += 0.205
                y += 0.09
            bottom = max(bottom, y - 0.295)
        return bottom

    def save(self, path: Path, notes_top: float, notes: list[str]) -> None:
        """
        Marks every footnote, writes them out and saves the page

        Args:
            path: Where the figure is written.
            notes_top: Where the footnotes start, in inches from the top.
            notes: The footnotes, the first numbered 1.
        """
        self.fig.canvas.draw()
        for artist, number in self.marked:
            mark(self.fig, artist, number)
        bottom = self.footnotes(notes_top, notes)
        if bottom > self.high - 0.3:
            print(f"[{path.stem}] the notes run {bottom - self.high + 0.3:.2f} in too low")
        self.fig.savefig(path, bbox_inches=Bbox([[0, 0], [self.wide, self.high]]))
        plt.close(self.fig)
        print(f"[{path.stem}] wrote {path}", flush=True)


def intents(
    ax: Axes, x: float, top: float, width: float, counts: dict[str, int], names: list[str]
) -> None:
    """
    Intents, each with what it means and a bar of how often it occurs, scaled to the most common

    Args:
        ax: The page's overlay, in inches from the top left.
        x: Left edge.
        top: Top edge.
        width: Width.
        counts: Every intent to its labeled citations.
        names: The intents to draw, in order.
    """
    total = sum(counts.values())
    biggest = max(counts.values())
    span = width - 1.45
    for index, intent in enumerate(names):
        y = top + index * 0.7
        ax.text(x, y + 0.15, intent, fontsize=11.5, fontweight="semibold", va="baseline")
        ax.text(x + 1.1, y + 0.15, MEANINGS[intent], fontsize=10.5, color=MUTED, va="baseline")
        length = span * counts[intent] / biggest
        ax.add_patch(
            FancyBboxPatch(
                (x, y + 0.25),
                length,
                0.18,
                boxstyle="round,pad=0,rounding_size=0.03",
                facecolor=STANCE_COLORS.get(intent, tint(GRAY, 0.55)),
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x + length + 0.08,
            y + 0.34,
            f"{short(counts[intent])}  ·  {counts[intent] / total:.1%}",
            fontsize=10.5,
            va="center",
        )
