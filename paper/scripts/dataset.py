"""
Figure 1: (a) the corpus at a glance, (b) how every citation gets its intent

Usage: direnv exec . uv run python paper/scripts/dataset.py [--refresh]
Counts are pinned in dataset.json beside this file to the store on 2026-11-11, as the paper states them, drawn at the ICML page size
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import psycopg
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
from matplotlib import font_manager

BIOLINUM = Path("/nix/store/5xhrl01vhivcmwvr3sdkz0ygdbsyd84l-texlive-2025-r78234-final-env-texmfdist/fonts/opentype/public/libertine")
for f in BIOLINUM.glob("LinBiolinum_R*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Linux Biolinum O"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "stix"
ROOT = HERE.parents[1]
DSN = f"host={ROOT / '.pg'} port=5433 dbname=arxivkg"

WIDTH, HEIGHT = 6.75, 3.9
A_W, B_X = 3.92, 4.12
INK = "#2A1E18"
MUTED = "#60504A"
FAINT = "#887C77"
RULE = "#D8C8C0"
CARD = "#F8F4F0"
SOFT = "#EDE3DD"
LEAD = "#902020"
SECOND = "#1F74A8"
SURVEY = 0.5
INTENTS = ["background", "uses", "compares", "critiques", "extends", "motivates"]
# the conditions stage0_create_citation.py tries, cheapest first
RESOLVERS = [
    "arXiv id in link",
    "arXiv id in text",
    "DOI",
    "journal reference",
    "exact title and year",
    "exact title",
    "checked fuzzy title",
]
# from the labeler runs, which the store keeps no record of
PRELIMINARY = "~10,000"
HUMAN = 985
GENERATIONS = 6
APE_BEST = 973
TEACHER = 9972
STUDENT_RATE = 1762


def tint(color: str, share: float) -> str:
    """
    A color mixed toward white

    Args:
        color: The hex color.
        share: How much of the color is kept.

    Returns:
        The mixed hex color.
    """
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - v) * share):02X}" for v in rgb)


def query() -> dict:
    """
    Every number panel a draws and the citation counts panel b prints

    Returns:
        The counts by name.
    """
    with psycopg.connect(DSN) as conn:
        years = conn.execute(
            "select extract(year from date)::int, count(*), count(*) filter (where survey_probability > %s) "
            "from papers group by 1 order by 1",
            (SURVEY,),
        ).fetchall()
        cites = conn.execute(
            "select extract(year from p.date)::int, count(*) from citations c "
            "join papers p on p.arxiv_id = c.source_id where c.target_id is not null group by 1 order by 1"
        ).fetchall()
        intents = dict(
            conn.execute("select intent, count(*) from citations where target_id is not null and stage = 1 group by 1").fetchall()
        )
        parsed, resolved = conn.execute("select count(*), count(target_id) from citations").fetchone()
        topics = conn.execute(
            "select t.level_0, n.name, count(*) from topic_tree t "
            "join topic_names n on n.level_0 = t.level_0 and n.level_1 is null group by 1, 2 order by 3 desc limit 5"
        ).fetchall()
        top0 = topics[0][0]
        subtopics = conn.execute(
            "select t.level_1, n.name, count(*) from topic_tree t "
            "join topic_names n on n.level_0 = t.level_0 and n.level_1 = t.level_1 and n.level_2 is null "
            "where t.level_0 = %s group by 1, 2 order by 3 desc limit 5",
            (top0,),
        ).fetchall()
        top1 = subtopics[0][0]
        leaves = conn.execute(
            "select t.level_2, n.name, count(*) from topic_tree t "
            "join topic_names n on (n.level_0, n.level_1, n.level_2) = (t.level_0, t.level_1, t.level_2) "
            "where t.level_0 = %s and t.level_1 = %s group by 1, 2 order by 3 desc limit 5",
            (top0, top1),
        ).fetchall()
    return {
        "years": years,
        "cites": cites,
        "intents": intents,
        "parsed": parsed,
        "resolved": resolved,
        "topics": [[n, c] for _, n, c in topics],
        "subtopics": [[n, c] for _, n, c in subtopics],
        "leaves": [[n, c] for _, n, c in leaves],
    }

BODY = 5.9
LINE = 0.09
HEAD = 0.2
PAD = 0.04
CARD_GAP = 0.06
B_W = WIDTH - B_X


def card(ax: Axes, x: float, top: float, w: float, h: float, color: str, title: str, step: str = "") -> float:
    """
    One item of the figure, a tinted rounded box with its title and an optional step badge

    Args:
        ax: The canvas, in inches.
        x: Left edge.
        top: Top edge.
        w: Width.
        h: Height.
        color: The item's hue.
        title: Its title.
        step: The step number, or empty for none.

    Returns:
        The y its content starts at.
    """
    ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=CARD, ec=RULE, lw=0.6))
    tx = x + 0.08
    if step:
        ax.add_patch(plt.Circle((x + 0.11, top - 0.098), 0.05, fc=color, ec="none"))
        ax.text(x + 0.11, top - 0.098, step, fontsize=5.4, color="white", weight="bold", ha="center", va="center_baseline")
        tx = x + 0.21
    ax.text(tx, top - 0.055, title, fontsize=6.6, color=INK, weight="semibold", va="top")
    return top - HEAD


def inset(fig: plt.Figure, x: float, y: float, w: float, h: float) -> Axes:
    """
    Axes placed by a box in inches

    Args:
        fig: The figure.
        x: Left edge.
        y: Bottom edge.
        w: Width.
        h: Height.

    Returns:
        The axes.
    """
    ax = fig.add_axes((x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT))
    ax.patch.set_alpha(0)
    return ax


def per_year(ax: Axes, rows: list, unit: float, label: str, part: list | None = None) -> None:
    """
    A count per year as a flat area under a line, with an optional share of it as a second area

    Args:
        ax: The axes.
        rows: (year, count) pairs.
        unit: What one tick unit is worth.
        label: The tick suffix.
        part: (year, count) pairs of the share drawn under the first, or None.
    """
    rows = [r for r in rows if r[0] >= 1991]
    xs = [r[0] for r in rows]
    ys = [r[1] / unit for r in rows]
    ax.fill_between(xs, ys, color=tint(LEAD, 0.22), lw=0)
    ax.plot(xs, ys, color=LEAD, lw=1.1, solid_capstyle="round")
    if part is not None:
        share = dict((y, c) for y, c in part)
        ps = [share.get(x, 0) / unit for x in xs]
        ax.fill_between(xs, ps, color=tint(SECOND, 0.35), lw=0)
        ax.plot(xs, ps, color=SECOND, lw=1.1, solid_capstyle="round")
    ax.tick_params(labelsize=5.8, colors=MUTED, length=2, width=0.6, pad=1.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_xlim(1991, 2026)
    ax.set_ylim(0, max(ys) * 1.08)
    ax.set_xticks([1995, 2005, 2015, 2025])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}{label}" if v else "0")
    ax.grid(False)
    ax.grid(axis="y", color=RULE, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def key(ax: Axes, x: float, y: float, color: str, label: str, back: str) -> None:
    """
    One legend entry inside a chart, a short colored stroke and its label in text ink

    Args:
        ax: The axes.
        x: Left, in axes fractions.
        y: Top, in axes fractions.
        color: The series color.
        label: What it is.
        back: The fill behind the label.
    """
    ax.plot([x, x + 0.07], [y - 0.05, y - 0.05], transform=ax.transAxes, color=color, lw=1.6, solid_capstyle="round", clip_on=False)
    ax.text(x + 0.1, y, label, transform=ax.transAxes, fontsize=5.6, color=MUTED, va="top", bbox=dict(fc=back, ec="none", pad=0.6))


def fit(ax: Axes, text: str, width: float) -> list[str]:
    """
    Words broken into lines no wider than a width, measured as the figure will render them

    Args:
        ax: The axes the text goes in.
        text: The words.
        width: The widest a line may be, in inches.

    Returns:
        The lines.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=BODY)
    lines: list[str] = []
    for word in text.split():
        trial = f"{lines[-1]} {word}" if lines else word
        probe.set_text(trial)
        if lines and probe.get_window_extent(renderer).width / fig.dpi > width:
            lines.append(word)
        elif lines:
            lines[-1] = trial
        else:
            lines.append(word)
    probe.remove()
    return lines


def pill(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """
    A bar with fully rounded ends

    Args:
        ax: The axes, in inches.
        x: Left edge.
        y: Bottom edge.
        w: Length.
        h: Thickness.
        color: Its fill.
    """
    w = max(w, h)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}", fc=color, ec="none", clip_on=False))


def column(ax: Axes, x0: float, width: float, top: float, rows: list, title: str, fmt: str) -> tuple:
    """
    One level of the topic tree, each row a name over a rounded bar with its value at the bar's end

    Args:
        ax: The canvas, in inches.
        x0: Left edge.
        width: Width the names and bars may use.
        top: Top of the column title.
        rows: (name, count) pairs, largest first.
        title: The column title.
        fmt: The format of the printed value, in thousands.

    Returns:
        The first row's top and bottom, the column's top and bottom, and the y it ends at.
    """
    ax.text(x0, top, title, fontsize=BODY, color=MUTED, va="top")
    y = top - 0.15
    start = y
    first = (0.0, 0.0)
    peak = max(c for _, c in rows)
    for i, (name, count) in enumerate(rows):
        lines = fit(ax, name, width)
        row_top = y
        ax.text(x0, y, "\n".join(lines), va="top", ha="left", fontsize=BODY, color=INK, linespacing=1.1)
        y -= len(lines) * 0.085 + 0.02
        bar = (width - 0.3) * count / peak
        pill(ax, x0, y - 0.04, bar, 0.04, LEAD if i == 0 else tint(LEAD, 0.45))
        ax.text(x0 + bar + 0.04, y - 0.02, fmt.format(count / 1e3), va="center", ha="left", fontsize=5.6, color=MUTED)
        if i == 0:
            first = (row_top + 0.01, y - 0.05)
        y -= 0.04 + 0.07
    return first, (start + 0.01, y + 0.06), y


def zoom(ax: Axes, x0: float, x1: float, src: tuple, dst: tuple) -> None:
    """
    A soft wedge from one highlighted row to the whole column it opens into

    Args:
        ax: The canvas, in inches.
        x0: Right edge of the source column.
        x1: Left edge of the destination column.
        src: Top and bottom of the source row.
        dst: Top and bottom of the destination column.
    """
    ax.add_patch(plt.Polygon([(x0, src[0]), (x1, dst[0]), (x1, dst[1]), (x0, src[1])], closed=True, fc=SOFT, ec="none"))
    for a, b in ((src[0], dst[0]), (src[1], dst[1])):
        ax.plot([x0, x1], [a, b], color=tint(LEAD, 0.4), lw=0.5)


def overview(fig: plt.Figure, ax: Axes, c: dict, top: float, row_h: float) -> None:
    """
    Panel a, the corpus by year and intent in three cards above its topic tree in a fourth

    Args:
        fig: The figure.
        ax: The canvas, in inches.
        c: The data.
        top: Top of the cards.
        row_h: Height of the top row of cards.
    """
    w = (A_W - 2 * CARD_GAP) / 3
    xs = [i * (w + CARD_GAP) for i in range(3)]
    charts = [
        ("Papers per year", LEAD, c["years"], 1e3, "k", [[y, s] for y, _, s in c["years"]]),
        ("Citations per citing year", LEAD, c["cites"], 1e6, "M", None),
    ]
    back = CARD
    for x, (title, color, rows, unit, label, part) in zip(xs, charts):
        card(ax, x, top, w, row_h, color, title)
        a = inset(fig, x + 0.3, top - row_h + 0.17, w - 0.38, row_h - HEAD - 0.23)
        per_year(a, rows, unit, label, part)
    a, b = fig.axes[-2:]
    key(a, 0.04, 1.0, LEAD, f"{sum(r[1] for r in c['years']):,} papers", back)
    key(a, 0.04, 0.86, SECOND, f"{sum(r[2] for r in c['years']):,} reviews", back)
    key(b, 0.04, 1.0, LEAD, f"{sum(r[1] for r in c['cites']):,} resolved", back)

    y = card(ax, xs[2], top, w, row_h, LEAD, "Citation intent") - 0.03
    total = sum(c["intents"].values())
    peak = max(c["intents"].values())
    for i, name in enumerate(INTENTS):
        n = c["intents"][name]
        ax.text(xs[2] + 0.08, y, name, fontsize=BODY, color=INK, va="top")
        bar = (w - 0.95) * n / peak
        pill(ax, xs[2] + 0.56, y - 0.065, bar, 0.04, LEAD if i == 0 else tint(LEAD, 0.45))
        ax.text(xs[2] + 0.6 + bar, y - 0.045, f"{100 * n / total:.1f}%", fontsize=5.6, color=MUTED, va="center")
        y -= (row_h - HEAD - 0.12) / 5.4

    tree_top = top - row_h - CARD_GAP
    t = card(ax, 0, tree_top, A_W, tree_top - 0.005, LEAD, "Topic tree") - 0.02
    widths = [0.84, 1.04, 1.34]
    wedge = 0.21
    cols = [0.1, 0.1 + widths[0] + wedge, 0.1 + widths[0] + widths[1] + 2 * wedge]
    levels = [(c["topics"], "16 topics", "{:.0f}k"), (c["subtopics"], "256 subtopics", "{:.1f}k"), (c["leaves"], "4,096 leaves", "{:.1f}k")]
    spans = [column(ax, x, wd, t, rows, title, fmt) for x, wd, (rows, title, fmt) in zip(cols, widths, levels)]
    for i in range(2):
        zoom(ax, cols[i] + widths[i] + 0.01, cols[i + 1] - 0.04, spans[i][0], spans[i + 1][1])
    print(f"tree ends at {min(s[2] for s in spans):.2f} in, card bottom 0.005")


def node(ax: Axes, x: float, y: float, w: float, h: float, label: str) -> None:
    """
    A small white rounded box with centered text, for the intents and the steps of the loop

    Args:
        ax: The canvas, in inches.
        x: Left edge.
        y: Vertical center.
        w: Width.
        h: Height.
        label: The text.
    """
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.035", fc="white", ec=RULE, lw=0.6))
    ax.text(x + w / 2, y, label, fontsize=5.7, color=INK, ha="center", va="center_baseline")


def cycle(ax: Axes, x0: float, top: float) -> None:
    """
    The prompt search drawn as a loop of four steps, with its exit

    Args:
        ax: The canvas, in inches.
        x0: Left edge of the content.
        top: The y the loop starts below.
    """
    w, h = 0.92, 0.13
    xl, xr = x0, B_X + B_W - 0.21 - w
    yt, yb = top - h / 2 - 0.01, top - h / 2 - 0.28
    for x, y, label in [(xl, yt, r"$\mathcal{L}$ labels the 985"), (xr, yt, r"score accuracy $s_t$"), (xr, yb, r"$\mathcal{J}$ critiques misses $M_t$"), (xl, yb, r"$\mathcal{D}$ drafts $p_{t+1}$")]:
        node(ax, x, y, w, h, label)
    arrow = dict(arrowstyle="-|>", mutation_scale=5.5, color=LEAD, lw=0.7, shrinkA=2, shrinkB=2)
    ax.add_patch(FancyArrowPatch((xl + w, yt), (xr, yt), **arrow))
    ax.add_patch(FancyArrowPatch((xr + w / 2, yt - h / 2), (xr + w / 2, yb + h / 2), **arrow))
    ax.add_patch(FancyArrowPatch((xr, yb), (xl + w, yb), **arrow))
    ax.add_patch(FancyArrowPatch((xl + w / 2, yb + h / 2), (xl + w / 2, yt - h / 2), **arrow))
    ax.text((xl + xr + w) / 2, (yt + yb) / 2, f"repeat for {GENERATIONS} generations", fontsize=5.7, color=LEAD, ha="center", va="center", style="italic")
    ax.text(x0, yb - h / 2 - 0.05, rf"then keep the best prompt $p^\ast$ at {APE_BEST} of {HUMAN} correct", fontsize=BODY, color=INK, va="top")


def labeling(ax: Axes, c: dict, top: float) -> None:
    """
    Panel b, the seven stages from a parsed citation to a labeled corpus, one card each

    Args:
        ax: The canvas, in inches.
        c: The data.
        top: Top of the first card.
    """
    gap = 0.04
    body = dict(fontsize=BODY, color=MUTED, va="top")
    one = HEAD + LINE + PAD
    x0 = B_X + 0.21
    texts = {
        0: f"bibliography entries with their citing sentences: {c['parsed']:,}",
        3: f"~1,000 drawn by intent and relabeled by hand into {HUMAN}",
        5: f"the best prompt labels {TEACHER:,} citations to train DistilBERT",
        6: f"{c['resolved']:,} citations at {STUDENT_RATE:,} per second in about 12 h",
    }
    stages = [
        (one, LEAD, "Parse citations"),
        (HEAD + 5 * LINE + 0.02 + PAD, LEAD, "Resolve to an arXiv paper"),
        (HEAD + LINE + 0.3 + PAD, LEAD, "Preliminary labels"),
        (one, LEAD, "Human labels"),
        (HEAD + 0.41 + 0.05 + LINE + PAD, LEAD, "Prompt search (APE)"),
        (one, LEAD, "Student training"),
        (one, LEAD, "Full corpus labeling"),
    ]
    y = top
    for i, (h, color, title) in enumerate(stages):
        t = card(ax, B_X, y, B_W, h, color, title, str(i + 1))
        if i in texts:
            ax.text(x0, t, texts[i], **body)
        elif i == 1:
            for j, name in enumerate(RESOLVERS):
                ax.text(x0 + (j % 2) * 1.18, t - (j // 2) * LINE, f"{'abcdefg'[j]}. {name}", **body)
            ax.text(x0, t - 4 * LINE - 0.02, f"first match wins with {c['resolved']:,} resolved ({c['resolved'] / c['parsed']:.1%})", fontsize=BODY, color=INK, va="top")
        elif i == 2:
            ax.text(x0, t, f"Gemma 4 26B labels {PRELIMINARY} random citations as one of", **body)
            w = (B_W - 0.42 - 2 * 0.05) / 3
            for j, name in enumerate(INTENTS):
                node(ax, x0 + (j % 3) * (w + 0.05), t - LINE - 0.08 - (j // 3) * 0.16, w, 0.12, name)
        elif i == 4:
            cycle(ax, x0, t)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((B_X + B_W / 2, y - h), (B_X + B_W / 2, y - h - gap), arrowstyle="-|>", mutation_scale=5, color=FAINT, lw=0.7, shrinkA=0.5, shrinkB=0.5))
        y -= h + gap
    print(f"panel b ends at {y + gap:.2f} in")


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--refresh", action="store_true", help="query the store again")
    args = ap.parse_args()
    cache = HERE / "dataset.json"
    if args.refresh or not cache.exists():
        cache.write_text(json.dumps(query(), indent=1))
    c = json.loads(cache.read_text())

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    canvas = fig.add_axes((0, 0, 1, 1), zorder=-1)
    canvas.set_xlim(0, WIDTH)
    canvas.set_ylim(0, HEIGHT)
    canvas.axis("off")
    top = HEIGHT - 0.2
    canvas.text(0, HEIGHT - 0.01, "(a) The corpus", fontsize=8, color=INK, weight="semibold", va="top")
    canvas.text(B_X, HEIGHT - 0.01, "(b) Labeling citation intent", fontsize=8, color=INK, weight="semibold", va="top")
    canvas.plot([(A_W + B_X) / 2] * 2, [0.005, HEIGHT - 0.01], color=RULE, lw=0.8)
    overview(fig, canvas, c, top, 1.75)
    labeling(canvas, c, top)
    fig.savefig(HERE.parent / "figures/dataset.pdf")


if __name__ == "__main__":
    main()
