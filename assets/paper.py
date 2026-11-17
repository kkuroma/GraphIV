"""
Renders the paper's four figures into assets/figures as paper-<name>.png, for the README and docs

Figure 1 is TikZ inside the paper, so it is compiled alone in the paper's class and cropped
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper" / "figures"
OUT = HERE / "figures"
DPI = 250
STANDALONE = r"""\documentclass[sigconf,nonacm]{acmart}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\newcommand{\term}[1]{\textit{#1}}
\settopmatter{printacmref=false}
\pagestyle{empty}
\begin{document}
\thispagestyle{empty}
\noindent\input{teaser}
\end{document}
"""


def teaser(work: Path) -> Path:
    """
    Compiles Figure 1 on a page of its own

    Args:
        work: A scratch directory for the build.

    Returns:
        The compiled pdf.
    """
    _ = shutil.copy(PAPER / "teaser.tex", work / "teaser.tex")
    _ = (work / "page.tex").write_text(STANDALONE, encoding="utf-8")
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "page.tex"]
    _ = subprocess.run(command, cwd=work, check=True, capture_output=True)
    return work / "page.pdf"


def render(pdf: Path, out: Path, crop: bool = False) -> None:
    """
    Rasterizes the first page of a pdf onto white

    Args:
        pdf: The figure.
        out: Where the png is written.
        crop: Whether to cut the page to everything drawn on it.
    """
    with fitz.open(pdf) as doc:
        page = doc[0]
        clip = None
        if crop:
            clip = fitz.Rect()
            for _, box in page.get_bboxlog():
                clip |= fitz.Rect(box)
            clip = clip + (-4, -4, 4, 4)
        page.get_pixmap(dpi=DPI, alpha=False, clip=clip).save(out)
    print(f"[paper] wrote {out}", flush=True)


def main() -> None:
    """
    Renders Figure 1 from its TikZ and Figures 2 to 4 from their pdfs
    """
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        render(teaser(Path(work)), OUT / "paper-teaser.png", crop=True)
    for name in ("dataset", "benchmarks", "embedders"):
        render(PAPER / f"{name}.pdf", OUT / f"paper-{name}.png")


if __name__ == "__main__":
    main()
