"""
Table appendix_ppr: the core, pool, damping, teleport and direction PPR picked for each recipe and embedder

Every value is read from the seed-0 PPR tune's report.json, scored on 200 train cases of the recipe's own benchmark
"""

from pathlib import Path

from common import CONFIG, EMBEDDERS, RECIPES, blocks_table, float_table, report, save

STEM = Path(__file__).stem
SOURCES = CONFIG["table_main_recipes"]["sources"]


def main() -> None:
    blocks, data = [], {}
    for r in RECIPES:
        rows = []
        for e in EMBEDDERS:
            rep = report(f"{e}-{r}_ppr_seed0")
            p = rep["pick"]
            data[f"{e}-{r}"] = p
            rows.append(f"{EMBEDDERS[e]} & {p['core']} & {p['pool']} & {p['alpha']} & {p['teleport']} & {p['direction']} & {p['recall']:.3f} \\\\")
        title = f"\\term{{{r}}}, tuned on {SOURCES[rep['spec']['dataset']]} at R@{rep['k']}"
        blocks.append((title, rows))
    heads = ["Core", "Pool", "$\\alpha$", "Teleport", "Direction", "Recall"]
    caption = "The core, pool, damping, teleport and direction PPR picked for each recipe and embedder, with the recall it reached on the 200 train cases it was picked on."
    save(STEM, float_table(STEM, [blocks_table("Embedder", heads, blocks)], caption, "tab:ppr", "scriptsize", star=False), data)


if __name__ == "__main__":
    main()
