"""
Table appendix_exponents: every embedder's tuned exponents for each recipe

Every value is read from the seed-0 BoundedField tune's recipe.toml, which extends table_main_recipes past Nomic
"""

from pathlib import Path

from common import EMBEDDERS, RECIPES, TERMS, blocks_table, float_table, save, weights
from table_main_recipes import term

STEM = Path(__file__).stem


def main() -> None:
    blocks, data = [], {}
    for r in RECIPES:
        rows = []
        for e in EMBEDDERS:
            w = weights(f"{e}-{r}_bf_seed0")
            data[f"{e}-{r}"] = w
            rows.append(f"{EMBEDDERS[e]} & " + " & ".join(f"{w.get(t, 0.0):.2f}" for t in TERMS) + " \\\\")
        blocks.append((f"\\term{{{r}}}", rows))
    caption = "The exponents BoundedField was tuned to for each recipe and embedder."
    save(STEM, float_table(STEM, [blocks_table("Embedder", [term(t) for t in TERMS], blocks)], caption, "tab:exponents", "scriptsize"), data)


if __name__ == "__main__":
    main()
