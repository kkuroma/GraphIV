"""
Table appendix_recipes_existing: every recipe as PPR and BoundedField on the existing sets with Nomic, beside the baselines

A dagger marks a score significantly below that set's own-mode BoundedField
"""

from pathlib import Path

from crossrecipe import build

if __name__ == "__main__":
    build(Path(__file__).stem)
