#!/usr/bin/env bash
# Copies every figure and table in results/ to the paper path its section of scripts/results/config.toml names
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import shutil
import tomllib
from pathlib import Path

config = tomllib.loads(Path("scripts/results/config.toml").read_text())
for stem, spec in config.items():
    if not isinstance(spec, dict) or "copy_to" not in spec:
        continue
    dest = Path(spec["copy_to"])
    source = Path("results") / stem.split("_", 1)[1] / f"{stem}{dest.suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    print(f"{source} -> {dest}")
    if "alt_to" in spec:
        alt = Path(spec["alt_to"])
        shutil.copyfile(source.with_suffix(".alt.tex"), alt)
        print(f"{source.with_suffix('.alt.tex')} -> {alt}")
PY
