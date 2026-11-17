# themes

The palette and the typeface every figure in this project uses: Akiba light, a red lead on warm paper, copied from the noctalia Akiba palette so the repository carries its own.

```
theme.json        the Akiba palette, dark and light, in the form the terminal and editor read
matplotlibrc      the light half of it as rcParams, plus the font and the Agg backend
fonts/            Google Sans Flex, the variable font matplotlibrc uses, with its OFL.txt
```

`.envrc` exports `MPLCONFIGDIR` pointing here, and matplotlib reads `matplotlibrc` out of that
directory on import, so any script launched through `direnv exec .` is themed without naming a
color, a font or a style file. A plot that sets its own colors is overriding the theme, which is
what to look for when one figure does not match the rest.

`scripts/citation_labeler/plot.py` is the one reader of `theme.json` itself, which it needs because
a matplotlibrc carries no dark half.

`fonts/` is the project's copy of the typeface. Matplotlib resolves it by family name through
fontconfig, so it uses whichever copy is installed on the machine and falls back to DejaVu Sans
when none is, in which case install the file here.

Google Sans Flex is under the SIL Open Font License 1.1, which asks that every copy of the font travel with its license, so `fonts/OFL.txt` is the copy from google/fonts and stays beside it.
