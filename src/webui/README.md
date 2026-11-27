# webui

The research library over saved runs, served by the frontend role.
There is no build step and no framework, and the one dependency, the markdown renderer, is vendored.

```
index.html            the page layout and the side panel
app.js                the library, a run's five tabs, and the citation panel
report.js             the report object at two depths, the poset, the papers and the check
style.css             the four palettes, light and dark, and the layout
report.css            the report, poset, papers and check styles, on the palette tokens
icon.svg              the G mark
vendor/marked.min.js  the markdown renderer
```

## A Run

| Tab | What it shows |
| --- | --- |
| Report | The answer, then one row per section with its lead claim; Skim shows the rows and Read all opens every finding. Below them the evidence card: the poset of which paper extends, critiques or compares against which, the coverage grid, the table and the open questions. A run saved before the report object falls back to the markdown report |
| Papers | Every paper read, numbered as the report cites them, filtered by text and by cited or not, with its table answers and kept sentences |
| Process | Where the time went as one stacked bar, then the run as a tree of dots read left to right, calls sent together branching; hovering a dot gives one line, and a click or the arrow keys open its parameters, response or thinking below |
| Timeline | The citation graph among every paper a tool returned, by date and citation count, colored by its role in the run, with citations by intent and the longest chain of each poset |
| Check | The rules a report is held to, every save with its faults, and the run's files to download |

The inspector docked on the right follows the citation under the pointer: the paper, the paragraph with the quote marked, the reader's claim, and the paper's other claims in the run. A click pins it.
The tab is part of the url, so `#/run/<slug>/process` is a link someone can send.
