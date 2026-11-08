# webui

The research library over saved runs, served by the frontend role.
There is no build step and no framework, and the one dependency, the markdown renderer, is vendored.

```
index.html            the page layout and the side panel
app.js                the library, a run's five tabs, and the citation panel
style.css             the four palettes, light and dark, and the layout
icon.svg              the G mark
vendor/marked.min.js  the markdown renderer
```

## A Run

| tab | what it shows |
| --- | --- |
| Report | the TL;DR as the answer card, the results table built from the reads, papers by year and by topic, the lineages as timelines, the method tables, then the report's sections as cards |
| Papers | every paper read, cited ones first, with each claim the reader kept from it |
| Process | where the time went as one stacked bar, then the run as a tree of dots read left to right, calls sent together branching; hovering a dot gives one line, and a click or the arrow keys open its parameters, response or thinking below |
| Timeline | the citation graph among every paper a tool returned, by date and citation count, colored by its role in the run, with citations by intent and the lines from trace_lineages |
| Check | whether the report passed the mechanical check, and each fault |

The inspector docked on the right follows the citation under the pointer: the paper, the paragraph with the quote marked, the reader's claim, and the paper's other claims in the run. A click pins it.
The tab is part of the url, so `#/run/<slug>/process` is a link someone can send.
