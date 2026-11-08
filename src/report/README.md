# report

The evidence a report rests on, and the check it passes before it is saved.
The prose belongs to the agent that called the tools, and nothing here writes or calls a model.

```
chunks.py  one paper as numbered paragraphs, tables and captions, and the ones a question needs
check.py   the mechanical check on an agent's report, string matching against what the run read
```

## The Paragraphs

`chunk_html` cuts a paper's html into paragraphs, keeping math as its LaTeX alttext and a table as its caption and rows, since results live there.
`chunk_pdf` falls back to `pdftotext` for a paper with no usable html, in windows cut at blank lines.
`rank` fuses cosine and BM25 to pick the paragraphs a question needs, and `around` adds their neighbors.
A quote is checked against the one paragraph it came from, so a paragraph id is the unit of evidence.

## The Check

`check` runs on every save, and a report passes only when it finds nothing.

| Fault | When |
| --- | --- |
| `unknown marker` | a label no read gave |
| `number not in its paragraph` | a number on a cited line that no paragraph behind its labels holds |
| `quote not in its paragraph` | a quoted phrase that none of the line's paragraphs holds |
| `name not in the papers cited` | a name on a cited sentence that no cited paper's full text holds |
| `paper cited but never read` | an arXiv id in the body that the reader never went through |
| `paper never returned by a tool` | an arXiv id in the further reading that no tool returned |
| `search claimed but never run` | a search the report describes with words the run never searched with |
| `statement with no marker` | a sentence of 12 words or more with no label, outside a line opening on Inference |

`save_report` in `src/mcp` adds three of its own: a TL;DR that cites nothing, one over 40 words, and a body citing nothing.
`split_tldr` takes the opening TL;DR paragraph off the report before the check.
`relist` rewrites the further reading from the store, so a listed title is never the agent's.
