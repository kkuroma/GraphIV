# report

The evidence a report rests on, and the check it passes before it is saved.
The prose belongs to the agent that called the tools, and nothing here writes or calls a model.

```
chunks.py  one paper as numbered paragraphs, tables and captions, and the ones a question needs
check.py   the mechanical check on an agent's report object, each fault under the path of its field
render.py  the report object as markdown, its tables, coverage, posets and searches from the session
```

## The Paragraphs

`chunk_html` cuts a paper's html into paragraphs, keeping math as its LaTeX alttext and a table as its caption and rows, since results live there.
`chunk_pdf` falls back to `pdftotext` for a paper with no usable html, in windows cut at blank lines.
`rank` fuses cosine and BM25 to pick the paragraphs a question needs, and `around` adds their neighbors.
A quote is checked against the one paragraph it came from, so a paragraph id is the unit of evidence.

## The Check

A report is an object: a question, a TL;DR, overview blocks, sections of blocks and further reading.
A block is a finding, an inference, an open area, or a table, coverage or poset placed by its id.
`check` runs on every save, and a report passes only when it finds nothing.

| Fault | When |
| --- | --- |
| `unknown marker` | A label no reply gave |
| `cites nothing` | A finding or TL;DR with no marker, or an open area stating a number with none |
| `number not in its paragraph` | A number that no paragraph behind the block's markers holds |
| `quote not in its paragraph` | A quoted phrase that none of the block's paragraphs holds |
| `name not in the papers cited` | A name that neither a cited paper's full text nor a read paper's title or abstract holds |
| `paper named but never read` | An arXiv id in a block's text that the reader never went through |
| `paper never returned by a tool` | An arXiv id in the further reading that no tool returned |
| `search never run` | An open area naming a search the session never made |
| `nearest hits never read` | An open area whose search's three nearest papers were none of them read |
| `names a paper without its marker` | A TL;DR naming a paper the session holds a marker of, without citing one |
| `no such table`, `no such poset` | A placed id the session lacks |
| `papers not in the poset` | A poset block drawing a paper its poset does not hold |

`contract` adds one fault to a deep report missing a table, its coverage, a poset, three findings citing the poset's sentences, or an open area, at the path of a new section.
`patch` applies a writer's fix by path, so a failed check costs only the fields it named.
`render` draws every table and poset the session made, the unplaced ones under Also from this study, and the further reading from the store.
It puts each marker beside the name of the paper it quotes when the text names it, and at the end of the sentence otherwise.
