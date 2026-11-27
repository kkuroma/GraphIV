---
name: graphiv-methods
description: Compare the methods or benchmarks of a field side by side in a table filled from the papers' full text. Use when the question asks how approaches differ, which works best, or what benchmarks exist.
---

# Comparing methods

1. Search, then pick 6 to 12 papers that each introduce a method, a tool or a benchmark.
2. Design the table once, before reading, from the titles and abstracts, as fields each paper is asked:
   - 1 or 2 fields of type "choice", each an axis the papers genuinely differ on, with 2 to 6 short options.
     One of them names the kind of approach, so it can be the rows of a coverage block.
     Include the plain option a paper doing nothing special on that axis would pick.
   - 2 to 4 fields of type "bool", each asking whether a paper has one kind of evidence or capability, such as "Does it run a user study?" or "Does it test across several judge models?".
     These become the columns of the coverage block, so pick ones whose absence would matter.
   - Exactly one field of type "float": the headline number, with its metric and benchmark named in the ask.
   - At most one "text" for a name or short phrase, and at most one "sentences" for what the paper says fails.
3. Call `read_paper` with the `schema` on the first batch; the reply names its table, like Table T1, and every later batch passes schema="T1" so all rows land in that table.

```json
{"fields": [
  {"name": "approach", "type": "choice", "ask": "How does the paper set evaluation criteria?", "options": ["human writes them", "human grades outputs", "model proposes them", "learned from labels"]},
  {"name": "user_study", "type": "bool", "ask": "Does it run a study with human participants?"},
  {"name": "cross_model", "type": "bool", "ask": "Does it test more than one judge model?"},
  {"name": "agreement", "type": "float", "ask": "Agreement with human grades, as the paper reports it"}
]}
```

   An answer the paper does not state comes back as not stated, which is a gap in what was read rather than a no.

4. In the report, place {"type": "table", "table": "T1"} and {"type": "coverage", "table": "T1", "rows": "approach", "columns": ["user_study", "cross_model"]}, and say in findings what they show, citing the answers' markers.
5. Zero cells of the coverage block are candidates for open areas; the skill graphiv-open-areas checks them before you call them open.
