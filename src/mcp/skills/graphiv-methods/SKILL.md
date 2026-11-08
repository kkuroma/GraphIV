---
name: graphiv-methods
description: Compare the methods or benchmarks of a field side by side in a table filled from the papers' full text. Use when the question asks how approaches differ, which works best, or what benchmarks exist.
---

# Comparing methods

1. Search, then pick 6 to 12 papers that each introduce a method or a benchmark.
2. Design the table once, before reading, from the titles and abstracts:
   - 2 to 4 columns of kind "choice", each an axis the methods genuinely differ on, with 2 to 6 short options.
     One choice column must be the axis the question turns on, so reading it top to bottom answers the question.
     Include the plain option a method doing nothing special on that axis would pick.
   - Exactly one column of kind "value": the headline number and what it was measured on.
   - At most one column of kind "text", a short phrase.
3. Call `read_paper` with the same `schema` for every batch of papers, so all rows land in one table:

```json
{"columns": [
  {"name": "retrieval_type", "kind": "choice", "description": "how candidates are found", "options": ["dense", "sparse", "hybrid", "graph"]},
  {"name": "training", "kind": "choice", "description": "what the method needs trained", "options": ["none", "fine-tuned", "pretrained from scratch"]},
  {"name": "result", "kind": "value", "description": "headline number and benchmark"}
]}
```

4. In the report, say what the table shows: which option of the question's axis dominates, and which method reports the best number on which benchmark, citing the cell markers.
   The server adds the table itself to the saved report.
5. Pairs of options no method in the table combines are candidates for open areas; the skill graphiv-open-areas checks them before you call them untested.
