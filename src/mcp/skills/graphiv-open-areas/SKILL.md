---
name: graphiv-open-areas
description: Find what a field has not done or not settled, and check each candidate before calling it open. Use when the question asks for gaps, open problems, future work, or whether something has been tried.
---

# Finding open areas

An absence in what you read is not an absence in the field, so every candidate is checked before it is called open.

1. Collect candidates from four places:
   - limitations and future work the reads state: ask "what do the authors say is unsolved, limited or left for future work".
   - disagreement: `list_citations` on a central paper, direction cited_by, intents critiques, shows who disputes it and why.
   - zero cells of a coverage block: an approach no paper tested with some kind of evidence.
   - untried combinations: two options of a table no paper combines.
2. Check each candidate with a `search_papers` call in focused mode that names the problem directly, with a date range starting a year before the newest paper you read.
   Read the one or two closest hits with the question "does this paper address <candidate>".
3. A candidate the check found nothing on is an open block:
   {"type": "open", "text": "what is open and why it matters", "search": "<the exact words of that search>", "cites": [markers of the papers that leave it open]}.
   The server checks the search was run and prints what it returned beside the block.
   A candidate the check found work on is not open: write findings about that work instead.
4. Prefer open areas a researcher would act on: specific, tied to the papers that leave them open, and saying what a first study would test.
