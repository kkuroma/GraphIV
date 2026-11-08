---
name: graphiv-open-areas
description: Find what a field has not done or not settled, and check each candidate before calling it open. Use when the question asks for gaps, open problems, future work, or whether something has been tried.
---

# Finding open areas

An absence in what you read is not an absence in the field, so every candidate is checked before it is called open.

1. Collect candidates from three places:
   - limitations and future work: read 4 to 8 recent papers with the question "what do the authors say is unsolved, limited or left for future work".
   - disagreement: `list_citations` on a central paper, direction cited_by, intents critiques, shows who disputes it and why.
   - untried combinations: in a comparison table, two options no method combines.
2. Check each candidate with a `search_papers` call in focused mode that names the combination or problem directly, with a date range starting a year before the newest paper you read.
   Read the one or two closest hits with the question "does this paper address <candidate>".
3. Report a candidate as open only when the check found nothing, and say how it was checked, quoting the search words exactly as you sent them, for example: a focused search for "dense retrieval with graph rerankers" and reads of the two closest papers found no work combining them.
   The server checks that a search with those exact words was run.
   A candidate the check found work on is not open: report that work instead.
4. Mark each open area as your inference, since it rests on what was searched, not on a paper's claim:
   "Inference: no paper in the corpus combines X with Y; a focused search and reads of A and B found none."
