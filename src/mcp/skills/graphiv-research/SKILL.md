---
name: graphiv-research
description: Answer a research question from the arXiv corpus with the graphiv tools and save a cited literature report. Use for any question about papers, methods, results or a research field.
---

# Research with graphiv

You answer research questions from about 3 million arXiv papers, using only the graphiv tools.
Never answer from memory, and never cite a paper you did not read with `read_paper`.

## Effort

The user's message names an effort, quick when it names none.

| effort | time | searches | papers read | report |
| --- | --- | --- | --- | --- |
| quick | 3 to 5 minutes | 1 or 2 | 4 to 6 | the answer and 4 to 8 cited findings |
| standard | about 10 minutes | 2 to 4 | 8 to 12 | sections, plus what the question asks for below |
| deep | at most 20 minutes | up to 6 | 12 to 20 | sections, a comparison table, lineages, and what is still open |

Stop gathering when you reach the effort's limits, even if more could be read.

## Steps

1. Split the question into the one to three things it actually asks, and pick a mode for each search:
   focused for a specific or recent question, survey for what a field is built on, frontier for what it works on now, consensus for what its reviews agree on.
2. Search with `search_papers`, twice with different wordings for each part.
   Write the query as a phrase describing what the papers do, like "fine-tuning on self-generated reasoning kept only when the answer is correct".
   Leave out words every paper in the area shares, such as "language model", "LLM", "deep learning" or "method", since they pull in famous papers on anything.
   If the question names a method or paper, look it up with `recall_paper` first.
3. Pick the papers to read from the titles and abstracts: the ones that test the question directly, the baselines they compare against, and at most one survey.
4. Read with `read_paper`, four papers per call, with one specific question per call, such as "what data, what metric, what number against which baseline, what fails".
5. Follow what the reads point to: a named baseline or rival goes to `recall_paper`, and who took a position on a paper goes to `list_citations` with direction cited_by and intents critiques, extends, compares.
6. Write the report into `save_report`, never into the chat. If it reports faults, fix exactly those lines and call it again, at most twice.
   The run is finished only when `save_report` has been called; after it, reply with one line saying where the report was saved.

## What the question asks for

Match the question to these before you search, and do every one that applies.
Standard effort does the ones the question asks for, and deep does all three of the last.

**A person's work.** Never put a name in a query. Call `search_papers` with no query and the filter {"authors": ["Name as on arXiv"], "date_range": ["YYYY-MM-DD", null]} to list their papers, then group and read them; two people can share a name, so judge by coauthors and topics.

**A list of papers.** The list is the answer: search with three or more wordings at limit 50, read 4 to 6 papers to confirm what they do, and list every relevant hit under Further reading.

**How methods compare, or which works best.** Design one table before reading, then pass the same `schema` to every `read_paper` batch of the papers that introduce a method:
2 to 4 columns of kind "choice", each an axis the methods differ on with 2 to 6 short options and one of them on the axis the question turns on, plus one column of kind "value" for the headline number and what it was measured on.
The server adds the table to the report, so say in prose what it shows. The skill graphiv-methods has the details.

**How a field developed, or where an idea came from.** Search the field at limit 40 to 50 in survey mode, and pass every id from that search to `trace_lineages` with the question.
Read the first paper of each line and the ones its citing sentences say extend or critique the one before, then describe each line in a few cited sentences, referring to it by number, like Line 1.2. The skill graphiv-lineages has the details.

**What is still open or untested.** Collect candidates from the limitations the reads state, from `list_citations` critiques of a central paper, and from option pairs no method in a table combines.
Check each with a focused search naming it, and read the closest one or two hits; only a candidate the check found nothing on is open, written as an Inference line quoting the search words exactly, like: a focused search for "graph rerankers for code retrieval" found none. The server checks that search was run. The skill graphiv-open-areas has the details.

## The report

The report opens on a TL;DR paragraph whenever one sentence can answer the question: "TL;DR: ", then one sentence of at most 40 words that answers it, cites its strongest evidence, and says whether any paper tests the question directly.
For a list of papers or a history of a field, the TL;DR names the two or three strongest papers, like "Popular approaches include X [A1], Y [B2] and Z [C3]", and the body holds the rest.

```
# <the question>

TL;DR: <the answer in one sentence, with its markers>

## <a heading you choose>

<findings, one per line, each ending in its marker>

## Further reading

- <arxiv id>, one a line, for papers you found but did not read; the server adds each title
```

- A TL;DR always cites. Never drop or soften it to pass the check; cite the read that supports it.
- Every finding ends with the marker a reply gave it, like [Karpukhin7], or several like [Chen3, Karpukhin7]. Never invent a marker.
- `read_paper` claims, `list_citations` sentences and `trace_lineages` steps all carry markers. Cite a step of a line with its own marker, and a claim with the read that states it, never a marker from the same paper that says something else.
- Write to explain, not to list. The server builds a results table from every measured result the reads gave, so the prose compares approaches, says which works best on what and why, and names what is unsettled.
- Group findings by approach or idea, drawing on many of the papers read, never one paragraph per paper.
- Use a number in prose only where a comparison needs it, at most one or two a paragraph, always with what it was measured on.
- A line of your own reasoning is its own line, starts with "Inference:" and carries no marker.
- A sentence of 12 words or more counts as a finding and needs a marker, so keep connecting sentences shorter than that.
- Do not quote: the marker carries the quote into the saved report. Write the finding in your own words, keeping its numbers.
- Comparison tables and lineages from your session are added to the saved report by the server. Refer to them in a sentence, never retype them.
- Write in plain sentences. No filler, no restating the question.
