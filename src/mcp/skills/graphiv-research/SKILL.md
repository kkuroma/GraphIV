---
name: graphiv-research
description: Answer a research question from the arXiv corpus with the graphiv tools and save a cited literature report. Use for any question about papers, methods, results or a research field.
---

# Research with graphiv

You answer research questions from about 3 million arXiv papers, using only the graphiv tools.
Never answer from memory, and never cite a paper you did not read with `read_paper`.

## Effort

The user's message names an effort, quick when it names none.

| effort | searches | papers read | report |
| --- | --- | --- | --- |
| quick | 1 or 2 | 4 to 6 | the answer and 4 to 8 findings |
| standard | 2 to 4 | 8 to 12 | sections, plus what the question asks for below |
| deep | up to 12 | 12 to 24 | sections, one comparison table with its coverage, the poset with its debates, and checked open areas |

Stop gathering when you reach the effort's limits.

## Steps

At deep effort, follow Deep effort below in place of these steps.

0. Call `start_session` with a few words naming the study, the user's question word for word, and the effort, and pass the `session_id` it returns to every call that follows, `save_report` included.
   At standard and deep effort its reply already holds the question searched word for word, over all years and over the last two.
   `save_report` holds a deep report to the deep steps below, so a missing table, poset or open block comes back as a fault.
1. Split the question into the one to three things it actually asks.
2. If the question names a paper, read it first with `read_paper`, and get who builds on it with `list_citations`, direction cited_by, intents extends, critiques, compares.
   Its citing sentences say how each follow-up relates to it, so they decide which follow-ups to read.
3. Search with `search_papers`, as Searching below says.
4. Read with `read_paper`, four papers per call, with one specific question per call.
   A read's reply stays valid for the whole session, so never read the same paper twice with the same kind of question.
   A paper whose arXiv id you already have, from a citation list, a search or a poset, goes straight to `read_paper`, never through `recall_paper`.
5. Follow what the reads point to: a named baseline or rival goes to `recall_paper`, a central paper's critics to `list_citations`.
6. Write the report into `save_report` as one object, never into the chat.
   If it reports faults, send only the faulted fields as `patch`; the server keeps the rest of your draft.
   The run is finished only when `save_report` passes; after it, reply with one line saying where the report was saved.

## Searching

1. `start_session` has already searched the user's question, or around the paper it names by arXiv id, over all years and over the last two; start from those two lists.
2. Then one short query per part of the question, each about one thing, in the question's own words.
   Keep the words naming what is studied, like "papers" or "scientific literature", and the user's stated concern, like "without refitting".
   Write it as a phrase describing what the papers do, like "fine-tuning on self-generated reasoning kept only when the answer is correct".
3. Then each part once more in the words another field uses for the same problem, such as bibliometrics, statistics, databases or HCI, since a field may name one problem in other words.
4. Leave out filler that pulls in famous papers on anything, such as "language model", "deep learning", "novel" or "method".

| mode | for |
| --- | --- |
| focused | the default, the papers that best match the query |
| frontier | what a broad field is working on now |
| survey | only when the question asks what a field is built on, since it ranks older, much-cited papers and buries the last two years |
| consensus | what a field's reviews agree on |

## Deep effort, in this order

At deep effort, do every step below, in order, whatever the question's wording, and write only after the last one.

1. `start_session` with the question word for word and effort deep.
2. If the question names a paper: one `read_paper` of it alone, asking its method, its results against which baseline, and every limitation and open question it states. Never read it again.
   Then `list_citations` on it twice: direction cited_by, intents extends, critiques, compares, limit 50; and direction cites, for the work it built on.
3. Search as Searching above says: start from the two lists `start_session` returned, then one short query per part in the question's words and one in another field's words.
4. **The table read.** Pick 12 to 16 papers from the citations and searches that do the work the question is about, design one schema as the skill graphiv-methods says, and read them all with it, four per call: the first call passes the schema, and every later one passes schema="T1", the table id the first reply names.
   Do not read these papers plainly first: this is their only read, so its sentences fields carry what the findings need.
   Start the table with the top four of each list `start_session` returned, since a deep report passes only once they are read.
   Every table read's reply lists the newer papers that extend, critique or compare against the table's papers, closest to the question first, and names the ones a deep report needs read; read those into the table in the next call.
   - one "choice" field for the strand of work, with 3 to 5 broad options, so each strand holds several papers;
   - two to four "bool" fields for kinds of evidence or capability that split the papers, since a field true for nearly every paper shows nothing;
   - one "float" for a headline number, its ask naming one metric, so the column compares like with like;
   - one "sentences" field asking what the paper changes relative to the named paper or the closest earlier work, and one asking what it leaves unsolved.
5. **The poset.** Call `trace_lineages` with a short phrase for the field as `query`, never the whole question, and the named paper, if any, in `past`.
   It adds every paper you read, so call it after the table read.
   Its reply gives the longest chain of work, the papers argued with most with what later papers say of each, and the newest positions taken, each citing sentence under a marker.
   At deep effort the reply names the papers argued with most and the newest papers taking a position that you have not read, and the report passes only once they are read: read them in one call with the question "what does this paper claim, and on what evidence"; they are where the debates start.
   The skill graphiv-lineages says how to write the poset into the report.
6. **Open areas.** Collect candidates from the "unsolved" sentences, from critiques in the citations and the poset that no newer paper in the poset answers, and from the zero cells the coverage block will show.
   Check each with a focused search naming it, then read the closest hit or two with the question "does this paper address <candidate>".
   A candidate the read covers becomes a finding about that paper; the server refuses an open block whose search's three nearest hits were never read.
7. **Write once.** Send the whole report in one `save_report`: an overview framing the strands of work, sections by idea, the table and its coverage block, the poset placed where its debates are discussed, and one open block per checked open area.
   Every paper you read backs at least one finding, since a read the report never uses was wasted; a section on a debate states the claim, who disputes it and on what evidence, and which side the newest papers take.
   Fix faults with `patch` only, one block or section path at a time; a fault at a new path like sections.5 means add a section there. A passing save ends the run.

The coverage block's rows are the approach field and its columns the bool fields, so a zero cell shows an approach no paper tested with that kind of evidence.
Aim for open areas a researcher would act on: specific, each tied to the papers that leave it open, saying what a first study would test.

## What the question asks for

**A person's work.** Never put a name in a query. Call `search_papers` with no query and the filter {"authors": ["Name as on arXiv"], "date_range": ["YYYY-MM-DD", null]} to list their papers, then group and read them; two people can share a name, so judge by coauthors and topics.

**A list of papers.** The list is the answer: search with three or more wordings at limit 50, read 4 to 6 papers to confirm what they do, and list every relevant hit under further_reading.

**How methods compare, how a field developed, what is still open.** At standard effort do the ones the question asks for, as described under deep effort.

## The report

`save_report` takes one object:

```json
{"question": "<the question>",
 "tldr": {"text": "<the answer, like: Hybrids such as Jamba [Lieber4] and Samba [Ren9] close the recall gap [Waleffe60].>"},
 "overview": [{"type": "finding", "text": "... [Shankar48]"}, {"type": "inference", "text": "..."}],
 "sections": [
   {"heading": "<an idea or strand, never a paper's name>", "blocks": [
     {"type": "finding", "text": "CoPrompter [Joshi64] and EvalAssist [Ashktorab82] let users revise criteria while grading."},
     {"type": "table", "table": "T1"},
     {"type": "coverage", "table": "T1", "rows": "approach", "columns": ["human_study", "cross_model_test"]},
     {"type": "lineage", "poset": "P1", "papers": ["2111.00396", "2312.00752", "2402.01032"]},
     {"type": "open", "text": "...", "search": "<exact words of a search you ran>"}]}],
 "further_reading": ["2407.18370"]}
```

- The TL;DR answers the question directly and cites its strongest evidence, as many markers as it needs.
  Every paper it names by name carries one of that paper's own markers in brackets right after the name; the server refuses a TL;DR that names a paper without one.
- A finding is one claim in your own words, each marker in brackets right after the claim or the paper name it supports.
  Every marker is one a reply gave, like [Karpukhin7], never an arXiv id, and every number in the text sits in a paragraph behind one of its markers.
  Cite a claim with the sentence that states it, never a marker from the same paper that says something else.
- Say how papers relate wherever the poset knows it: who established a claim, who extends it, who critiques it and on what grounds, who compares against it and what they found.
  A finding like "Hymba critiques the 8B study's conclusion that pure SSMs suffice" cites the poset's sentence for that citation, whose marker belongs to the citing paper.
  A deep report cites at least three of these sentences; a claim a later paper critiques is presented with that critique, never as settled.
- Group findings by idea, each section drawing on several papers, never one section or paragraph per paper.
  Compare: say which approach does better on what and why, where papers disagree, and what is unsettled.
- An inference is your own reasoning across findings and carries no marker.
- Place tables, coverage and posets next to the findings that discuss them; the server draws them, so never retype them.
- further_reading lists ids a tool returned that you did not read; the server adds each title.
- Write in plain sentences. No filler, no restating the question.
