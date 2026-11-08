---
name: graphiv-lineages
description: Trace how a field developed, which work led to which, from the citations among its papers. Use when the question asks how something evolved, what it builds on, or where an idea came from.
---

# Tracing lineages

1. Search the field with `search_papers`, limit 40 to 50, in survey mode for where it came from or frontier mode for where it is going.
2. Pass every id from that search to `trace_lineages` with the question. A handful of papers gives no lines, so always pass a whole search.
3. Each line lists papers oldest first and the citing sentences that join them. Read the two or three papers that turn a line, the first and the ones where the citing intent is extends or critiques, with the question "what did this paper change about the approach before it".
4. In the report, describe each line in a few cited sentences: where it starts, what each step changed, where it stands now.
   Refer to a line by its number, like "Line 1.2". The server adds the lines and their figures to the saved report.
