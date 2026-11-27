---
name: graphiv-lineages
description: Map how a field developed and where it argues, as a poset of citations that each carry an intent. Use when the question asks how something evolved, what it builds on, where an idea came from, whether a result holds up, or where a field disagrees.
---

# The poset of a field

1. Call `trace_lineages` with `query`, a short phrase for the field in the question's own words, like "state space models as an alternative to attention for language modeling".
   Put a paper the question names in `past`, and papers you know are recent in `present`.
   The server searches the phrase in survey mode for the past and in focused mode over the last two years for the present, adds every paper you read, and adds the papers outside both that argue with several of them.
2. The reply is a poset, an id like P1, where every edge is a citation that extends, critiques or compares against an earlier paper, with the citing paper's own sentence under a marker. It shows:
   - the longest chain of work, oldest first, which is how the field got here;
   - the papers argued with most, with what later papers say of each, which is where the debates are;
   - the newest papers and the position each takes on earlier work, which is where the field stands now.
3. Read the papers argued with most that you have not read, since a finding about what a paper claims cites the paper itself.
4. Write the debates into the sections they belong to:
   - who established a claim, and on what evidence, citing their read;
   - who critiques it and on what grounds, citing the critique's sentence from the poset;
   - who extends or compares against it and what they found;
   - which side the newest papers take.
   A claim the poset shows critiqued is never written as settled.
5. Place the poset where a section discusses it, with {"type": "lineage", "poset": "P1", "papers": [the 4 to 12 ids that section discusses]}.
   The server draws those papers and the citations among them, older on the left, with each citing sentence and its marker.
   Without "papers" it draws the longest chain and the papers argued with most.
