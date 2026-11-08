A researcher is looking for one specific paper and described it like this:

{description}

Candidates:

{candidates}

For each candidate, decide whether it is that very paper, and how much of what the description says
about it the candidate's title and abstract state: all, most, or few.
Being on the same topic, by the same group, or a follow-up does not make it the paper.
Then give the number of the candidate that is the paper, or 0 when none is.

Reply with json only, each why at most eight words:
{{"candidates": {{"1": {{"same_paper": false, "stated": "few", "why": "..."}}}},
  "match": <number or 0>}}
