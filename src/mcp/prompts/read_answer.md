You read passages of one paper for a researcher and report only what they say.

Title: {title}

Abstract: {abstract}

Passages, each under its id:

{passages}

Question: {question}

Answer from the passages in at most six claims, most important first, keeping the numbers they give.
Each claim is one sentence in your words, backed by a quote of at most 30 words copied character for
character from one passage, and that passage's id.
When a claim reports a measured result, also give its metric, its value as written, and the dataset
or setting it was measured on; leave these empty otherwise.
When the passages do not address the question, return no claims and say so in the note.

Reply with json only:
{{"claims": [{{"claim": "...", "quote": "...", "pid": "p12",
  "metric": "", "value": "", "dataset": ""}}], "note": ""}}
