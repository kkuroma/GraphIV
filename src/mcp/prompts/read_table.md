You fill one row of a comparison table from passages of one paper.

Title: {title}

Abstract: {abstract}

Passages, each under its id:

{passages}

Columns:
{columns}

Describe the method this paper contributes, the one it is titled after, on every column.
Name a second method only when the paper introduces two genuinely different ones.
A choice column must be answered with one of its options; when none fits, answer in two words of
your own, never "other".
Every cell carries a quote of at most 30 words copied character for character from one passage, and
that passage's id.
Write "n/a" with an empty quote when the passages do not say, and never guess.

Reply with json only:
{{"methods": [{{"name": "what the authors call it",
  "cells": {{"<column>": {{"value": "...", "quote": "...", "pid": "p12"}}}}}}]}}
