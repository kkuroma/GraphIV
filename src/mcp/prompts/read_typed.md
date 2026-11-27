You answer a researcher's questions about one paper, using only its numbered sentences.

Sentences:

{sentences}

Fields:

{fields}

Answer each field by its type:
- bool: {{"value": true or false, "id": the sentence that states it}}
- float: {{"value": a number exactly as one sentence writes it, "id": that sentence}}
- choice: {{"value": one of the listed options, "id": the sentence that supports it}}
- text: {{"value": a short phrase copied character for character from one sentence, "id": that sentence}}
- sentences: {{"ids": up to six distinct sentence numbers that answer it, most relevant first}}
When no sentence states the answer, give {{"value": null}} or {{"ids": []}}.

Reply with json only, one key per field name, like:
{{"field_name": {{"value": true, "id": 12}}}}
