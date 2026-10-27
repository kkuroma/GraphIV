"""
SciNCL over the whole corpus, loaded and pooled exactly as the malteos/scincl model card does

SciNCL has no query encoder of its own, so a question is encoded as the card encodes any text
"""

from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src import common
from src.methods.specter2 import BATCH, embed_corpus, search
from src.schema import Case, Method, Ranking


class Encoder:
    """
    The card's tokenizer and model, called on a list of texts
    """

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("malteos/scincl")
        self.model = AutoModel.from_pretrained("malteos/scincl").half().cuda().eval()

    @torch.inference_mode()
    def __call__(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), BATCH):
            inputs = self.tokenizer(texts[i : i + BATCH], padding=True, truncation=True, return_tensors="pt", max_length=512)
            result = self.model(**{k: v.cuda() for k, v in inputs.items()})
            out.append(torch.nn.functional.normalize(result.last_hidden_state[:, 0, :].float(), p=2, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float16)


class SciNCL(Method):
    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        shards = common.INDEX / "scincl"
        embed_corpus(Encoder, lambda e: e.tokenizer.sep_token, shards)
        queries = Encoder()([c["query"] for c in cases])
        torch.cuda.empty_cache()
        return search(cases, queries, shards)


METHOD = SciNCL
