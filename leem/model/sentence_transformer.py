from typing import List

import torch
from sentence_transformers import SentenceTransformer

from leem.model import base


class SentenceTransformerBasedEncoder(base.TextEncoderBase):
    def __init__(self, model_name_or_path: str) -> None:
        super(SentenceTransformerBasedEncoder, self).__init__()
        self._model = SentenceTransformer(model_name_or_path)

    def forward(self, texts: List[str]) -> torch.Tensor:
        features = self._model.tokenize(texts)
        embeddings = self._model.forward(features)
        return embeddings["sentence_embedding"]
