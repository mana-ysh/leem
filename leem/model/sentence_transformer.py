from typing import List

import torch
from sentence_transformers import SentenceTransformer

from leem.models import base


class SentenceTransformerBasedEncoder(base.TextEncoderBase):
    def __init__(self, model_name_or_path: str) -> None:
        self._model = SentenceTransformer(model_name_or_path)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self._model.encode(texts)
