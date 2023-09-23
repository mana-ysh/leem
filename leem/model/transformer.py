from typing import Any, List

import torch
from transformers import AutoTokenizer, AutoModel

from leem.model import base


class TransformerBasedEncoder(base.TextEncoderBase):
    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
        super(TransformerBasedEncoder, self).__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **kwargs
        )
        self._model = AutoModel.from_pretrained(
            model_name_or_path, **kwargs
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        ret = self._model(
            **self._tokenizer(texts, return_tensors="pt", padding=True)
        )
        return ret.last_hidden_state[:, 0, :]
