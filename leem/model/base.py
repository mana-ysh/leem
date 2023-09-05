from typing import List

import torch
from torch import nn


class TextEncoderBase(nn.Module):
    def forward(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError
