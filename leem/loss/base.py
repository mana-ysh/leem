import torch
from torch import nn


class TripletLossBase(nn.Module):
    def compute_loss(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class PairwiseLossBase(nn.Module):
    def compute_loss(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
