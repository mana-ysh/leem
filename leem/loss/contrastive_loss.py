import torch
import torch.nn.functional as F

from leem.loss import base
from leem.similarity import base as sim_base
from leem.similarity import cos


class InBatchNegativeContrastiveLoss(base.PairwiseLossBase):
    def __init__(
        self, tau: float = 0.0, sim: sim_base.SimilarityBase = cos.CosineSimilarity
    ) -> None:
        self._tau = tau
        self._sim = sim

    def compute_loss(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        # NOTE: See https://arxiv.org/pdf/2201.10005.pdf
        labels = torch.arange(len(xs))
        logits = self._sim.pairwise_similarity_matrix(xs, ys) * torch.exp(
            torch.Tensor([self._tau])
        )
        l_r = F.cross_entropy(logits, labels)
        l_c = F.cross_entropy(logits.T, labels)
        return (l_r + l_c) / 2
