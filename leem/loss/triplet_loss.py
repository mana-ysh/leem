import torch
import torch.nn.functional as F

from leem.loss import base as loss_base
from leem.similarity import base as sim_base
from leem.similarity import cos


class TripletLoss(loss_base.TripletLossBase):
    def __init__(
        self, margin: float = 1.0, sim: sim_base.SimilarityBase = cos.CosineSimilarity
    ) -> None:
        self._sim = sim
        self._margin = margin

    def compute_loss(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ):
        anchor_pos_dist = -1 * self._sim.similarity_batch(
            anchor_embeddings, positive_embeddings
        )
        anchor_neg_dist = -1 * self._sim.similarity_batch(
            anchor_embeddings, negative_embeddings
        )
        loss = F.relu(self._margin + anchor_pos_dist - anchor_neg_dist)
        return loss.mean()
