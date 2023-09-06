import torch

from leem.eval import metrics
from leem.similarity import base, cos


class TripletAccuracy(metrics.TripletMetricsBase):
    num_sample: int = 0
    num_correct: int = 0
    accuracy: float = -1
    similariy: base.SimilarityBase = cos.CosineSimilarity()

    class Config:
        arbitrary_types_allowed = True

    def update_batch(
        self,
        anchor_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: torch.Tensor,
    ) -> None:
        anchor_pos_sim = self.similariy.similarity_batch(
            anchor_embeddings, pos_embeddings
        )
        anchor_neg_sim = self.similariy.similarity_batch(
            anchor_embeddings, neg_embeddings
        )
        _num_correct = int(sum(anchor_pos_sim > anchor_neg_sim))

        self.num_sample += len(anchor_embeddings)
        self.num_correct += _num_correct
        self.accuracy = self.num_correct / self.num_sample
