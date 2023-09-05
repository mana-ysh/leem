import torch
import torch.nn.functional as F

from leem.similarity import base


class CosineSimilarity(base.SimilarityBase):
    def similarity_batch(
        self, batch_xs: torch.Tensor, batch_ys: torch.Tensor
    ) -> torch.Tensor:
        return F.cosine_similarity(batch_xs, batch_ys)

    def pairwise_similarity_matrix(
        self, xs: torch.Tensor, ys: torch.Tensor
    ) -> torch.Tensor:
        normed_x = xs.div(torch.norm(xs, dim=1, keepdim=True))
        normed_y = ys.div(torch.norm(ys, dim=1, keepdim=True))
        return torch.mm(normed_x, normed_y.t())
