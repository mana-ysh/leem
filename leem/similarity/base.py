import torch


class SimilarityBase:
    def similarity_batch(
        self, batch_xs: torch.Tensor, batch_ys: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def pairwise_similarity_matrix(
        self, xs: torch.Tensor, ys: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
