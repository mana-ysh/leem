import pydantic
import torch


class TripletMetricsBase(pydantic.BaseModel):
    def update_batch(
        self,
        anchor_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: torch.Tensor,
    ) -> None:
        raise NotImplementedError
