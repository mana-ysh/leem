from typing import List

from torch.utils.data import Dataset

from leem.data import example


class PairwiseDataset(Dataset[example.PairwiseExample]):
    def __init__(self, examples: List[example.PairwiseExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> example.PairwiseExample:
        return self._examples[index]


class TripletDataset(Dataset[example.TripletExample]):
    def __init__(self, examples: List[example.TripletExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> example.TripletExample:
        return self._examples[index]
