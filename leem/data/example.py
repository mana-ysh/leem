import enum
from typing import Generic, TypeVar

import pydantic

T = TypeVar("T")


class PairwiseLabel(enum.Enum):
    """Label type for classification task."""

    UNKNOW = 0
    POSITIVE = 1
    NEGATIVE = 2


class PairwiseExample(pydantic.BaseModel, Generic[T]):
    letf: T
    right: T


class LabeledPairwiseExample(pydantic.BaseModel, Generic[T]):
    left: T
    right: T
    label: PairwiseLabel
    # FIXME: add score if necessary


class TripletExample(pydantic.BaseModel, Generic[T]):
    anchor: T
    positive: T
    negative: T
