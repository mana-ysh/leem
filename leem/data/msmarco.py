from typing import Optional

import datasets

from leem.data import dataset as dataset_lib
from leem.data import example


def load_dataset(
    head_size: Optional[int] = None,
    split: Optional[str] = None,
) -> dataset_lib.TripletDataset:
    dataset = datasets.load_dataset("ms_marco", "v1.1", split=split)
    examples = []

    # NOTE: generate one triple per document, but we can generate more triples.
    for ex in dataset:
        query = ex["query"]
        positive = None
        negative = None
        for idx in range(len(ex["passages"]["passage_text"])):
            if ex["passages"]["is_selected"][idx] == 1:
                positive = ex["passages"]["passage_text"][idx]
            else:
                negative = ex["passages"]["passage_text"][idx]
            if positive is not None and negative is not None:
                break

        if positive is None or negative is None:
            continue

        examples.append(
            example.TripletExample(
                anchor=query,
                positive=positive,
                negative=negative,
            )
        )
        if len(examples) == head_size:
            break

    return dataset_lib.TripletDataset(examples)
