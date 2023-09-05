from typing import Optional

import datasets

from leem.data import dataset, example


def load_dataset(
    head_size: Optional[int] = None,
    split: Optional[str] = None,
) -> dataset.TripletDataset:
    dataset = datasets.load_dataset("ms_marco", "v1.1", split=split)
    if head_size is not None:
        dataset = dataset[:head_size]
    examples = []

    # NOTE: generate one triple per document, but we can generate more triples.
    for ex in dataset:
        query = ex["query"]
        positive = None
        negative = None
        for idx in range(len(example["passages"]["passage_text"])):
            if example["passages"]["is_selected"][idx] == 1:
                positive = example["passages"]["passage_text"][idx]
            else:
                negative = example["passages"]["passage_text"][idx]
            if positive is not None and negative is not None:
                break
        examples.append(example.TripletExample(query, positive, negative))
    return dataset.TripletDataset(examples)
