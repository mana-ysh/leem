import torch

from leem.data import dataset as dataset_lib
from leem.data import example
from leem.loss import triplet_loss
from leem.model import sentence_transformer
from leem.trainer import trainer as trainer_lib


def test_fit():
    dataset = dataset_lib.TripletDataset(
        examples=[
            example.TripletExample(
                anchor="hello world",
                positive="positive words",
                negative="negative words",
            ),
            example.TripletExample(
                anchor="hello world2",
                positive="positive words2",
                negative="negative words2",
            ),
        ]
    )

    model = sentence_transformer.SentenceTransformerBasedEncoder(
        model_name_or_path="all-MiniLM-L6-v2"
    )
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = trainer_lib.Trainer(
        model=model,
        optimizer=optimizer,
        loss=triplet_loss.TripletLoss(),
        batch_size=24,
        num_epoch=3,
        train_dataset=dataset,
        validation_dataset=dataset,
    )

    trainer.fit()
