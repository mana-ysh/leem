"""Training scripts for the model."""

import argparse
import tempfile

import torch

from leem.data import msmarco
from leem.loss import triplet_loss
from leem.model import sentence_transformer
from leem.trainer import trainer as trainer_lib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the training routine.")
    p.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    p.add_argument(
        "--model_save_path",
        type=str,
        default=None,
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    p.add_argument(
        "--num_epoch",
        type=int,
        default=3,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset = msmarco.load_dataset(
        head_size=100,
        split="test",
    )
    model = sentence_transformer.SentenceTransformerBasedEncoder(
        model_name_or_path=args.model_name
    )
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = trainer_lib.Trainer(
        model=model,
        optimizer=optimizer,
        loss=triplet_loss.TripletLoss(),
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        train_dataset=dataset,
        validation_dataset=dataset,
    )

    trainer.fit()

    # save the model
    if args.model_save_path is not None:
        torch.save(model, args.model_save_path)
    else:
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model, f.name)


if __name__ == "__main__":
    main()
