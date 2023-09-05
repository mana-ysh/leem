import enum
import logging
from typing import Optional

import torch
from torch import optim
from torch.utils.data import DataLoader

from leem.data import dataset
from leem.eval import metrics, triplet_accuracy
from leem.loss import base as loss_base
from leem.model import base as model_base


class MetricType(enum.Enum):
    TRIPLET_ACCURACY = 0


class Trainer:
    def __init__(
        self,
        model: model_base.TextEncoderBase,
        loss: loss_base.TripletLossBase,
        optimizer: optim.Optimizer,
        batch_size: int,
        num_epoch: int,
        train_dataset: dataset.TripletDataset,
        validation_dataset: Optional[dataset.TripletDataset] = None,
        validation_metric_type: MetricType = MetricType.TRIPLET_ACCURACY,
    ) -> None:
        # FIXME: add lr_scheduler if necessary
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._num_epoch = num_epoch
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._validation_metric_type = validation_metric_type

    def fit(self):
        num_steps = 0
        for epoch in range(self._num_epoch):
            self._model.train()
            train_dataset_loader = DataLoader(
                dataset=self._train_dataset, batch_size=self._batch_size, shuffle=True
            )
            for batch in train_dataset_loader:
                num_steps += 1
                self._optimizer.zero_grad()
                batch_anc = [ex.anchor for ex in batch]
                batch_pos = [ex.positive for ex in batch]
                batch_neg = [ex.negative for ex in batch]
                anc_embeddings = self._model.forward(batch_anc)
                pos_embeddings = self._model.forward(batch_pos)
                neg_embeddings = self._model.forward(batch_neg)
                loss = self._loss.compute_loss(
                    anc_embeddings, pos_embeddings, neg_embeddings
                )
                loss.backward()
                self._optimizer.step()

                num_steps += 1
                if num_steps % 1000 == 0:
                    logging.info(
                        f"Processed {num_steps}th batch at {(epoch+1)}th epoch. loss={loss}"
                    )

            if self._validation_dataset:
                metrics = self._evaluate()
                logging.info(
                    f"Evalation metrics for validation dataset at {(epoch+1)}th epoch: {metrics}"
                )

    def _evaluate(self) -> metrics.MetricsBase:
        self._model.eval()
        validation_dataset_loader = DataLoader(
            dataset=self._validation_dataset, batch_size=self._batch_size, shuffle=True
        )
        if self._validation_metric_type == MetricType.TRIPLET_ACCURACY:
            metric = triplet_accuracy.TripletAccuracy()
        else:
            raise ValueError(
                f"Unavailable metric for validation: {self._validation_metric_type}"
            )
        with torch.no_grad():
            for batch in validation_dataset_loader:
                batch_anc = [ex.anchor for ex in batch]
                batch_pos = [ex.positive for ex in batch]
                batch_neg = [ex.negative for ex in batch]
                anc_embeddings = self._model.forward(batch_anc)
                pos_embeddings = self._model.forward(batch_pos)
                neg_embeddings = self._model.forward(batch_neg)
                metric.update_batch(anc_embeddings, pos_embeddings, neg_embeddings)
        return metric
