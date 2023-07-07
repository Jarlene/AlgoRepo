import os
from argparse import Namespace
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from typing import Union, Any, Dict
from transformers.training_args import TrainingArguments
from models.base import Base
from lightning.pytorch.core import LightningModule, LightningDataModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.plugins.environments import TorchElasticEnvironment
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins import TorchCheckpointIO
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelSummary,
    TQDMProgressBar
)
import transformers.trainer as trainer


class Wrap(LightningModule):

    def __init__(self,
                 model: Base,
                 args: Namespace,
                 example=None) -> None:
        super(Wrap, self).__init__()
        self.save_hyperparameters(args)
        self.model = model
        self.args = args
        self.example_input_array = example

    def prepare_inputs(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Dict):
            return type(data)({k: self.prepare_inputs(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self.prepare_inputs(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.as_tensor(data).to(self.device)
        else:
            return data

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss(self, inputs, stage='train', prog_bar=True):
        loss = -1.0
        if isinstance(inputs, torch.Tensor):
            loss = self.model.loss(inputs)
        elif isinstance(inputs, Dict):
            loss = self.model.loss(**inputs)
        elif isinstance(inputs, (tuple, list)):
            loss = self.model.loss(*inputs)
        else:
            loss = self.model.loss(inputs)
        self.log(name=stage + '_loss', value=loss,
                 sync_dist=True, prog_bar=prog_bar, logger=True)
        return loss

    def metrics(self, inputs, stage='train'):
        metrics_val = None
        if isinstance(inputs, torch.Tensor):
            metrics_val = self.model.metric(inputs)
        elif isinstance(inputs, Dict):
            metrics_val = self.model.metric(**inputs)
        elif isinstance(inputs, (tuple, list)):
            metrics_val = self.model.metric(*inputs)
        else:
            metrics_val = self.model.metric(inputs)
        for k, v in metrics_val.items():
            self.log(name=stage + '_' + k, value=v,
                     sync_dist=True, logger=True)

    def training_step(self, batch, batch_idx):
        inputs = self.prepare_inputs(batch)
        loss = self.loss(inputs, stage='train')
        self.metrics(inputs)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.prepare_inputs(batch)
        loss = self.loss(inputs, stage='val', prog_bar=False)
        self.metrics(inputs, stage='val')
        return loss

    def test_step(self, batch, batch_idx):
        inputs = self.prepare_inputs(batch)
        loss = self.loss(inputs, stage='test', prog_bar=False)
        self.metrics(inputs, stage='test')
        return loss

    def on_train_epoch_end(self):
        self.model.reset()

    def on_validation_epoch_end(self):
        self.model.reset()

    def on_test_epoch_end(self):
        self.model.reset()

    def configure_optimizers(self):
        optim = AdamW(params=self.model.parameters(),
                      lr=self.args.lr, weight_decay=self.args.weight_decay)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optim),
                "monitor": "val_loss",
            },
        }


class AutoDateSet(LightningDataModule):

    def __init__(
        self,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = False,
        collate_fn=None,
        **kwargs,
    ):
        super(AutoDateSet, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if val_dataset is None:
            train_length = int(len(train_dataset) * 0.9)
            valid_length = len(train_dataset) - train_length
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, (train_length, valid_length))
        self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                shuffle=False,
                drop_last=True,
            )
        return None

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                shuffle=False,
                drop_last=True,
            )

        return None


def get_trainer(args: Namespace):
    logger = TensorBoardLogger(
        save_dir=args.log_dir, log_graph=True, name=args.name, version=args.version)
    runner = Trainer(
        logger=logger,
        strategy=args.strategy,
        devices=args.devices,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(save_top_k=2,
                            dirpath=os.path.join(
                                args.save_dir, "checkpoints", args.name),
                            monitor=args.monitor,
                            save_last=True),
            EarlyStopping(monitor=args.monitor),
            GradientAccumulationScheduler(scheduling={2: 1}),
            TQDMProgressBar(),
            ModelSummary(2),
        ],
        plugins=[TorchCheckpointIO(), TorchElasticEnvironment()],
        # profiler=PyTorchProfiler(),
        max_epochs=args.num_epochs)
    return runner


def get_train_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('train_base_args')
    group.add_argument("--batch_size", type=int, default=2)
    group.add_argument("--lr", type=float, default=0.000001)
    group.add_argument("--weight_decay", type=float, default=0.0)
    group.add_argument("--resume", action="store_true")
    group.add_argument("--split_head", type=bool, default=True)
    group.add_argument("--save_dir", type=str,  default='save')
    group.add_argument("--strategy",  default='auto')
    group.add_argument("--devices", default='auto')
    group.add_argument("--log_dir", type=str,  default='logs')
    group.add_argument("--name", type=str,  default='tensorboard')
    group.add_argument("--version", type=str,  default='v1')
    group.add_argument("--monitor", type=str, default='val_loss')
    group.add_argument("--num_epochs", type=int, default=-1)
    group.add_argument("--num_workers", type=int, default=4)
    group.add_argument("--pin_memory", action="store_true", default=True)
    group.add_argument("--label", type=str, default=None)
    args, unknow = parser.parse_known_args()
    return args


class HTrainer(trainer.Trainer):

    def __init__(self, model: Base, args: TrainingArguments, train_dataset: Dataset, collate_fn: Any | None = None,  eval_dataset: Dataset = None, **kwargs):
        super().__init__(model=model, args=args, data_collator=collate_fn,
                         train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if hasattr(model, "module"):
            return model.module.loss(**inputs)
        else:
            return model.loss(**inputs)

    # def training_step(self, model, inputs):
    #     print(self.args.device)
    #     loss = super().training_step(model, inputs)
    #     if hasattr(model, "module"):
    #         metrics = model.module.metric(**inputs)
    #     else:
    #         metrics = model.metric(**inputs)
    #     if len(metrics) > 0:
    #         res = {}
    #         for k, v in metrics.items():
    #             res["train_" + k] = v.detach().cpu().item()
    #         self.log(res)
    #     return loss

    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
    #     result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    #     if hasattr(model, "module"):
    #         model.module.reset()
    #         metrics = model.module.metric(**inputs)
    #     else:
    #         model.reset()
    #         metrics = model.metric(**inputs)

    #     if len(metrics) > 0:
    #         res = {}
    #         for k, v in metrics.items():
    #             res["pred_"+k] = v.detach().cpu().item()
    #         self.log(res)
    #     return result


def get_transformers_trainer(args: TrainingArguments, model: nn.Module, train_dataset: Dataset, eval_dataset: Dataset = None, collate_fn=None):
    args.remove_unused_columns = False
    h_trainer = HTrainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, collate_fn=collate_fn)
    return h_trainer
