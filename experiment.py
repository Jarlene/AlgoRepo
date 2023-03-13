import os
from argparse import Namespace
import argparse
import numpy as np
from typing import List, Union, Any, Optional, Dict
import torch
from torch import optim
from models.BaseModel import Base
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, TQDMProgressBar


class Experiment(pl.LightningModule):

    def __init__(self,
                 model: Base,
                 args: Namespace) -> None:
        super(Experiment, self).__init__()
        self.save_hyperparameters(args)
        self.model = model
        self.args = args

    def forward(self, **kwargs) -> torch.Tensor:
        return self.model(kwargs)

    def configure_example_input(self, data):
        self._example_input_array = self.prepare_inputs(data)

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

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        inputs = self.prepare_inputs(batch)
        if self._example_input_array is None:
            self._example_input_array = inputs
        train_loss = self.model.loss(**inputs)
        self.log(name='train_loss', value=train_loss, sync_dist=True)
        if self.args.metrics is not None:
            metrics = self.model.metric(**inputs)
            for k, v in metrics.items():
                self.log(name="train_" + k, value=v, sync_dist=True)
        return train_loss

    def training_epoch_end(self, training_step_outputs):
        metrics = self.model.reset()
        for k, v in metrics.items():
            self.log(name="train_epoch_" + k, value=v, sync_dist=True)

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        inputs = self.prepare_inputs(batch)
        val_loss = self.model.loss(**inputs)
        self.log(name='val_loss', value=val_loss, sync_dist=True)
        if self.args.metrics is not None:
            metrics = self.model.metric(**inputs)
            for k, v in metrics.items():
                self.log(name="val_" + k, value=v, sync_dist=True)
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        metrics = self.model.reset()
        for k, v in metrics.items():
            self.log(name="val_epoch_" + k, value=v, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.lr,
                               weight_decay=self.args.weight_decay)

        return {'optimizer': optimizer}

    def lr_schedulers(self):
        optimizer = self.configure_optimizers()['optimizer']
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=self.args.scheduler_gamma)
        return {'scheduler': scheduler}


class AutoDateSet(pl.LightningDataModule):
    def __init__(
        self,
        dataset: List[Dataset],
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
        self.dataset = dataset
        self.collate_fn = collate_fn

    def setup(self, stage: Optional[str] = None) -> None:
        if len(self.dataset) == 1:
            self.train_dataset = self.dataset[0]
            self.val_dataset = self.dataset[0]
            self.test_dataset = self.dataset[0]
        elif len(self.dataset) == 2:
            self.train_dataset = self.dataset[0]
            self.val_dataset = self.dataset[1]
            self.test_dataset = self.dataset[1]
        elif len(self.dataset) == 3:
            self.train_dataset = self.dataset[0]
            self.val_dataset = self.dataset[1]
            self.test_dataset = self.dataset[2]

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

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )


def train(args, experiment: Experiment, data: AutoDateSet):
    if torch.cuda.device_count() > 0:
        args.gpus = [i for i in range(torch.cuda.device_count())]
    logger = TensorBoardLogger(
        save_dir=args.log_dir, log_graph=True, name=args.name, version=args.version)
    runner = pl.Trainer(logger=logger,
                        callbacks=[
                            LearningRateMonitor(),
                            ModelCheckpoint(save_top_k=2,
                                            dirpath=os.path.join(
                                                args.save_dir, "checkpoints"),
                                            monitor=args.monitor,
                                            save_last=True),
                            EarlyStopping(monitor=args.monitor),
                            TQDMProgressBar()],
                        gpus=args.gpus,
                        max_epochs=args.num_epochs)
    runner.fit(model=experiment, datamodule=data)

    return runner


def get_train_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('train_base_args')
    group.add_argument("--batch_size", type=int, default=64)
    group.add_argument("--lr", type=float, default=0.001)
    group.add_argument("--weight_decay", type=float, default=0.0)
    group.add_argument("--use_kl_loss", action="store_true")
    group.add_argument("--use_cl_loss", action="store_true")
    group.add_argument("--save_dir", type=str,  default='save')
    group.add_argument("--log_dir", type=str,  default='logs')
    group.add_argument("--name", type=str,  default='tensorboard')
    group.add_argument("--version", type=str,  default='v1')
    group.add_argument("--monitor", type=str, default='val_loss')
    group.add_argument("--scheduler_gamma", type=float, default=0.99)
    group.add_argument("--num_epochs", type=int, default=100)
    group.add_argument("--num_workers", type=int, default=2)
    group.add_argument("--pin_memory", action="store_true")
    group.add_argument("--gpus", type=int, nargs='+', default=None)
    group.add_argument("--find_unused_parameters", action="store_true")
    group.add_argument("--label", type=str, default=None)
    group.add_argument("--threshold", type=float, default=0.5)
    group.add_argument("--temperature", type=float, default=0.9)
    group.add_argument("--debiased", action="store_true")
    group.add_argument("--tau_plus", type=float, default=0.1)
    args, unknow = parser.parse_known_args()
    return args
