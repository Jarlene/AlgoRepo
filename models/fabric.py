
import os
import argparse

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins import TorchCheckpointIO
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelSummary,
    TQDMProgressBar,
    DeviceStatsMonitor
)
from lightning import Fabric
from models.base import Base


def get_train_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('train_base_args')
    group.add_argument("--batch_size", type=int, default=2)
    group.add_argument("--lr", type=float, default=0.000001)
    group.add_argument("--weight_decay", type=float, default=0.0)
    group.add_argument("--resume", action="store_true")
    group.add_argument("--split_head", type=bool, default=True)
    group.add_argument("--save_dir", type=str,  default='save')
    group.add_argument("--devices", default='auto')
    group.add_argument("--log_dir", type=str,  default='logs')
    group.add_argument("--name", type=str,  default='tensorboard')
    group.add_argument("--version", type=str,  default='v1')
    group.add_argument("--monitor", type=str, default='val_loss')
    group.add_argument("--num_epochs", type=int, default=1000)
    group.add_argument("--num_workers", type=int, default=8)
    group.add_argument("--pin_memory",  type=bool, default=True)
    group.add_argument("--label", type=str, default=None)
    args, unknow = parser.parse_known_args()
    return args


def train(args, model: Base, optimizer, train_dataloader, val_dataloader=None):
    logger = TensorBoardLogger(
        save_dir=args.log_dir, log_graph=True, name=args.name, version=args.version)
    callbacks = [
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
    Fabric.seed_everything(42)
    fabric = Fabric(loggers=logger, callbacks=callbacks, accelerator='cuda', strategy='deepspeed',
                    plugins=TorchCheckpointIO(), devices=args.devices)
    fabric.launch()
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    if val_dataloader is not None:
        val_dataloader = fabric.setup_dataloaders(val_dataloader)

    if args.num_epochs == -1:
        args.num_epochs = 10000

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.loss(**batch)
            metrics = model.metric(**batch)
            fabric.backward(loss)
            optimizer.step()
            fabric.print(
                f"{step}/{epoch}| Train step Loss: {loss.detach()}")
            results = {}
            results['train_loss'] = loss
            for k, v in metrics.items():
                results['train_' + k] = v
            fabric.log_dict(results, step=step)

        if val_dataloader is not None:
            model.eval()
            for step, batch in enumerate(train_dataloader):
                loss = model.loss(**batch)
                metrics = model.metric(**batch)
                results = {}
                results['val_loss'] = loss
                for k, v in metrics.items():
                    results['val_' + k] = v
                fabric.log_dict(results, step=step)
