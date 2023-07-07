
import os
import torch
import argparse
from torch.utils.data.dataloader import DataLoader
from dataset.nuplan import NuPlanDataSetGPT
from models.trajectory.traj_gpt import TrajGPT
import datasets
from accelerate import Accelerator
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import get_scheduler
import math
import os
from pathlib import Path
import torch.utils.data.IterableDataset
logger = get_logger(__name__)


def get_last_dir_name(dir):
    paths = sorted(Path(dir).iterdir(), key=os.path.getmtime)
    return paths[-1].name


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
    group.add_argument("--checkpointing_steps", type=str, default='epoch')
    args, unknow = parser.parse_known_args()
    return args


def get_model(args):
    model = TrajGPT(args)
    return model


def get_data(args):
    train_dataset = datasets.load_from_disk(args.data_dir)['train']
    return NuPlanDataSetGPT(train_dataset, args)


def get_args():
    args = get_train_args()
    args.batch_size = 4
    args.label = 'target'
    args.name = 'traj_gpt'
    args.version = 'v3'
    args.max_length = 200
    args.hidden_size = 256
    args.ego_attribs = 7
    args.agent_attribs = 9
    args.num_of_agents = 80
    args.lanes_attribs = 10
    args.num_of_lanes = 10
    args.num_of_lane_path_point = 50

    args.embd_pdrop = 0.5
    args.num_hidden_layers = 6
    args.ff_pdrop = 0.5
    args.resid_pdrop = 0.5
    args.attn_pdrop = 0.5
    args.num_heads = 16
    args.data_dir = '/root/Code/Projects/AiPilot/training/data/data_gpt_201_data_mini'

    # accelerate 相关参数
    args.gradient_accumulation_steps = 4
    args.lr_scheduler_type = 'cosine'
    args.num_warmup_steps = 0
    return args


def collate(batch):
    ego = []
    agents = []

    lanes_path = []
    lane_arrtib = []
    ego_y = []
    agents_y = []
    for b in batch:
        ego.append(b['ego'])
        agents.append(b['agents'])
        ego_y.append(b['y_ego'])
        agents_y.append(b['y_agents'])
        lanes_path.append(b['lanes'][0])
        lane_arrtib.append(b['lanes'][1])
    ego = torch.stack(ego)
    agents = torch.stack(agents)
    lanes_path = torch.stack(lanes_path)
    lane_arrtib = torch.stack(lane_arrtib)

    ego_y = torch.stack(ego_y)
    agents_y = torch.stack(agents_y)

    return {"ego": ego, "agents": agents, 'lanes': (lanes_path, lane_arrtib), 'y_ego': ego_y, 'y_agents': agents_y}


def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='tensorboard'
    )
    accelerator.init_trackers(project_name=os.path.join(
        args.log_dir, args.name, args.version))

    set_seed(42)
    accelerator.wait_for_everyone()
    train_dataset = get_data(args)
    model = get_model(args)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    completed_steps = 0
    starting_epoch = 0
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    total_batch_size = args.batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps
    checkpoint_path = os.path.join(
        args.save_dir, "checkpoints", args.name)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  checkpoint path = {checkpoint_path}")

    if args.resume:
        accelerator.print(f"load from checkpoint: {checkpoint_path}")
        last_training = get_last_dir_name(checkpoint_path)
        accelerator.load_state(os.path.join(checkpoint_path, last_training))
        if "epoch" in last_training:
            starting_epoch = int(last_training.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(last_training.replace(
                "step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = model.module.loss(**batch)
                metrics = model.module.metric(**batch)
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                res = {}
                progress_bar.set_postfix(
                    {'train_loss':  loss.detach().float()})
                res['train_loss'] = loss.detach().float()
                for k, v in metrics.items():
                    res['train_' + k] = v.detach().float()
                accelerator.log(res, epoch * len(train_dataloader) + step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = os.path.join(
                        checkpoint_path, f"step_{completed_steps }")
                    accelerator.save_state(output_dir)

        if args.checkpointing_steps == "epoch":
            output_dir = os.path.join(
                checkpoint_path, f"epoch_{epoch}")
            accelerator.save_state(output_dir)

    accelerator.end_training()
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        torch.save(unwrapped_model, os.path.join(
            checkpoint_path, "final_model.pt"))


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
