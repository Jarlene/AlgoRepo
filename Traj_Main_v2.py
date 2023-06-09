import os
import torch
from dataset.nuplan import NuPlanDataSetT5
from models.wrap import AutoDateSet, Wrap, get_trainer, get_train_args
from models.trajectory.traj_t5 import TrajT5Model


def get_model(args):
    model = TrajT5Model(args)
    return model


def get_data(args):
    data = torch.load(args.train_file)
    train_dataset = NuPlanDataSetT5(data)
    return train_dataset


def get_args():
    args = get_train_args()
    args.batch_size = 2
    args.hidden_size = 128
    args.name = 'traj_t5'
    args.max_length = 50
    args.d_ff = 512
    args.dropout_rate = 0.5
    args.is_decoder = True
    args.num_layers = 6
    args.num_heads = 8
    args.train_file = '/home/jarlene/Code/Projects/Experiment/data/traj/traj.pt'

    args.ego_attribs = 7
    args.agent_attribs = 8
    args.num_of_agents = 80
    args.use_lanes = False
    args.use_agents = True
    # moe
    args.use_moe = True
    args.num_experts = 100
    args.topk = 10
    args.expert_capacity = 4
    args.router_jitter_noise = 0.1
    return args


def collate(batch):
    ego_xs = []
    agent_xs = []
    ego_ys = []
    agent_ys = []
    for b in batch:
        ego_x, agent_x = b['feature']
        ego_xs.append(ego_x)
        agent_xs.append(agent_x)
        ego_y, agent_y = b['target']
        ego_ys.append(ego_y)
        agent_ys.append(agent_y)

    ego_x = torch.stack(ego_xs)
    agent_x = torch.stack(agent_xs)
    ego_y = torch.stack(ego_ys)
    agent_y = torch.stack(agent_ys)

    return {"ego": ego_x, "agents": agent_x, "y_ego": ego_y, "y_agents": agent_y}


def main():
    args = get_args()
    train_dataset = get_data(args)
    trainer = get_trainer(args)
    if trainer.is_global_zero:
        print(args)
        print(
            "-------------data size: {0}--------------".format(len(train_dataset)))
    data = AutoDateSet(train_dataset=train_dataset,
                       train_batch_size=args.batch_size,
                       val_batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=args.pin_memory,
                       collate_fn=collate)
    model = get_model(args)
    example = collate([train_dataset[0], train_dataset[1]])
    wrap = Wrap(
        model, args, (example['ego'], example['y_ego'], example['agents'], example['y_agents']))
    trainer.fit(model=wrap, datamodule=data,
                ckpt_path='last' if args.resume else None)


if __name__ == "__main__":
    main()
