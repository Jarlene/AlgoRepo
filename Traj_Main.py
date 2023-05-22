import os
import torch

from dataset.nuplan import NuPlanDataSet, get_scenario
from models.wrap import AutoDateSet, Wrap, train, get_train_args
from models.trajectory.GPT import TrajGPT
from torch.utils.data import random_split


TRAIN_NUPLAN_DB_FILES = '/media/jarlene/Samsung_T5/nuPlan/dataset/nuplan-v1.1/mini/'
VAL_NUPLAN_DB_FILES = '/media/jarlene/Samsung_T5/nuPlan/dataset/train/data/cache/train_pittsburgh'


def get_model(args):
    model = TrajGPT(args)
    return model


def get_data(args):
    data = torch.load(args.train_file)
    train_dataset = NuPlanDataSet(data)
    return train_dataset


def get_args():
    args = get_train_args()
    args.batch_size = 2
    args.label = 'target'
    args.name = 'traj'
    args.max_length = 100
    args.num_of_agent = 90
    args.hidden_size = 128
    args.embd_pdrop = 0.5
    args.num_hidden_layers = 12
    args.ff_pdrop = 0.5
    args.resid_pdrop = 0.5
    args.attn_pdrop = 0.5
    args.num_heads = 8
    args.train_file = '/home/jarlene/Code/Projects/Experiment/data/test.pt'
    print(args)
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

    return {"feature": (ego_x, agent_x), "target": (ego_y, agent_y)}


def main():
    args = get_args()
    train_dataset = get_data(args)
    data = AutoDateSet(train_dataset=train_dataset,
                       train_batch_size=args.batch_size,
                       val_batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=args.pin_memory,
                       collate_fn=collate)
    model = get_model(args)
    ego_x, agent_x = train_dataset[0]['feature']
    example = (ego_x.unsqueeze(0), agent_x.unsqueeze(0))
    wrap = Wrap(model, args, example)

    train(args, wrap, data)


if __name__ == "__main__":
    main()
