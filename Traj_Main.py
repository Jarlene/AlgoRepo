import torch

from dataset.nuplan import NuPlanDataSetGPT
from models.wrap import AutoDateSet, Wrap, get_trainer, get_train_args, get_transformers_trainer
from models.trajectory.traj_gpt import TrajGPT
from transformers import TrainingArguments, HfArgumentParser


def get_model(args):
    model = TrajGPT(args)
    return model


def get_data(args):
    data = torch.load(args.train_file)
    train_dataset = NuPlanDataSetGPT(data)
    return train_dataset


def get_args():
    args = get_train_args()
    args.batch_size = 2
    args.name = 'traj_gpt'
    args.max_length = 100

    args.hidden_size = 128
    args.embd_pdrop = 0.5
    args.num_hidden_layers = 12
    args.ff_pdrop = 0.5
    args.resid_pdrop = 0.5
    args.attn_pdrop = 0.5
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
    example.pop('y_ego')
    example.pop('y_agents')
    wrap = Wrap(model, args, tuple(example.values()))
    trainer.fit(model=wrap, datamodule=data,
                ckpt_path='last' if args.resume else None)


def h_trainer():
    args = get_args()
    train_dataset = get_data(args)
    model = get_model(args)
    parser = HfArgumentParser(TrainingArguments)
    training_args: TrainingArguments = parser.parse_args_into_dataclasses()[0]
    training_args.remove_unused_columns = False
    trainer = get_transformers_trainer(
        training_args, model=model, train_dataset=train_dataset, collate_fn=collate)

    trainer.train()


if __name__ == "__main__":
    h_trainer()
