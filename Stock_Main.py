import os
import torch
import numpy as np
from dataset.stock import StockDateset, preprocess
from models.wrap import AutoDateSet, Wrap, train, get_train_args
from models.fin.fint import FintModel


def get_model(args):
    model = FintModel(args)
    return model


def get_data(args):
    file = args.train_file
    data = torch.load(file)
    train_dataset = StockDateset(data, max_length=args.max_length)
    return train_dataset


def get_args():
    args = get_train_args()
    args.label = 'y'
    args.name = 'stock'
    args.train_file = 'data/trade.pt'
    args.max_length = 60
    args.hidden_size = 128
    args.embd_pdrop = 0.5
    args.num_hidden_layers = 12
    args.ff_pdrop = 0.5
    args.resid_pdrop = 0.5
    args.attn_pdrop = 0.5
    args.num_heads = 8
    print(args)
    return args


def collate(batch):
    res_x = []
    res_y = []
    for b in batch:
        res_x.append(b['x'])
        res_y.append(b['y'])
    x = torch.stack(res_x)
    y = torch.stack(res_y)
    return {"x": x, "y": y}


def main():
    args = get_args()
    train_dataset = get_data(args)
    data = AutoDateSet(train_dataset=train_dataset,
                       train_batch_size=args.batch_size,
                       val_batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       pin_memory=args.pin_memory,
                       collate_fn=collate)
    model = get_model(args)
    example = train_dataset[0]['x'].unsqueeze(0)
    wrap = Wrap(model, args, example)

    train(args, wrap, data)


if __name__ == "__main__":
    main()
