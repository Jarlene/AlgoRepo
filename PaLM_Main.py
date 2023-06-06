import gzip
import numpy as np
import torch
from models.palm.palm import PaLM
from models.wrap import AutoDateSet, Wrap, get_trainer, get_train_args
from torch.utils.data import Dataset

NUM_BATCHES = int(1e5)
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        x, y = full_seq[:-1], full_seq[1:]
        return {'x': x, 'y': y}

    def __len__(self):
        return self.data.size(0) // self.seq_len


def get_data(path):
    with gzip.open(path) as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

    return data_train, data_val


def get_dataSets(path, args):
    train, val = get_data(path=path)
    train_dataset = TextSamplerDataset(train, SEQ_LEN)
    val_dataset = TextSamplerDataset(val, SEQ_LEN)
    dataset = AutoDateSet([train_dataset, val_dataset],
                          train_batch_size=args.batch_size, val_batch_size=args.batch_size)
    return dataset


def get_model(args):
    palm = PaLM(dim=args.dim, num_tokens=args.num_tokens, depth=args.depth,
                dim_head=args.dim_head, heads=args.heads)
    return palm


def get_args():
    args = get_train_args()
    args.dim = 512
    args.depth = 8
    args.label = 'y'
    args.num_tokens = 10000
    args.dim_head = 64
    args.metric = True
    args.heads = 8
    return args


if __name__ == '__main__':
    args = get_args()
    path = '/home/jarlene/Code/Projects/Experiment/data/enwik8.gz'
    data = get_dataSets(path, args)
    model = get_model(args)
    trainer = get_trainer(args)
    if trainer.is_global_zero:
        print(args)
        print(
            "-------------data size: {0}--------------".format(len(data.train_dataset)))

    example = data.train_dataset[0]['x'].unsqueeze(0)
    wrap = Wrap(model, args, example)
    trainer.fit(model=wrap, datamodule=data,
                ckpt_path='last' if args.resume else None)
