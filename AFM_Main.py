
from wrap import Wrap, AutoDateSet, train, get_train_args
from models.rec.AFM import AFM

from dataset.criteo import CriteoDataset
from torch.utils.data import random_split


def get_model(args):
    model = AFM(args)
    return model


def get_dataset(args):
    dataset = CriteoDataset(
        dataset_path=args.dataset_paths[0])
    train_length = int(len(dataset) * 0.9)
    valid_length = len(dataset) - train_length
    train_dataset, valid_dataset = random_split(
        dataset, (train_length, valid_length))
    test_dataset = CriteoDataset(
        dataset_path=args.dataset_paths[1])
    return train_dataset, valid_dataset, test_dataset, dataset.field_dims


def main():
    args = get_train_args()
    args.metrics = ['auc', 'precision', 'recall', 'acc']
    args.attn_size = 16
    args.embed_dim = 8
    args.dropouts = (0.2, 0.2)
    args.label = 'y'
    args.dataset_paths = ['data/criteo/train.txt', 'data/criteo/test.txt']
    train_dataset, valid_dataset, test_dataset, field_dims = get_dataset(args)
    args.field_dims = field_dims
    model = get_model(args)

    model = Wrap(model=model, args=args)
    data = AutoDateSet(train_dataset=train_dataset, val_dataset=valid_dataset, test_dataset=test_dataset, train_batch_size=args.batch_size,
                       val_batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    train(args, model, data)


if __name__ == "__main__":
    main()
