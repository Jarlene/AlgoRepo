import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


def preprocess(path, threadhold=90):
    data = pd.read_csv(path, sep='\t')
    trade_date_lenth = 6
    cols_len = len(data.columns) - 1
    trade_num = int(cols_len/trade_date_lenth)
    dataes = [data.iloc[:, 1+i*trade_date_lenth:trade_date_lenth *
                        (i+1) + 1] for i in range(trade_num)]
    for d in dataes:
        d.columns = ['open', 'high', 'low',
                     'close', 'adj.close', 'volume']

    val_datas = []
    for d in dataes:
        d = d.dropna(subset=['open'])
        if len(d) >= threadhold:
            val_datas.append(d)

    res = [i.to_numpy() for i in val_datas]
    return res


class StockDateset(Dataset):
    def __init__(self, data: list, max_length=60) -> None:
        super().__init__()
        self.max_length = max_length

        self.idxs = [np.random.randint(
            0, len(i) - max_length, (1,)) for i in data]
        self.data_len = len(self.idxs)
        self.x = []
        self.y = []
        for i, d in enumerate(data):
            rand_start = self.idxs[i].item()
            data = d[rand_start: rand_start +
                     self.max_length + 1] - d[rand_start]
            x_sub = data[:-1]
            y_sub = data[1:]
            self.x.append(x_sub)
            self.y.append(y_sub)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        x, y = self.x[index], self.y[index]
        x = torch.from_numpy(x).float()[:, :-1]
        y = torch.from_numpy(y).float()[:, :-1]
        return {'x': x, 'y': y}
