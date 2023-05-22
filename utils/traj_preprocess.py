from dataset.nuplan import get_scenario, processs
import os
import torch

TRAIN_NUPLAN_DB_FILES = 'path/to/db/files'
SAVE_DIR = 'path/to/save'


def main(max_length, max_agent):
    train_scenarios = get_scenario(TRAIN_NUPLAN_DB_FILES)
    processs(train_scenarios[:100], max_length,
             max_agent, SAVE_DIR, num_workers=20)


def merge():
    files = os.listdir(SAVE_DIR)
    pt_files = []
    for f in files:
        if f.endswith('.pt'):
            pt_files.append(os.path.join(SAVE_DIR, f))
    print(pt_files)
    res = {}
    for f in pt_files:
        data = torch.load(f)
        for k, s in data.items():
            if k in res.keys():
                res[k].extend(s)
            else:
                res[k] = s
    torch.save(res, os.path.join(SAVE_DIR, 'traj.pt'))


if __name__ == "__main__":
    main(100, 80)
