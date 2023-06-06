from dataset.nuplan import get_scenario, processs
import os
import json
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from collections import defaultdict

TRAIN_NUPLAN_DB_FILES = 'path/to/db/files'
SAVE_DIR = 'path/to/save'


class Quaternion(object):

    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z])
        self.normalized = False

    def norm(self):
        if not self.normalized:
            n = np.sqrt(np.dot(self.q, self.q))
            if n > 0:
                self.q = self.q / n
                self.normalized = True

    def yaw_pitch_roll(self):
        self.norm()
        yaw = np.arctan2(2 * (self.q[0] * self.q[3] - self.q[1] * self.q[2]),
                         1 - 2 * (self.q[2] ** 2 + self.q[3] ** 2))
        pitch = np.arcsin(2 * (self.q[0] * self.q[2] + self.q[3] * self.q[1]))
        roll = np.arctan2(2 * (self.q[0] * self.q[1] - self.q[2] * self.q[3]),
                          1 - 2 * (self.q[1] ** 2 + self.q[2] ** 2))

        return yaw, pitch, roll


def as_matrix(x, y, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0.0, 0.0, 1.0],
    ])


def from_matrix(matrix: np.array):
    x, y, theta = matrix[0, 2].item(), matrix[1, 2].item(
    ), np.arctan2(matrix[1, 0], matrix[0, 0]).item()
    return x, y, theta


def convert_absolute_to_relative_states(data):
    origin = data[0]
    origin_matrix = as_matrix(origin[0], origin[1], origin[2])
    origin_transform = np.linalg.inv(origin_matrix)
    res = []
    for d in data:
        res.append(d)
        x, y, theta = d[0], d[1], d[2]
        d_matrix = as_matrix(x, y, theta)
        relative_transforms = origin_transform @ d_matrix
        x, y, theta = from_matrix(relative_transforms)
        res[-1][0] = x
        res[-1][1] = y
        res[-1][2] = theta

    return res


def convert_absolute_to_relative(origin_transform, list_status):
    if len(list_status) == 0:
        return list_status
    res = []
    for s in list_status:
        res.append(s)
        x, y, theta = s[0], s[1], s[2]
        matrix = as_matrix(x, y, theta)
        relative_transforms = origin_transform @ matrix
        x, y, theta = from_matrix(relative_transforms)
        res[-1][0] = x
        res[-1][1] = y
        res[-1][2] = theta
    return res


def group_by(d: Dataset, col, join):

    groups = defaultdict(list)
    d.select([0, 1, 2]).to_list()

    def create_groups_indices(row, i):
        groups[row[col]].append(i)
    d.map(create_groups_indices, with_indices=True)

    groups = {key: d.select(indices) for key, indices in groups.items()}
    # Apply join function
    groups = {
        key: dataset_group.map(join, batched=True, batch_size=len(
            dataset_group), remove_columns=d.column_names)
        for key, dataset_group in groups.items()
    }
    # Return concatenation of all the joined groups
    return concatenate_datasets(groups.values())


def ego_agent_map_encoder(row):
    data = row['result']
    data.sort(key=lambda k: k['timestamp'])
    ego = []
    agents = []
    lanes = []
    x0 = data[0]['x']
    y0 = data[0]['y']
    qw0 = data[0]['qw']
    qx0 = data[0]['qx']
    qy0 = data[0]['qy']
    qz0 = data[0]['qz']
    q0 = Quaternion(qw0, qx0, qy0, qz0)
    heading0, _, _ = q0.yaw_pitch_roll()
    origin_matrix = as_matrix(x0, y0, heading0)
    origin_transform = np.linalg.inv(origin_matrix)
    for r in data:
        qw = r['qw']
        qx = r['qx']
        qy = r['qy']
        qz = r['qz']
        q = Quaternion(qw, qx, qy, qz)
        heading, pitch, roll = q.yaw_pitch_roll()
        x = r['x']
        y = r['y']
        vx = r['vx']
        vy = r['vy']
        ax = r['acceleration_x']
        ay = r['acceleration_y']
        timestamp = r['timestamp']
        ego.append([x, y, heading, vx, vy, ax, ay, timestamp])

        origin_matrix = as_matrix(x, y, heading)
        origin_transform = np.linalg.inv(origin_matrix)
        a = r['agents']
        one_frame_agents = []
        if a is not None:
            for aa in a:
                if aa['name'] is None:
                    break
                agent_name = aa['name']
                agent_x = aa['agent_x']
                agent_y = aa['agent_y']
                agent_heading = aa['yaw']
                agent_vx = aa['agent_vx']
                agent_vy = aa['agent_vy']
                width = aa['width']
                length = aa['length']
                token = aa['lidar_box_token'].hex()
                one_frame_agents.append(
                    [agent_x, agent_y, agent_heading, agent_vx, agent_vy, width, length, agent_name, token])

            one_frame_agents = convert_absolute_to_relative(
                origin_transform, one_frame_agents)
        agents.append(one_frame_agents)

        l = json.loads(r['lanes'])
        one_frame_lanes = []
        for ll in l:
            one_frame_lanes.append(ll)
            if not ll['is_on_routing']:
                continue
            path = ll['path']
            path = convert_absolute_to_relative(origin_transform, path)
            one_frame_lanes[-1]['path'] = path
            lanes.append(one_frame_lanes)

    ego = convert_absolute_to_relative_states(ego)
    ego_str = json.dumps(ego)
    agent_str = json.dumps(agents)
    map_str = json.dumps(lanes)
    result = {'group_key': row['group_key'], 'ego': ego_str,
              'agents': agent_str, 'maps': map_str}
    return result


def join(batch):
    return {"total": [batch["i"]]}


if __name__ == "__main__":
    DIR = '/pnc-data/data/nuplan/train_table/traj_data_v2'
    files = []
    for f in os.listdir(DIR):
        f_path = os.path.join(DIR, f)
        if f.endswith('.parquet'):
            files.append(f_path)

    data = load_dataset('parquet',
                        num_proc=50,
                        data_files=files)
    data.map(with_indices=True)
    group_data = group_by(data, 'group_key', join)
    data = group_data.with_format('torch')
    data.save_to_disk('/root/Code/Projects/AiPilot/training/data/traj_data')
    print(data)
