
from typing import List, Union, Any, Optional, Dict
import numpy as np
import torch.utils.data
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import threading
import os
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import sampled_past_timestamps_to_tensor, sampled_past_ego_states_to_tensor, filter_agents_tensor, sampled_tracked_objects_to_tensor_list, compute_yaw_rate_from_state_tensors, convert_absolute_quantities_to_relative, pad_agent_states, pack_agents_tensor
from datasets import load_dataset, Dataset
NUPLAN_DATA_ROOT = '/media/jarlene/Samsung_T5/nuPlan/dataset/'
NUPLAN_MAPS_ROOT = '/media/jarlene/Samsung_T5/nuPlan/dataset/map/'
NUPLAN_MAP_VERSION = 'nuplan-maps-v1.0'



Dataset.from_parquet()


def get_scenario(db_path):
    scenario_builder = NuPlanScenarioBuilder(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=db_path,
        map_version=NUPLAN_MAP_VERSION,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=True,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    scenarios = scenario_builder.get_scenarios(
        scenario_filter, SingleMachineParallelExecutor(use_process_pool=True))
    return scenarios


def ego_states_to_tensor(egos):
    size_of_egos = len(egos)
    data = np.empty((size_of_egos, 7), dtype=np.float32)
    for i in range(size_of_egos):
        e = egos[i]
        data[i] = np.array([e.rear_axle.x, e.rear_axle.y, e.rear_axle.heading,
                            e.dynamic_car_state.rear_axle_velocity_2d.x, e.dynamic_car_state.rear_axle_velocity_2d.y,
                            e.dynamic_car_state.rear_axle_acceleration_2d.x, e.dynamic_car_state.rear_axle_acceleration_2d.y])
    output = torch.from_numpy(data)
    return output


def pad_agent_states_to_max(agents_states: torch.Tensor, max_agent: int):
    num_of_agent = agents_states.shape[1]
    size_of = agents_states.size()
    if num_of_agent < max_agent:
        size = max_agent - num_of_agent
        size_of[1] = size
        zeros = torch.zeros(size_of)
        return torch.cat([agents_states, zeros], dim=1)
    elif num_of_agent > max_agent:
        return agents_states[:, :max_agent,]
    else:
        return agents_states


def process_scenario(scenario: AbstractScenario, max_length, max_agent):
    agent_data = []
    ego_data = []
    num_of_frame = scenario.get_number_of_iterations()
    assert num_of_frame > max_length
    egos = [scenario.get_ego_state_at_iteration(
            i) for i in range(num_of_frame)]
    agents = [scenario.get_tracked_objects_at_iteration(
        i).tracked_objects for i in range(num_of_frame)]
    timestamps = [scenario.get_time_point(i) for i in range(num_of_frame)]

    timestamps = sampled_past_timestamps_to_tensor(timestamps)
    egos_states = ego_states_to_tensor(egos)
    anchor_ego_state = egos_states[0, :].squeeze()

    agent_types = ['VEHICLE',
                   'PEDESTRIAN',
                   'BICYCLE',
                   'TRAFFIC_CONE',
                   'BARRIER',
                   'CZONE_SIGN',
                   'GENERIC_OBJECT',]
    list_tensor_data = {}
    for agent_type in agent_types:
        tracked_agents_tensor = sampled_tracked_objects_to_tensor_list(
            agents, TrackedObjectType[agent_type])
        agent_history = filter_agents_tensor(
            tracked_agents_tensor, reverse=True)
        if agent_history[-1].shape[0] == 0:
            agents_tensor: torch.Tensor = torch.zeros(
                (len(agent_history), 0, GenericAgents.agents_states_dim())).float()
        else:
            padded_agent_states = pad_agent_states(
                agent_history, reverse=True)
            local_coords_agent_states = convert_absolute_quantities_to_relative(
                padded_agent_states, anchor_ego_state
            )
            yaw_rate_horizon = compute_yaw_rate_from_state_tensors(
                padded_agent_states, timestamps)
            agents_tensor = pack_agents_tensor(
                local_coords_agent_states, yaw_rate_horizon)

        list_tensor_data[agent_type] = agents_tensor

    idx = np.random.randint(0, num_of_frame - max_length, (1,)).item()
    ego_data.append([egos_states[i:i+max_length+1] for i in range(idx)])
    agents = [v for k, v in list_tensor_data.items()]
    agents_states = torch.cat(agents, dim=1)
    agents_states = pad_agent_states_to_max(agents_states, max_agent)
    agent_data.append([agents_states[i:i+max_length+1,]
                      for i in range(idx)])

    t = threading.currentThread()
    print('in thread ', t.getName(), ' process scenario name:', scenario.scenario_name, ' type:', scenario.scenario_type,
          ' done ego size:', len(ego_data), ' agent size:', len(agent_data))

    result = {'ego': ego_data, 'agent': agent_data}
    return result


def internal_processs(scenarios: List[AbstractScenario], max_length, max_agent, save_path):
    agent_data = []
    ego_data = []
    print('scenarios num:', len(scenarios))
    for scenario in scenarios:
        res = process_scenario(scenario, max_length, max_agent)
        agent_data.append(res['agent'][0])
        ego_data.append(res['ego'][0])

    result = {'ego': ego_data, 'agent': agent_data}
    torch.save(result, save_path)
    return result


def processs(scenarios: List[AbstractScenario],  max_length, max_agent, save_dir, num_workers=20):

    splits = np.array_split(scenarios, num_workers)
    with ThreadPoolExecutor(max_workers=num_workers) as t:
        all_task = [t.submit(internal_processs, list(array),
                             max_length,
                             max_agent,
                             os.path.join(save_dir, 'mini_train_' + str(i) + '.pt')) for i, array in enumerate(splits)]
        wait(all_task, return_when=ALL_COMPLETED)


class NuPlanDataSet(torch.utils.data.Dataset):
    def __init__(self, data: Dict):

        self.ego = []
        self.agent = []
        for k, s in data.items():
            if k == 'ego':
                for v in s:
                    self.ego.extend(v)
            elif k == 'agent':
                for v in s:
                    self.agent.extend(v)
        self.len = len(self.ego)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ego = self.ego[index]
        agent = self.agent[index]

        ego_x = ego[:50,]
        ego_y = ego[50:-1,]
        agent_x = agent[:50,]
        agent_y = agent[50:-1,]

        return {'feature': (ego_x.float(), agent_x.float()), 'target': (ego_y.float(), agent_y.float())}
