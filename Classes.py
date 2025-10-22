# -*- coding: utf-8 -*-
import numpy as np
import torch
from Parameters import RL_N_STATES_CSI
from copy import deepcopy
from logger import debug, debug_print
from Parameters import (
    RL_ALPHA, RL_EPSILON, SCENE_SCALE_X, SCENE_SCALE_Y, VEHICLE_SPEED_M3S,
    CROSS_POSITION_LIST, DIRECTION_H_LEFT, DIRECTION_H_STEADY, DIRECTION_H_RIGHT,
    DIRECTION_V_UP, DIRECTION_V_STEADY, DIRECTION_V_DOWN, USE_UMI_NLOS_MODEL,
)

class DQN(torch.nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, dqn_id, start_x, start_y, end_x, end_y):
        super(DQN, self).__init__()
        self.ln = torch.nn.LayerNorm(n_states)
        self.fc1 = torch.nn.Linear(n_states, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_actions)

        self.dqn_id = dqn_id
        self.start = (start_x, start_y)
        self.end = (end_x, end_y)
        self.bs_loc = (min(start_x, end_x) + abs(start_x - end_x) / 2,
                       min(start_y, end_y) + abs(start_y - end_y) / 2)

        self.vehicle_exist_curr = False
        self.vehicle_exist_next = False
        self.curr_state = []
        self.next_state = []
        self.action = None
        self.reward = 0.0
        self.q_estimate = 0.0
        self.q_target = 0.0
        self.loss = 0.0
        self.loss_list = []
        self.epsilon = RL_EPSILON
        self.prev_loss = 0.0
        self.prev_snr = 0.0
        self.prev_delay = 0.0  # 新增：记录上一次延迟

        self.optimizer = torch.optim.Adam(self.parameters(), lr=RL_ALPHA)

        # CSI状态管理
        self.csi_states_curr = []
        self.csi_states_next = []
        self.csi_states_history = []

        # GNN相关属性
        self.gnn_enhanced = False
        self.graph_features = None

        self.vehicle_in_dqn_range_by_distance = []
        self.delay_list = []
        self.snr_list = []
        self.vehicle_count_list = []

        debug(f"DQN {self.dqn_id} created from {self.start} to {self.end}")

    def update_csi_states(self, vehicles, is_current=True):
        if USE_UMI_NLOS_MODEL:
            from NewRewardCalculator import new_reward_calculator

            csi_states = []
            for vehicle in vehicles[:min(len(vehicles), 3)]:  # 最多3辆车
                csi_state = new_reward_calculator.get_csi_for_state(vehicle, self)
                csi_states.extend(csi_state)

            # 填充到固定长度
            target_length = RL_N_STATES_CSI
            if len(csi_states) < target_length:
                csi_states.extend([0.0] * (target_length - len(csi_states)))
            else:
                csi_states = csi_states[:target_length]

            if is_current:
                self.csi_states_curr = csi_states
            else:
                self.csi_states_next = csi_states
    def forward(self, x):
        # x = self.ln(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        actions_tensor = self.fc2(x)
        return actions_tensor

    def __repr__(self):
        return (
            f"DQN {self.dqn_id} from {self.start} to {self.end}, bs_loc {self.bs_loc}"
        )

    # 参数同步
    def sync_parameters(self, source_dqn):
        self.load_state_dict(source_dqn.state_dict())


class Vehicle:
    def __init__(self, index, x, y, horizontal, vertical):
        self.first_occur = True
        self.id = index
        self.curr_loc = (x, y)
        self.curr_dir = (horizontal, vertical)

        debug(f"Vehicle {self.id} created at {self.curr_loc} with direction {self.curr_dir}")

        self.next_loc = (
            self.curr_loc[0] + self.curr_dir[0] * VEHICLE_SPEED_M3S,
            self.curr_loc[1] + self.curr_dir[1] * VEHICLE_SPEED_M3S,
        )
        self.distance_to_bs = None
        self.communication_metrics = {
            'snr_history': [],
            'delay_history': [],
            'throughput_history': []
        }

    def move(self):
        self.first_occur = False
        flag_turned = False
        curr_loc_for_debug = deepcopy(self.curr_loc)  # 备份移动前的位置
        for cross_position in CROSS_POSITION_LIST:  # 判断交叉路口转向
            if self.curr_loc[1] == cross_position[1]:  # 即当前为水平移动
                if self.curr_dir[0] == DIRECTION_H_RIGHT:  # 当前为向右移动
                    # 当向右移动时, 只有横坐标小于交叉点时才会转向
                    if (
                        self.curr_loc[0] < cross_position[0]
                        and abs(self.curr_loc[0] - cross_position[0])
                        <= VEHICLE_SPEED_M3S
                    ):
                        flag_turned = True  # 标记已转向
                        if (
                            self.curr_loc[0] < SCENE_SCALE_X / 3
                        ):  # 即在1上到了1和2的交叉点, 随机选择左转或右转
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_UP, DIRECTION_V_DOWN]
                            )
                            if turn_direction == DIRECTION_V_UP:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_V_DOWN:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                        elif (
                            SCENE_SCALE_X / 3 < self.curr_loc[0] < 2 * SCENE_SCALE_X / 3
                        ):  # 即在5上到了5和6和7和8的交叉点, 随机选择左转或直行或右转
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_UP, DIRECTION_V_STEADY, DIRECTION_V_DOWN]
                            )
                            if turn_direction == DIRECTION_V_UP:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_V_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_V_DOWN:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                        # 实际上, 此处的curr_loc为上个状态的位置, 而非当前状态的位置
                        # 因此, 此处应该根据转弯后剩余可移动距离对当前状态的位置进行更新
                        residue_distance = VEHICLE_SPEED_M3S - abs(
                            self.curr_loc[0] - cross_position[0]
                        )
                        self.curr_loc = (
                            cross_position[0] + residue_distance * self.curr_dir[0],
                            cross_position[1] + residue_distance * self.curr_dir[1],
                        )
                elif self.curr_dir[0] == DIRECTION_H_LEFT:  # 当前为向左移动
                    # 当向左移动时, 只有横坐标大于交叉点时才会转向
                    if (
                        self.curr_loc[0] > cross_position[0]
                        and abs(self.curr_loc[0] - cross_position[0])
                        <= VEHICLE_SPEED_M3S
                    ):
                        flag_turned = True
                        if (
                            SCENE_SCALE_X / 3 < self.curr_loc[0] < 2 * SCENE_SCALE_X / 3
                        ):  # 即在5上到了3和4和5的交叉点, 随机选择左转或右转
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_DOWN, DIRECTION_V_UP]
                            )
                            if turn_direction == DIRECTION_V_DOWN:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_V_UP:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                        elif (
                            self.curr_loc[1] == 2 * SCENE_SCALE_Y / 3
                        ):  # 即在7上到了6和7和8的交叉点, 随机选择左转或直行或右转
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_DOWN, DIRECTION_V_STEADY, DIRECTION_V_UP]
                            )
                            if turn_direction == DIRECTION_V_DOWN:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_V_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_V_UP:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                        elif (
                            self.curr_loc[1] == SCENE_SCALE_Y / 3
                        ):  # 即在9上到了6和9和10的交叉点, 随机选择左转或右转
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_DOWN, DIRECTION_V_UP]
                            )
                            if turn_direction == DIRECTION_V_DOWN:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_V_UP:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                        # 实际上, 此处的curr_loc为上个状态的位置, 而非当前状态的位置
                        # 因此, 此处应该根据转弯后剩余可移动距离对当前状态的位置进行更新
                        residue_distance = VEHICLE_SPEED_M3S - abs(
                            self.curr_loc[0] - cross_position[0]
                        )
                        self.curr_loc = (
                            cross_position[0] + residue_distance * self.curr_dir[0],
                            cross_position[1] + residue_distance * self.curr_dir[1],
                        )

            elif self.curr_loc[0] == cross_position[0]:  # 即当前为垂直移动
                if self.curr_dir[1] == DIRECTION_V_UP:  # 当前为向上移动
                    # 当向上移动时, 只有纵坐标小于交叉点时才会转向
                    if (
                        self.curr_loc[1] < cross_position[1]
                        and abs(self.curr_loc[1] - cross_position[1])
                        <= VEHICLE_SPEED_M3S
                    ):
                        flag_turned = True
                        if (
                            self.curr_loc[1] < SCENE_SCALE_Y / 3
                            and self.curr_loc[0] == SCENE_SCALE_X / 3
                        ):  # 即在2上到了1和2和3的交叉点, 随机选择左转或直行
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_LEFT, DIRECTION_H_STEADY]
                            )
                            if turn_direction == DIRECTION_H_LEFT:  # 左转
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                        elif (
                            self.curr_loc[1] < SCENE_SCALE_Y / 3
                            and self.curr_loc[0] == 2 * SCENE_SCALE_X / 3
                        ):  # 即在10上到了6和9和10的交叉点, 随机选择直行或右转
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_STEADY, DIRECTION_H_RIGHT]
                            )
                            if turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_H_RIGHT:  # 右转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                        elif (
                            SCENE_SCALE_X / 3 < self.curr_loc[1] < 2 * SCENE_SCALE_X / 3
                            and self.curr_loc[0] == SCENE_SCALE_X / 3
                        ):  # 即在3上到了3和4和5的交叉点, 随机选择直行或右转
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_STEADY, DIRECTION_H_RIGHT]
                            )
                            if turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_H_RIGHT:  # 右转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                        elif (
                            SCENE_SCALE_X / 3 < self.curr_loc[1] < 2 * SCENE_SCALE_X / 3
                            and self.curr_loc[0] == 2 * SCENE_SCALE_X / 3
                        ):  # 即在6上到了5和6和7和8的交叉点, 随机选择左转或直行或右转
                            turn_direction = np.random.default_rng().choice(
                                [
                                    DIRECTION_H_LEFT,
                                    DIRECTION_H_STEADY,
                                    DIRECTION_H_RIGHT,
                                ]
                            )
                            if turn_direction == DIRECTION_H_LEFT:  # 左转
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_H_RIGHT:  # 右转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                        # 实际上, 此处的curr_loc为上个状态的位置, 而非当前状态的位置
                        # 因此, 此处应该根据转弯后剩余可移动距离对当前状态的位置进行更新
                        residue_distance = VEHICLE_SPEED_M3S - abs(
                            self.curr_loc[1] - cross_position[1]
                        )
                        self.curr_loc = (
                            cross_position[0] + residue_distance * self.curr_dir[0],
                            cross_position[1] + residue_distance * self.curr_dir[1],
                        )
                elif self.curr_dir[1] == DIRECTION_V_DOWN:  # 当前为向下移动
                    # 当向下移动时, 只有纵坐标大于交叉点时才会转向
                    if (
                        self.curr_loc[1] > cross_position[1]
                        and abs(self.curr_loc[1] - cross_position[1])
                        <= VEHICLE_SPEED_M3S
                    ):
                        flag_turned = True
                        if (
                            self.curr_loc[1] > 2 * SCENE_SCALE_Y / 3
                            and self.curr_loc[0] == SCENE_SCALE_X / 3
                        ):  # 即在4上到了3和4和5的交叉点, 随机选择左转或直行
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_RIGHT, DIRECTION_H_STEADY]
                            )
                            if turn_direction == DIRECTION_H_RIGHT:  # 左转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                        elif (
                            self.curr_loc[1] > 2 * SCENE_SCALE_Y / 3
                            and self.curr_loc[0] == 2 * SCENE_SCALE_X / 3
                        ):  # 即在8上到了5和6和7和8的交叉点, 随机选择左转或直行或右转
                            turn_direction = np.random.default_rng().choice(
                                [
                                    DIRECTION_H_RIGHT,
                                    DIRECTION_H_STEADY,
                                    DIRECTION_H_LEFT,
                                ]
                            )
                            if turn_direction == DIRECTION_H_RIGHT:  # 左转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_H_LEFT:  # 右转
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                        elif (
                            SCENE_SCALE_X / 3 < self.curr_loc[1] < 2 * SCENE_SCALE_X / 3
                            and self.curr_loc[0] == SCENE_SCALE_X / 3
                        ):  # 即在3上到了1和2和3的交叉点, 随机选择直行或右转
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_STEADY, DIRECTION_H_LEFT]
                            )
                            if turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_H_LEFT:  # 右转
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                        elif (
                            SCENE_SCALE_X / 3 < self.curr_loc[1] < 2 * SCENE_SCALE_X / 3
                            and self.curr_loc[0] == 2 * SCENE_SCALE_X / 3
                        ):  # 即在6上到了6和9和10的交叉点, 随机选择左转或直行
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_RIGHT, DIRECTION_H_STEADY]
                            )
                            if turn_direction == DIRECTION_H_RIGHT:  # 左转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                        # 实际上, 此处的curr_loc为上个状态的位置, 而非当前状态的位置
                        # 因此, 此处应该根据转弯后剩余可移动距离对当前状态的位置进行更新
                        residue_distance = VEHICLE_SPEED_M3S - abs(
                            self.curr_loc[1] - cross_position[1]
                        )
                        self.curr_loc = (
                            cross_position[0] + residue_distance * self.curr_dir[0],
                            cross_position[1] + residue_distance * self.curr_dir[1],
                        )

        if not flag_turned:  # 如果未发生转向, 则直接基于速度更新位置
            self.curr_loc = self.next_loc

        self.next_loc = (
            self.curr_loc[0] + self.curr_dir[0] * VEHICLE_SPEED_M3S,
            self.curr_loc[1] + self.curr_dir[1] * VEHICLE_SPEED_M3S,
        )  # 基于速度计算的下一步位置

        debug(f"Vehicle {self.id} moved from {curr_loc_for_debug} to {self.curr_loc}")

    def record_communication_metrics(self, delay, snr, throughput=None):
        """安全记录通信指标"""
        if delay is not None and not np.isnan(delay) and delay > 0:
            self.communication_metrics['delay_history'].append(delay)
        else:
            self.communication_metrics['delay_history'].append(1.0)

        if snr is not None and not np.isnan(snr) and snr > 0 and not np.isinf(snr):
            self.communication_metrics['snr_history'].append(snr)
        else:
            self.communication_metrics['snr_history'].append(0.0)

        if throughput:
            self.communication_metrics['throughput_history'].append(throughput)