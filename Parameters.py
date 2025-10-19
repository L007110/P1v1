# -*- coding: utf-8 -*-
import torch
import numpy as np
import itertools
from DebugPrint import *

# 全局列表
global_dqn_list = []

# 强化学习超参数 (保持不变)
RL_ALPHA = 0.001
RL_EPSILON = 0.9
RL_EPSILON_MIN = 0.01
RL_EPSILON_MAX = 0.99
RL_EPSILON_DECAY = 0.99
RL_GAMMA = 0.8

# 信道模型选择标志
USE_UMI_NLOS_MODEL = True  # True: 使用新UMi NLOS模型, False: 使用旧模型

# 功能标志位
FLAG_ADAPTIVE_EPSILON_ADJUSTMENT = True
FLAG_EMA_LOSS = True
LOS = False  # 改为False，使用NLOS模型
NLOSS = True  # 改为True，使用NLOS模型

# ==================== 新增：UMi NLOS 信道参数 ====================
# 毫米波频段参数
CENTER_FREQUENCY = 28e9  # 载波频率 28 GHz
ANTENNA_HEIGHT_BS = 10  # RSU天线高度 10m (微基站)
ANTENNA_HEIGHT_UE = 1.5  # 车辆天线高度 1.5m

# 3GPP UMi NLOS 路径损耗模型参数
PATH_LOSS_A = 35.3  # 距离系数
PATH_LOSS_B = 22.4  # 常量项
PATH_LOSS_C = 21.3  # 频率系数
SHADOWING_STD = 7.0  # 阴影衰落标准差 (dB)

# 系统带宽 (毫米波典型带宽)
SYSTEM_BANDWIDTH = 400e6  # 系统带宽 400 MHz

# 噪声参数
NOISE_POWER_DENSITY = -174  # 热噪声功率谱密度 (dBm/Hz)
BOLTZMANN_CONSTANT = 1.38e-23  # 玻尔兹曼常数
NOISE_TEMPERATURE = 290  # 噪声温度 (K)

# GNN
USE_GNN_ENHANCEMENT = True  # GNN增强开关
GNN_OUTPUT_DIM = 64         # GNN输出维度
ATTENTION_HEADS = 8        # 注意力头数

ATTENTION_MECHANISMS = {
    'multi_head': True,
    'hierarchical': True,
    'temporal': True,
    'spatial_temporal': True,
    'graph_aware': True
}

ATTENTION_DROPOUT = 0.1
TEMPORAL_SEQ_LEN = 5  # 时序注意力序列长度

# ==================== 原有参数更新 ====================
# 场景参数
SCENE_SCALE_X = 1200
SCENE_SCALE_Y = 1200
VEHICLE_SAFETY_DISTANCE = 50
VEHICLE_CAPACITY_PER_LANE = (SCENE_SCALE_X / 3) // VEHICLE_SAFETY_DISTANCE + 1

# 状态空间大小 - 需要扩展以包含CSI信息
# 原状态: 位置(x,y) + 方向(水平,垂直) = 4个维度
# 新增CSI状态: 距离 + 路径损耗 + 阴影衰落 + 当前SNR + 历史SNR = 5个维度
RL_N_STATES_BASE = int(VEHICLE_CAPACITY_PER_LANE * 4)  # 基础状态
RL_N_STATES_CSI = int(VEHICLE_CAPACITY_PER_LANE * 5)  # CSI状态
RL_N_STATES = RL_N_STATES_BASE + RL_N_STATES_CSI  # 总状态维度


# 动作空间
def formulate_action_space():
    action_space = []
    for params in itertools.product(range(5), range(3), range(3), range(10)):
        action_space.append(list(params))
    return action_space


RL_ACTION_SPACE = formulate_action_space()
RL_N_ACTIONS = len(RL_ACTION_SPACE)
RL_N_HIDDEN = RL_N_ACTIONS * 2

# 基站和车辆参数
BASE_STATION_HEIGHT = 10  # 更新为UMi模型中的10m

DIRECTION_H_RIGHT = 1
DIRECTION_H_STEADY = 0
DIRECTION_H_LEFT = -1
DIRECTION_V_UP = 1
DIRECTION_V_STEADY = 0
DIRECTION_V_DOWN = -1

BOUNDARY_POSITION_LIST = [
    (0, SCENE_SCALE_Y / 3),
    (SCENE_SCALE_X / 3, 0),
    (SCENE_SCALE_X / 3, SCENE_SCALE_Y),
    (SCENE_SCALE_X, 2 * SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y),
    (SCENE_SCALE_X, SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, 0),
]

CROSS_POSITION_LIST = [
    (SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3),
    (SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3),
]

VEHICLE_OCCUR_PROB = 0.2
VEHICLE_SPEED_KMH = 60
VEHICLE_SPEED_M3S = VEHICLE_SPEED_KMH * 1000 * 3 / 3600
SYNC_FREQUENCY = 10


# ==================== 原有通信参数 ====================
# 以下参数将被新的UMi NLOS模型替代
CARRIER_FREQUENCY_DEPRECATED = 28e9  # 使用 CENTER_FREQUENCY 替代
BASE_STATION_HEIGHT_DEPRECATED = 20  # 使用 ANTENNA_HEIGHT_BS 替代
BANDWIDTH_DEPRECATED = 10e6  # 使用 SYSTEM_BANDWIDTH 替代
TRANSMITTDE_POWER = 3  # 保持，但将在新模型中使用



# 输出参数
def print_parameters():
    debug_print("######## 参数 begin ########")
    debug_print("=== 强化学习参数 ===")
    debug_print(f"RL_ALPHA: {RL_ALPHA}")
    debug_print(f"RL_EPSILON: {RL_EPSILON}")
    debug_print(f"RL_GAMMA: {RL_GAMMA}")

    debug_print("=== UMi NLOS 信道参数 ===")
    debug_print(f"CENTER_FREQUENCY: {CENTER_FREQUENCY / 1e9} GHz")
    debug_print(f"ANTENNA_HEIGHT_BS: {ANTENNA_HEIGHT_BS} m")
    debug_print(f"ANTENNA_HEIGHT_UE: {ANTENNA_HEIGHT_UE} m")
    debug_print(f"SYSTEM_BANDWIDTH: {SYSTEM_BANDWIDTH / 1e6} MHz")
    debug_print(f"PATH_LOSS_MODEL: UMi NLOS (A={PATH_LOSS_A}, B={PATH_LOSS_B}, C={PATH_LOSS_C})")
    debug_print(f"SHADOWING_STD: {SHADOWING_STD} dB")

    debug_print("=== 场景参数 ===")
    debug_print(f"SCENE_SCALE_X: {SCENE_SCALE_X}")
    debug_print(f"SCENE_SCALE_Y: {SCENE_SCALE_Y}")
    debug_print(f"RL_N_STATES: {RL_N_STATES} (Base: {RL_N_STATES_BASE} + CSI: {RL_N_STATES_CSI})")
    debug_print(f"RL_N_ACTIONS: {RL_N_ACTIONS}")

    debug_print("=== GNN增强参数 ===")
    debug_print(f"USE_GNN_ENHANCEMENT: {USE_GNN_ENHANCEMENT}")
    debug_print(f"GNN_OUTPUT_DIM: {GNN_OUTPUT_DIM}")
    debug_print(f"ATTENTION_HEADS: {ATTENTION_HEADS}")

    debug_print("######## 参数 end ########")