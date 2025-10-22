<<<<<<< HEAD
# -*- coding: utf-8 -*-
from logger import debug, debug_print
import numpy as np
from Parameters import TRANSMITTDE_POWER


def reward_calculator(
    action,
    vehicle_in_dqn_range_by_distance,
    snr_curr,
    snr_bef,
    overall_vehicle_list,
    distance_v_bs,
    delay,
):
    reward = 0.0

    # 统计分配的波束数量
    beam_count = sum(action)

    # 根据波束利用率给予奖励, 最大波束数为8
    if len(overall_vehicle_list) > 0:
        if beam_count == 0:
            reward -= 1
        else:
            reward += beam_count / 8
            debug(f"Reward: {reward} with beam_count: {beam_count}")

    # # 根据信噪比给予奖励
    # reward_snr = 0.0
    # reward_snr = snr_curr - snr_bef
    # reward += reward_snr * 0.5
    #
    # beam_count = sum(action)
    # if beam_count == 0:
    #     reward -= 1
    # else:
    #     reward += 1 / beam_count
    #
    # # 功率约束奖励
    # total_power = TRANSMITTDE_POWER * beam_count
    # if total_power > 100:
    #     reward -= 1

    #  波束有效性奖励
    if len(vehicle_in_dqn_range_by_distance) > 0:
        beam_weights  = [1 / (distance_v_bs + 1e-5) for v in overall_vehicle_list]
        total_weight = sum(beam_weights)

        # 修复：截断动作到车辆数
        valid_action = action[: len(overall_vehicle_list)]
        beam_utilization = (
            sum([action[i] * beam_weights[i] for i in range(len(valid_action))])
            / total_weight
        )
        reward += beam_utilization * 0.5
        debug(f"beam_utilization:{beam_utilization}")

    # reward += (snr_curr - snr_bef) * 0.5

    # 传输速率奖励(即时延奖励）

    # 第二次奖励计算归总
    # 时延
    reward -= delay * 1.5e7
    debug(f"reward: {reward}")

    # SNR
    # 1.信噪比变化奖励(提升时奖励，下降时惩罚)
    reward += (snr_curr - snr_bef) * 0.5  # 权重系数
    # 2. 波束效率奖励(波束越少奖励越高)
    beam_count = action[0] + 1
    if beam_count == 0:
        reward -= 1.0  # 惩罚未分配波束
    else:
        reward += 1.0 / beam_count
    # 3. 功率效率惩罚(总功率越高惩罚越大)
    total_power = TRANSMITTDE_POWER * beam_count * ((action[3] + 1) / 10.0)
    reward -= total_power * 0.01  # 权重系数可根据实际效果调整

    return reward


def delay_calculator(
    GAIN_ANTENNA_T,
    distance_v_bs,
    BANDWIDTH,
    TRANSMITTDE_POWER,
    noise_power,
    dqn,
):
    # 解析动作参数(包含波束数量、水平方向、垂直方向、功率分配)
    beam_count = dqn.action[0] + 1  # 波束数量(1~5，action[0]范围 0~4)
    horizontal_dir = dqn.action[1]  # 水平方向参数(0:左，1:稳态，2:右)
    vertical_dir = dqn.action[2]  # 垂直方向参数(0:上，1:稳态，2:下)
    power_ratio = (dqn.action[3] + 1) / 10.0  # 功率分配比例(0.1~1.0，action[3]范围 0~9)
    # 计算总发射功率和方向增益
    total_power = (
            TRANSMITTDE_POWER * beam_count * power_ratio
    )  # 总功率 = 单波束功率 x 波束数 x分配比例
    # 方向性增益模型(左/右或上/下转向时增益降低)
    gain_h = 1.0 - 0.2 * abs(
        horizontal_dir - 1
    )  # 水平方向增益(左/右转降为 0.8，稳态保持 1.0)
    gain_v = 1.0 - 0.2 * abs(
        vertical_dir - 1
    )  # 垂直方向增益(上/下转降为 0.8，稳态保持 1.0)
    effective_gain = GAIN_ANTENNA_T * gain_h * gain_v

    delay = distance_v_bs / (
        BANDWIDTH * np.log2(1 + effective_gain * TRANSMITTDE_POWER / noise_power)
    )
    return delay


def calculate_snr(
    TRANSMITTDE_POWER,
    GAIN_ANTENNA_T,
    distance_v_bs,
    SPEED_C,
    SIGNAL_FREQUENCY,
    dqn,
    vehicle_in_dqn_range_by_distance,
    noise_power,
):
    # 解析动作参数(包含波束数量、水平方向、垂直方向、功率分配)
    beam_count = dqn.action[0] + 1  # 波束数量(1~5，action[0]范围 0~4)
    horizontal_dir = dqn.action[1]  # 水平方向参数(0:左，1:稳态，2:右)
    vertical_dir = dqn.action[2]  # 垂直方向参数(0:上，1:稳态，2:下)
    power_ratio = (dqn.action[3] + 1) / 10.0  # 功率分配比例(0.1~1.0，action[3]范围 0~9)
    # 计算总发射功率和方向增益
    total_power = (
        TRANSMITTDE_POWER * beam_count * power_ratio
    )  # 总功率 = 单波束功率 x 波束数 x分配比例
    # 方向性增益模型(左/右或上/下转向时增益降低)
    gain_h = 1.0 - 0.2 * abs(
        horizontal_dir - 1
    )  # 水平方向增益(左/右转降为 0.8，稳态保持 1.0)
    gain_v = 1.0 - 0.2 * abs(
        vertical_dir - 1
    )  # 垂直方向增益(上/下转降为 0.8，稳态保持 1.0)
    effective_gain = GAIN_ANTENNA_T * gain_h * gain_v
    # 动态计算 SNR(目前仅针对最近的车辆，可以利用循环改为计算所有车辆，缺点为snr的值考虑过多，增大计算量)
    if len(vehicle_in_dqn_range_by_distance) > 0:
        closest_vehicle = vehicle_in_dqn_range_by_distance[0]
        distance = closest_vehicle.distance_to_bs
        lambda_ = SPEED_C / SIGNAL_FREQUENCY
        Pr = (total_power * effective_gain * GAIN_ANTENNA_T * (lambda_**2)) / (
            (4 * np.pi * distance) ** 2
        )
        noise = noise_power
        snr_curr = Pr / noise
    else:
        snr_curr = 0.0
    # 更新 SNR 历史(借用 DQN 的 prev_loss属性)
    snr_bef = dqn.prev_snr  # 上一轮的 SNR
    dqn.prev_snr = snr_curr  # 保存当前 SNR
    return snr_curr, snr_bef
=======
# -*- coding: utf-8 -*-
from logger import debug, debug_print
import numpy as np
from Parameters import TRANSMITTDE_POWER


def reward_calculator(
    action,
    vehicle_in_dqn_range_by_distance,
    snr_curr,
    snr_bef,
    overall_vehicle_list,
    distance_v_bs,
    delay,
):
    reward = 0.0

    # 统计分配的波束数量
    beam_count = sum(action)

    # 根据波束利用率给予奖励, 最大波束数为8
    if len(overall_vehicle_list) > 0:
        if beam_count == 0:
            reward -= 1
        else:
            reward += beam_count / 8
            debug(f"Reward: {reward} with beam_count: {beam_count}")

    # # 根据信噪比给予奖励
    # reward_snr = 0.0
    # reward_snr = snr_curr - snr_bef
    # reward += reward_snr * 0.5
    #
    # beam_count = sum(action)
    # if beam_count == 0:
    #     reward -= 1
    # else:
    #     reward += 1 / beam_count
    #
    # # 功率约束奖励
    # total_power = TRANSMITTDE_POWER * beam_count
    # if total_power > 100:
    #     reward -= 1

    #  波束有效性奖励
    if len(vehicle_in_dqn_range_by_distance) > 0:
        beam_weights  = [1 / (distance_v_bs + 1e-5) for v in overall_vehicle_list]
        total_weight = sum(beam_weights)

        # 修复：截断动作到车辆数
        valid_action = action[: len(overall_vehicle_list)]
        beam_utilization = (
            sum([action[i] * beam_weights[i] for i in range(len(valid_action))])
            / total_weight
        )
        reward += beam_utilization * 0.5
        debug(f"beam_utilization:{beam_utilization}")

    # reward += (snr_curr - snr_bef) * 0.5

    # 传输速率奖励(即时延奖励）

    # 第二次奖励计算归总
    # 时延
    reward -= delay * 1.5e7
    debug(f"reward: {reward}")

    # SNR
    # 1.信噪比变化奖励(提升时奖励，下降时惩罚)
    reward += (snr_curr - snr_bef) * 0.5  # 权重系数
    # 2. 波束效率奖励(波束越少奖励越高)
    beam_count = action[0] + 1
    if beam_count == 0:
        reward -= 1.0  # 惩罚未分配波束
    else:
        reward += 1.0 / beam_count
    # 3. 功率效率惩罚(总功率越高惩罚越大)
    total_power = TRANSMITTDE_POWER * beam_count * ((action[3] + 1) / 10.0)
    reward -= total_power * 0.01  # 权重系数可根据实际效果调整

    return reward


def delay_calculator(
    GAIN_ANTENNA_T,
    distance_v_bs,
    BANDWIDTH,
    TRANSMITTDE_POWER,
    noise_power,
    dqn,
):
    # 解析动作参数(包含波束数量、水平方向、垂直方向、功率分配)
    beam_count = dqn.action[0] + 1  # 波束数量(1~5，action[0]范围 0~4)
    horizontal_dir = dqn.action[1]  # 水平方向参数(0:左，1:稳态，2:右)
    vertical_dir = dqn.action[2]  # 垂直方向参数(0:上，1:稳态，2:下)
    power_ratio = (dqn.action[3] + 1) / 10.0  # 功率分配比例(0.1~1.0，action[3]范围 0~9)
    # 计算总发射功率和方向增益
    total_power = (
            TRANSMITTDE_POWER * beam_count * power_ratio
    )  # 总功率 = 单波束功率 x 波束数 x分配比例
    # 方向性增益模型(左/右或上/下转向时增益降低)
    gain_h = 1.0 - 0.2 * abs(
        horizontal_dir - 1
    )  # 水平方向增益(左/右转降为 0.8，稳态保持 1.0)
    gain_v = 1.0 - 0.2 * abs(
        vertical_dir - 1
    )  # 垂直方向增益(上/下转降为 0.8，稳态保持 1.0)
    effective_gain = GAIN_ANTENNA_T * gain_h * gain_v

    delay = distance_v_bs / (
        BANDWIDTH * np.log2(1 + effective_gain * TRANSMITTDE_POWER / noise_power)
    )
    return delay


def calculate_snr(
    TRANSMITTDE_POWER,
    GAIN_ANTENNA_T,
    distance_v_bs,
    SPEED_C,
    SIGNAL_FREQUENCY,
    dqn,
    vehicle_in_dqn_range_by_distance,
    noise_power,
):
    # 解析动作参数(包含波束数量、水平方向、垂直方向、功率分配)
    beam_count = dqn.action[0] + 1  # 波束数量(1~5，action[0]范围 0~4)
    horizontal_dir = dqn.action[1]  # 水平方向参数(0:左，1:稳态，2:右)
    vertical_dir = dqn.action[2]  # 垂直方向参数(0:上，1:稳态，2:下)
    power_ratio = (dqn.action[3] + 1) / 10.0  # 功率分配比例(0.1~1.0，action[3]范围 0~9)
    # 计算总发射功率和方向增益
    total_power = (
        TRANSMITTDE_POWER * beam_count * power_ratio
    )  # 总功率 = 单波束功率 x 波束数 x分配比例
    # 方向性增益模型(左/右或上/下转向时增益降低)
    gain_h = 1.0 - 0.2 * abs(
        horizontal_dir - 1
    )  # 水平方向增益(左/右转降为 0.8，稳态保持 1.0)
    gain_v = 1.0 - 0.2 * abs(
        vertical_dir - 1
    )  # 垂直方向增益(上/下转降为 0.8，稳态保持 1.0)
    effective_gain = GAIN_ANTENNA_T * gain_h * gain_v
    # 动态计算 SNR(目前仅针对最近的车辆，可以利用循环改为计算所有车辆，缺点为snr的值考虑过多，增大计算量)
    if len(vehicle_in_dqn_range_by_distance) > 0:
        closest_vehicle = vehicle_in_dqn_range_by_distance[0]
        distance = closest_vehicle.distance_to_bs
        lambda_ = SPEED_C / SIGNAL_FREQUENCY
        Pr = (total_power * effective_gain * GAIN_ANTENNA_T * (lambda_**2)) / (
            (4 * np.pi * distance) ** 2
        )
        noise = noise_power
        snr_curr = Pr / noise
    else:
        snr_curr = 0.0
    # 更新 SNR 历史(借用 DQN 的 prev_loss属性)
    snr_bef = dqn.prev_snr  # 上一轮的 SNR
    dqn.prev_snr = snr_curr  # 保存当前 SNR
    return snr_curr, snr_bef
>>>>>>> d177c06cd79adbc5bd91dbc020ffa10ee606353d
