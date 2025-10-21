# -*- coding: utf-8 -*-
import numpy as np
from ChannelModel import global_channel_model
from DebugPrint import *
from Parameters import *


class NewRewardCalculator:

    def __init__(self):
        self.channel_model = global_channel_model

        self.BEAM_ROLLOFF_EXPONENT = 2  # 波束滚降指数
        self.ANGLE_PER_DIRECTION = 10  # 每个方向的角度范围

        #debug("NewRewardCalculator initialized with UMi NLOS model")

    def _calculate_directional_gain(self, horizontal_dir, vertical_dir):

        # === 方向性增益模型 ===
        # 计算水平/垂直角度偏移
        theta_h = (horizontal_dir - 1) * self.ANGLE_PER_DIRECTION
        theta_v = (1 - vertical_dir) * self.ANGLE_PER_DIRECTION

        # 转化为弧度并计算增益
        theta_h_rad = np.deg2rad(theta_h)
        theta_v_rad = np.deg2rad(theta_v)
        gain_h = np.cos(theta_h_rad) ** self.BEAM_ROLLOFF_EXPONENT
        gain_v = np.cos(theta_v_rad) ** self.BEAM_ROLLOFF_EXPONENT

        effective_gain = gain_h * gain_v

        #debug(f"Directional gain - H:{horizontal_dir}({theta_h}°), "
        #      f"V:{vertical_dir}({theta_v}°), Gain:{effective_gain:.3f}")

        return effective_gain

    def _record_communication_metrics(self, dqn, delay, snr):
        """安全记录通信指标到DQN"""
        try:
            # 记录延迟（确保有效值）
            if delay is not None and not np.isnan(delay) and delay > 0:
                dqn.delay_list.append(delay)
                debug(f"Recorded delay for DQN {dqn.dqn_id}: {delay:.6f}")
            else:
                dqn.delay_list.append(1.0)  # 默认值
                debug(f"Used default delay for DQN {dqn.dqn_id}: 1.0")

            # 记录SNR（确保有效值）
            if (snr is not None and not np.isnan(snr) and
                    not np.isinf(snr) and snr > -100):  # 合理的SNR范围
                dqn.snr_list.append(snr)
                debug(f"Recorded SNR for DQN {dqn.dqn_id}: {snr:.2f}dB")
            else:
                dqn.snr_list.append(0.0)  # 默认值
                debug(f"Used default SNR for DQN {dqn.dqn_id}: 0.0dB")

        except Exception as e:
            debug(f"Error recording metrics for DQN {dqn.dqn_id}: {e}")
            # 降级方案
            dqn.delay_list.append(1.0)
            dqn.snr_list.append(0.0)

    def calculate_complete_reward(self, dqn, vehicles, action):
        debug(f"=== Reward Calculation Start for DQN {dqn.dqn_id} ===")
        debug(f"Vehicles count: {len(vehicles)}")
        debug(f"Action: {action}")

        if not vehicles:
            debug("No vehicles - returning default reward 0.0")
            # 即使没有车辆也记录默认指标
            self._record_communication_metrics(dqn, 1.0, 0.0)
            return 0.0

        reward = 0.0
        delay = 1.0  # 默认延迟
        snr_curr = 0.0  # 默认SNR

        try:
            # 1. 获取最近的车辆信息
            closest_vehicle = vehicles[0]
            distance_3d = self.channel_model.calculate_3d_distance(
                (dqn.bs_loc[0], dqn.bs_loc[1]), closest_vehicle.curr_loc)

            # 2. 解析动作参数
            beam_count = action[0] + 1  # 波束数量(1~5，action[0]范围 0~4)
            horizontal_dir = action[1]  # 水平方向参数(0:左，1:稳态，2:右)
            vertical_dir = action[2]  # 垂直方向参数(0:上，1:稳态，2:下)
            power_ratio = (action[3] + 1) / 10.0  # 功率分配比例(0.1~1.0)

            # 3. 计算方向性增益
            directional_gain = self._calculate_directional_gain(horizontal_dir, vertical_dir)

            # 4. 计算总发射功率
            total_power = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain

            # 5. 计算SNR
            snr_db, snr_linear, _ = self.channel_model.calculate_snr(total_power, distance_3d)
            snr_curr = snr_db  # 使用dB值

            # 6. SNR变化奖励 (提升时奖励，下降时惩罚)
            snr_change = snr_curr - dqn.prev_snr
            reward += snr_change * 0.5

            # 7. 波束效率奖励
            if beam_count == 0:
                reward -= 1.0  # 惩罚未分配波束
            else:
                reward += 1.0 / beam_count  # 波束越少奖励越高

            # 8. 功率效率惩罚
            reward -= total_power * 0.01

            # 9. 时延计算
            delay = self.calculate_delay(distance_3d, action, directional_gain)
            reward -= delay * 0.1

            debug(f"New reward calculation: "
                  f"SNR={snr_curr:.1f}dB (change: {snr_change:.1f}dB), "
                  f"Beams={beam_count}, DirGain={directional_gain:.3f}, "
                  f"Power={total_power:.1f}W, Delay={delay:.2e}s, Reward={reward:.3f}")

            # 更新SNR历史
            dqn.prev_snr = snr_curr

        except Exception as e:
            debug(f"Error in new reward calculation: {e}")
            reward = 0.0

        # === 修复：确保记录通信指标 ===
        self._record_communication_metrics(dqn, delay, snr_curr)

        debug(f"=== Reward Calculation Complete for DQN {dqn.dqn_id} ===")
        debug(f"Final - delay={delay:.6f}, snr={snr_curr:.2f}dB, reward={reward:.3f}")

        # 归一化奖励
        return max(min(reward, 1.0), -1.0)

    def calculate_delay(self, distance_3d, dqn_action, directional_gain=1.0):

        try:
            beam_count = dqn_action[0] + 1
            power_ratio = (dqn_action[3] + 1) / 10.0

            # 计算发射功率（包含方向性增益）
            tx_power = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain

            # 使用新信道模型计算SNR
            snr_db, snr_linear, _ = self.channel_model.calculate_snr(tx_power, distance_3d)

            if snr_linear > 0:
                # 计算数据传输速率
                data_rate = SYSTEM_BANDWIDTH * np.log2(1 + snr_linear)
                delay = distance_3d / data_rate if data_rate > 0 else 1.0
            else:
                delay = 1.0  # SNR为负时的默认时延

            debug(f"New delay calculation: distance={distance_3d:.1f}m, "
                  f"DirGain={directional_gain:.3f}, SNR={snr_db:.1f}dB, "
                  f"data_rate={data_rate:.2e}bps, delay={delay:.2e}s")

        except Exception as e:
            debug(f"Error in new delay calculation: {e}")
            delay = 1.0

        return delay

    def calculate_snr_with_direction(self, tx_power, distance_3d, horizontal_dir, vertical_dir):

        # 计算方向性增益
        directional_gain = self._calculate_directional_gain(horizontal_dir, vertical_dir)

        # 应用方向性增益到发射功率
        effective_tx_power = tx_power * directional_gain

        return self.channel_model.calculate_snr(effective_tx_power, distance_3d)

    def get_csi_for_state(self, vehicle, dqn):

        if vehicle is None:
            return [0.0] * 5  # 返回零填充

        try:
            # 计算3D距离
            distance_3d = self.channel_model.calculate_3d_distance(
                (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)

            # 获取完整的信道状态信息
            csi_info = self.channel_model.get_channel_state_info(
                (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc, TRANSMITTDE_POWER)

            # CSI状态向量: [距离, 总路径损耗, 阴影衰落, 当前SNR, 历史SNR]
            csi_state = [
                csi_info['distance_3d'],
                csi_info['path_loss_total_db'],
                csi_info['shadowing_db'],
                csi_info['snr_db'],
                dqn.prev_snr
            ]

            debug(f"CSI state for vehicle {vehicle.id}: {csi_state}")

        except Exception as e:
            debug(f"Error getting CSI state: {e}")
            csi_state = [0.0] * 5

        return csi_state


# 全局实例
new_reward_calculator = NewRewardCalculator()