# -*- coding: utf-8 -*-
import numpy as np
import torch
from ActionChooser import choose_action
from logger import global_logger, debug_print, debug, set_debug_mode
from Parameters import *
from Topology import formulate_global_list_dqn, vehicle_movement

# === 新增：根据标志位选择奖励计算模块 ===
if USE_UMI_NLOS_MODEL:
    from NewRewardCalculator import new_reward_calculator

    debug_print("Main.py: Using NewRewardCalculator with UMi NLOS model")
else:
    from RewardCalculator import reward_calculator, delay_calculator, calculate_snr

    debug_print("Main.py: Using original RewardCalculator")


def calculate_mean_metrics(dqn_list):
    """安全计算平均指标"""
    delays = []
    snrs = []

    debug("=== Calculating Mean Metrics ===")

    for dqn in dqn_list:
        dqn_id = getattr(dqn, 'dqn_id', 'unknown')

        # 延迟数据 - 更宽松的验证
        if hasattr(dqn, 'delay_list') and dqn.delay_list:
            debug(f"DQN {dqn_id} delay_list length: {len(dqn.delay_list)}")
            # 取所有有效延迟值
            valid_delays = [d for d in dqn.delay_list
                            if d is not None and not np.isnan(d) and d > 0]
            if valid_delays:
                # 使用最近的一些值而不是最后一个
                recent_delays = valid_delays[-min(5, len(valid_delays)):]
                delays.extend(recent_delays)
                debug(f"DQN {dqn_id} valid delays: {len(recent_delays)}")
            else:
                debug(f"DQN {dqn_id} NO valid delays")

        # SNR数据 - 更宽松的验证
        if hasattr(dqn, 'snr_list') and dqn.snr_list:
            debug(f"DQN {dqn_id} snr_list length: {len(dqn.snr_list)}")
            valid_snrs = [s for s in dqn.snr_list
                          if s is not None and not np.isnan(s) and not np.isinf(s)]
            if valid_snrs:
                recent_snrs = valid_snrs[-min(5, len(valid_snrs)):]
                snrs.extend(recent_snrs)
                debug(f"DQN {dqn_id} valid SNRs: {len(recent_snrs)}")
            else:
                debug(f"DQN {dqn_id} NO valid SNRs")

    # 计算平均值
    mean_delay = np.mean(delays) if delays else 1.0  # 默认1.0秒
    mean_snr_linear = np.mean(snrs) if snrs else 1.0  # 默认线性SNR=1

    # 转换为dB，处理边界情况
    if mean_snr_linear > 0:
        mean_snr_db = 10 * np.log10(mean_snr_linear)
    else:
        mean_snr_db = -100  # 合理的最小值

    debug(f"=== Mean Metrics Summary ===")
    debug(f"Total delays collected: {len(delays)}")
    debug(f"Total SNRs collected: {len(snrs)}")
    debug(f"Final mean_delay: {mean_delay:.6f}")
    debug(f"Final mean_snr_db: {mean_snr_db:.2f}dB")

    return mean_delay, mean_snr_db


def rl():
    epoch = 1
    global_vehicle_id = 0
    overall_vehicle_list = []
    global_vehicle_id, overall_vehicle_list = vehicle_movement(
        global_vehicle_id, overall_vehicle_list
    )

    loss_list_per_epoch = []
    mean_loss = 0.0
    prev_mean_loss = 0.0
    mean_loss_across_epochs = []
    mean_delay_list = []
    mean_snr_list = []

    # === 添加调试信息 ===
    debug_print("Starting RL training with UMi NLOS model")
    debug_print(f"Number of DQNs: {len(global_dqn_list)}")
    debug_print(f"Number of vehicles: {len(overall_vehicle_list)}")

    # 检查DQN的指标列表初始化
    for dqn in global_dqn_list:
        if not hasattr(dqn, 'delay_list'):
            dqn.delay_list = []
        if not hasattr(dqn, 'snr_list'):
            dqn.snr_list = []
        debug_print(f"DQN {dqn.dqn_id} - delay_list: {len(dqn.delay_list)}, snr_list: {len(dqn.snr_list)}")

    while True:
        if len(loss_list_per_epoch) > 0 and len(mean_loss_across_epochs) > 10:
            debug_print(
                f"Epoch {epoch} Prev mean loss {mean_loss} "
                f"Vehicle count {len(overall_vehicle_list)}"
            )
        else:
            debug_print(f"Epoch {epoch}")
            debug(f"Epoch {epoch}: Overall vehicles: {len(overall_vehicle_list)}")
            for i, vehicle in enumerate(overall_vehicle_list[:3]):
                debug(f"Vehicle {vehicle.id} at {vehicle.curr_loc}, dir: {vehicle.curr_dir}")

        cumulative_reward_per_epoch = 0.0
        loss_list_per_epoch.clear()

        debug(f"Vehicle count: {len(overall_vehicle_list)}")

        # 遍历DQN列表
        for dqn in global_dqn_list:
            dqn.vehicle_exist_curr = False
            dqn.curr_state = [0 for _ in range(RL_N_STATES)]

            # 遍历车辆列表, 计算到基站的距离, 并按距离从近到远排序
            dqn.vehicle_in_dqn_range_by_distance = []
            for vehicle in overall_vehicle_list:
                if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and
                        dqn.start[1] <= vehicle.curr_loc[1] <= dqn.end[1]):
                    dqn.vehicle_exist_curr = True

                    if USE_UMI_NLOS_MODEL:
                        vehicle.distance_to_bs = new_reward_calculator.channel_model.calculate_3d_distance(
                            (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                    else:
                        vehicle.distance_to_bs = np.sqrt(
                            (vehicle.curr_loc[0] - dqn.bs_loc[0]) ** 2 +
                            (vehicle.curr_loc[1] - dqn.bs_loc[1]) ** 2 +
                            BASE_STATION_HEIGHT ** 2
                        )

                    dqn.vehicle_in_dqn_range_by_distance.append(vehicle)

            # 按距离从近到远排序
            dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs, reverse=False)
            debug(f"Vehicle in DQN {dqn.dqn_id} range: {len(dqn.vehicle_in_dqn_range_by_distance)}")

            if dqn.vehicle_exist_curr:
                # 形成当前状态
                iState = 0
                for iVehicle in range(min(RL_N_STATES // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                    dqn.curr_state[iState] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[0]
                    dqn.curr_state[iState + 1] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[1]
                    dqn.curr_state[iState + 2] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[0]
                    dqn.curr_state[iState + 3] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[1]
                    iState += 4
                debug(f"Current state of DQN {dqn.dqn_id}: {dqn.curr_state}")

                # CSI状态更新
                if USE_UMI_NLOS_MODEL and hasattr(dqn, 'update_csi_states'):
                    dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=True)
                    if hasattr(dqn, 'csi_states_curr'):
                        csi_start_idx = min(RL_N_STATES_BASE, len(dqn.curr_state))
                        for i, csi_val in enumerate(dqn.csi_states_curr):
                            if csi_start_idx + i < RL_N_STATES:
                                dqn.curr_state[csi_start_idx + i] = csi_val

                # 选择动作
                choose_action(dqn, RL_ACTION_SPACE, device)
                debug(f"DQN {dqn.dqn_id}: Action {dqn.action}, Estimate Q {dqn.q_estimate}")

                # 奖励计算
                if USE_UMI_NLOS_MODEL:
                    dqn.reward = new_reward_calculator.calculate_complete_reward(
                        dqn, dqn.vehicle_in_dqn_range_by_distance, dqn.action)
                else:
                    # 原有奖励计算逻辑
                    distance_v_bs = dqn.vehicle_in_dqn_range_by_distance[
                        0].distance_to_bs if dqn.vehicle_in_dqn_range_by_distance else 0
                    delay = delay_calculator(GAIN_ANTENNA_T, distance_v_bs, BANDWIDTH, TRANSMITTDE_POWER, dqn)
                    dqn.delay_list.append(delay)
                    snr_curr, snr_bef = calculate_snr(TRANSMITTDE_POWER, GAIN_ANTENNA_T, distance_v_bs,
                                                      SPEED_C, SIGNAL_FREQUENCY, dqn,
                                                      dqn.vehicle_in_dqn_range_by_distance,
                                                      CARRIER_FREQUENCY, GAIN_ANTENNA_b)
                    dqn.snr_list.append(snr_curr)
                    dqn.reward = reward_calculator(dqn.action, dqn.vehicle_in_dqn_range_by_distance,
                                                   snr_curr, snr_bef, overall_vehicle_list,
                                                   distance_v_bs, delay)

                cumulative_reward_per_epoch += dqn.reward

                debug(f"DQN {dqn.dqn_id} reward: {dqn.reward:.3f}")
                if dqn.delay_list:
                    debug(f"Latest delay: {dqn.delay_list[-1]:.6f}")
                if dqn.snr_list:
                    debug(f"Latest SNR: {dqn.snr_list[-1]:.2f}dB")

        # 车辆移动
        global_vehicle_id, overall_vehicle_list = vehicle_movement(global_vehicle_id, overall_vehicle_list)
        debug(f"{len(overall_vehicle_list)} vehicles remain after movement")

        # 计算下一状态和损失
        for dqn in global_dqn_list:
            dqn.vehicle_exist_next = False
            dqn.next_state = [0 for _ in range(RL_N_STATES)]
            dqn.vehicle_in_dqn_range_by_distance = []

            for vehicle in overall_vehicle_list:
                if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and
                        dqn.start[1] <= vehicle.curr_loc[1] <= dqn.end[1]):
                    dqn.vehicle_exist_next = True
                    if USE_UMI_NLOS_MODEL:
                        vehicle.distance_to_bs = new_reward_calculator.channel_model.calculate_3d_distance(
                            (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                    else:
                        vehicle.distance_to_bs = np.sqrt(
                            (vehicle.curr_loc[0] - dqn.bs_loc[0]) ** 2 +
                            (vehicle.curr_loc[1] - dqn.bs_loc[1]) ** 2 +
                            BASE_STATION_HEIGHT ** 2
                        )
                    dqn.vehicle_in_dqn_range_by_distance.append(vehicle)

            dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs, reverse=False)

            if dqn.vehicle_exist_next and dqn.vehicle_exist_curr:
                # 形成下一状态
                iState = 0
                for iVehicle in range(min(RL_N_STATES // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                    dqn.next_state[iState] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[0]
                    dqn.next_state[iState + 1] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[1]
                    dqn.next_state[iState + 2] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[0]
                    dqn.next_state[iState + 3] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[1]
                    iState += 4

                # CSI状态更新
                if USE_UMI_NLOS_MODEL and hasattr(dqn, 'update_csi_states'):
                    dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=False)
                    if hasattr(dqn, 'csi_states_next'):
                        csi_start_idx = min(RL_N_STATES_BASE, len(dqn.next_state))
                        for i, csi_val in enumerate(dqn.csi_states_next):
                            if csi_start_idx + i < RL_N_STATES:
                                dqn.next_state[csi_start_idx + i] = csi_val

                # 计算目标Q值和损失
                dqn.q_target = dqn.reward + RL_GAMMA * torch.max(dqn(torch.tensor(dqn.next_state).float().to(device)))
                dqn.loss = torch.nn.MSELoss()(dqn.q_estimate, dqn.q_target)
                loss_list_per_epoch.append(dqn.loss.item())
                dqn.loss_list.append(dqn.loss.item())

                debug(f"DQN {dqn.dqn_id}: Target Q {dqn.q_target}, Loss {dqn.loss}")

                # ε-greedy策略调整
                if FLAG_ADAPTIVE_EPSILON_ADJUSTMENT:
                    if dqn.loss < dqn.prev_loss:
                        dqn.epsilon = min(RL_EPSILON_MAX, dqn.epsilon / RL_EPSILON_DECAY)
                    elif dqn.loss > dqn.prev_loss:
                        dqn.epsilon = max(RL_EPSILON_MIN, dqn.epsilon * RL_EPSILON_DECAY)
                    dqn.prev_loss = dqn.loss

                # 反向传播
                dqn.optimizer.zero_grad()
                dqn.loss.backward()
                dqn.optimizer.step()

        # 记录全局指标到日志系统
        mean_delay, mean_snr_db = calculate_mean_metrics(global_dqn_list)
        mean_delay_list.append(mean_delay)
        mean_snr_list.append(mean_snr_db)

        if len(loss_list_per_epoch) > 0:
            if FLAG_EMA_LOSS:
                EMA_WEIGHT = 0.2
                mean_loss = (np.min(loss_list_per_epoch) * EMA_WEIGHT +
                             np.mean(loss_list_per_epoch) * EMA_WEIGHT * 0.1 +
                             prev_mean_loss * (1 - EMA_WEIGHT) * 0.9)
                prev_mean_loss = mean_loss
            else:
                mean_loss = np.mean(loss_list_per_epoch)
            mean_loss_across_epochs.append(mean_loss)

        # 使用新的日志系统记录epoch信息
        global_logger.log_epoch(
            epoch=epoch,
            cumulative_reward=cumulative_reward_per_epoch,
            mean_loss=mean_loss,
            mean_delay=mean_delay,
            mean_snr=mean_snr_db,
            vehicle_count=len(overall_vehicle_list)
        )

        # 记录每个DQN的性能
        for dqn in global_dqn_list:
            dqn_metrics = {
                'loss': getattr(dqn, 'loss', 0),
                'reward': getattr(dqn, 'reward', 0),
                'epsilon': getattr(dqn, 'epsilon', RL_EPSILON),
                'vehicle_count': len(getattr(dqn, 'vehicle_in_dqn_range_by_distance', [])),
                'snr': getattr(dqn, 'prev_snr', 0),
                'delay': getattr(dqn, 'prev_delay', 0)
            }
            global_logger.log_dqn_performance(dqn.dqn_id, dqn_metrics)

        # 参数同步
        if epoch % SYNC_FREQUENCY == 0:
            try:
                global_params = {}
                for dqn in global_dqn_list:
                    for name, param in dqn.named_parameters():
                        if name in global_params:
                            global_params[name] += param.data.clone()
                        else:
                            global_params[name] = param.data.clone()

                for name in global_params:
                    global_params[name] /= len(global_dqn_list)

                for dqn in global_dqn_list:
                    for name, param in dqn.named_parameters():
                        if name in global_params:
                            param.data.copy_(global_params[name])
            except Exception as e:
                debug(f"Parameter sync error: {e}")

        # 收敛判断
        if epoch == 1500:
            global_logger.log_convergence(epoch, mean_loss)
            debug_print(f"Converged at epoch {epoch} with loss {mean_loss}")

            # 输出最终结果
            for dqn in global_dqn_list:
                debug_print(f"DQN {dqn.dqn_id} loss: {dqn.loss_list}")
                debug_print(f"DQN {dqn.dqn_id} vehicle count: {len(getattr(dqn, 'vehicle_count_list', []))}")
                debug_print(f"DQN {dqn.dqn_id} delay: {dqn.delay_list}")
                debug_print(f"DQN {dqn.dqn_id} SNR: {dqn.snr_list}")

            break

        epoch += 1

    # 训练结束后保存所有结果
    global_logger.finalize()


if __name__ == "__main__":
    set_debug_mode(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(f"device is {device}")

    # 显示当前使用的模型
    if USE_UMI_NLOS_MODEL:
        debug_print("Using UMi NLOS Channel Model with NewRewardCalculator")
    else:
        debug_print("Using Original Channel Model with RewardCalculator")

    print_parameters()

    # 根据DQN类为每段车道创建智能体
    formulate_global_list_dqn(global_dqn_list, device)
    for dqn in global_dqn_list:
        debug_print(dqn)

    # 强化学习主函数
    rl()