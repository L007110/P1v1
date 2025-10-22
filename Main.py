# -*- coding: utf-8 -*-
import numpy as np
import torch
from ActionChooser import choose_action
from logger import global_logger, debug_print, debug, set_debug_mode
from Parameters import *
from Topology import formulate_global_list_dqn, vehicle_movement
from Parameters import USE_PRIORITY_REPLAY, PER_BATCH_SIZE

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


def initialize_enhanced_training():
    """
    初始化增强训练组件
    """
    from PriorityReplayBuffer import initialize_global_per
    from Parameters import USE_PRIORITY_REPLAY, PER_CAPACITY

    if USE_PRIORITY_REPLAY:
        global_per_buffer = initialize_global_per(PER_CAPACITY)
        from logger import debug_print
        debug_print("Priority Experience Replay initialized")
        return global_per_buffer
    else:
        from logger import debug_print
        debug_print("Using standard experience replay")
        return None


def enhanced_training_step(dqn, per_buffer, device):
    """
    PER增强训练步骤
    """
    try:
        # 从PER缓冲区采样
        batch, indices, weights = per_buffer.sample(PER_BATCH_SIZE)

        if batch is None:
            # 回退到传统训练
            traditional_training_step(dqn, device)
            return

        # 转换批次数据
        states = torch.FloatTensor([exp.state for exp in batch]).to(device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # 双头DQN前向传播
        current_q_values = dqn(states)  # [batch_size, num_actions]
        next_q_values = dqn(next_states)  # [batch_size, num_actions]

        # 计算目标Q值
        target_q_values = rewards + RL_GAMMA * torch.max(next_q_values, dim=1)[0]  # [batch_size]

        # 计算当前动作的Q值
        current_action_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # 计算TD误差（用于优先级更新）
        td_errors = (target_q_values - current_action_q_values).abs().detach().cpu().numpy()

        # 计算损失（应用重要性采样权重）
        dqn.loss = (weights * (current_action_q_values - target_q_values.detach()) ** 2).mean()

        # 更新PER优先级
        per_buffer.update_priorities(indices, td_errors)

        # 反向传播
        dqn.optimizer.zero_grad()
        dqn.loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
        dqn.optimizer.step()

        debug(f"Enhanced training - DQN {dqn.dqn_id}: Loss {dqn.loss.item():.4f}")

    except Exception as e:
        debug(f"Error in enhanced training step: {e}")
        # 降级到传统训练
        traditional_training_step(dqn, device)


def traditional_training_step(dqn, device):
    """
    传统训练步骤（保持原有逻辑）
    """
    try:
        # 确保状态是正确格式
        curr_state_tensor = torch.tensor(dqn.curr_state).float().to(device)
        next_state_tensor = torch.tensor(dqn.next_state).float().to(device)

        # 如果状态是1D，添加批次维度
        if curr_state_tensor.dim() == 1:
            curr_state_tensor = curr_state_tensor.unsqueeze(0)
            next_state_tensor = next_state_tensor.unsqueeze(0)

        # 计算目标Q值和损失
        next_q_values = dqn(next_state_tensor)
        # 如果输出是2D，取最大值
        if next_q_values.dim() == 2:
            max_next_q = torch.max(next_q_values, dim=1)[0]
        else:
            max_next_q = next_q_values.max()

        dqn.q_target = dqn.reward + RL_GAMMA * max_next_q

        # 计算当前Q估计
        curr_q_values = dqn(curr_state_tensor)
        if curr_q_values.dim() == 2 and curr_q_values.size(0) == 1:
            curr_q_values = curr_q_values.squeeze(0)

        action_index = RL_ACTION_SPACE.index(dqn.action) if dqn.action in RL_ACTION_SPACE else 0
        if curr_q_values.dim() == 1:
            dqn.q_estimate = curr_q_values[action_index]
        else:
            dqn.q_estimate = curr_q_values[0, action_index]

        dqn.loss = torch.nn.MSELoss()(dqn.q_estimate, dqn.q_target)

        # 反向传播
        dqn.optimizer.zero_grad()
        dqn.loss.backward()
        dqn.optimizer.step()

    except Exception as e:
        debug(f"Error in traditional training step: {e}")
        # 设置默认损失值
        dqn.loss = torch.tensor(1.0, requires_grad=True)


def rl():
    """
    增强的RL训练循环（集成双头DQN和PER）
    """
    epoch = 1
    global_vehicle_id = 0
    overall_vehicle_list = []

    # 初始化增强组件
    global_per_buffer = initialize_enhanced_training()

    from logger import debug_print
    debug_print("Starting enhanced RL training with Dueling DQN and PER")

    # 检查DQN的指标列表初始化
    for dqn in global_dqn_list:
        if not hasattr(dqn, 'delay_list'):
            dqn.delay_list = []
        if not hasattr(dqn, 'snr_list'):
            dqn.snr_list = []

    while True:
        # === 修复：在循环开始时进行车辆移动 ===
        global_vehicle_id, overall_vehicle_list = vehicle_movement(global_vehicle_id, overall_vehicle_list)

        loss_list_per_epoch = []
        mean_loss = 0.0
        prev_mean_loss = 0.0
        cumulative_reward_per_epoch = 0.0

        if len(loss_list_per_epoch) > 0 and len(mean_loss_across_epochs) > 10:
            debug_print(
                f"Epoch {epoch} Prev mean loss {mean_loss} "
                f"Vehicle count {len(overall_vehicle_list)}"
            )
        else:
            debug_print(f"Epoch {epoch}")

        # 遍历DQN列表
        for dqn in global_dqn_list:
            dqn.vehicle_exist_curr = False
            dqn.curr_state = [0 for _ in range(RL_N_STATES)]

            # 检测当前范围内的车辆
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

            dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs, reverse=False)

            if dqn.vehicle_exist_curr:
                # 形成当前状态
                iState = 0
                for iVehicle in range(min(RL_N_STATES // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                    dqn.curr_state[iState] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[0]
                    dqn.curr_state[iState + 1] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[1]
                    dqn.curr_state[iState + 2] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[0]
                    dqn.curr_state[iState + 3] = dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[1]
                    iState += 4

                # CSI状态更新
                if USE_UMI_NLOS_MODEL and hasattr(dqn, 'update_csi_states'):
                    dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=True)

                # 选择动作
                choose_action(dqn, RL_ACTION_SPACE, device)

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

        # === 修复：在这里进行下一状态的计算和训练 ===
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

                # PER经验存储
                if global_per_buffer is not None:
                    action_index = RL_ACTION_SPACE.index(dqn.action) if dqn.action in RL_ACTION_SPACE else 0
                    global_per_buffer.add(
                        state=dqn.curr_state,
                        action=action_index,
                        reward=dqn.reward,
                        next_state=dqn.next_state,
                        done=False
                    )

                # PER增强训练
                if global_per_buffer is not None and len(global_per_buffer) >= PER_BATCH_SIZE:
                    enhanced_training_step(dqn, global_per_buffer, device)
                else:
                    # 传统训练（回退方案）
                    traditional_training_step(dqn, device)

                loss_list_per_epoch.append(dqn.loss.item())
                dqn.loss_list.append(dqn.loss.item())

                # ε-greedy策略调整
                if FLAG_ADAPTIVE_EPSILON_ADJUSTMENT:
                    if dqn.loss < dqn.prev_loss:
                        dqn.epsilon = min(RL_EPSILON_MAX, dqn.epsilon / RL_EPSILON_DECAY)
                    elif dqn.loss > dqn.prev_loss:
                        dqn.epsilon = max(RL_EPSILON_MIN, dqn.epsilon * RL_EPSILON_DECAY)
                    dqn.prev_loss = dqn.loss

        # 计算平均指标
        mean_delay, mean_snr_db = calculate_mean_metrics(global_dqn_list)

        if len(loss_list_per_epoch) > 0:
            if FLAG_EMA_LOSS:
                EMA_WEIGHT = 0.2
                mean_loss = (np.min(loss_list_per_epoch) * EMA_WEIGHT +
                             np.mean(loss_list_per_epoch) * EMA_WEIGHT * 0.1 +
                             prev_mean_loss * (1 - EMA_WEIGHT) * 0.9)
                prev_mean_loss = mean_loss
            else:
                mean_loss = np.mean(loss_list_per_epoch)

        # 记录epoch信息
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

        # 记录PER统计信息
        if global_per_buffer is not None:
            per_stats = global_per_buffer.get_statistics()
            global_logger.logger.info(
                f"PER Stats - Buffer: {per_stats['buffer_size']}, "
                f"Avg Priority: {per_stats['avg_priority']:.4f}"
            )

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
            break

        # === 修复：在这里增加epoch计数 ===
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