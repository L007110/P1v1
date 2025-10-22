# -*- coding: utf-8 -*-
import numpy as np
import torch
from logger import debug, debug_print


def choose_action(dqn, action_space, device):
    # 确保状态是正确格式的tensor
    state_tensor = torch.tensor(dqn.curr_state).float().to(device)

    # 如果状态是1D，确保DQN能正确处理
    if state_tensor.dim() == 1:
        state_tensor = state_tensor.unsqueeze(0)  # 添加批次维度

    actions_tensor = dqn(state_tensor)

    # 如果输出是2D（批次），取第一个元素
    if actions_tensor.dim() == 2 and actions_tensor.size(0) == 1:
        actions_tensor = actions_tensor.squeeze(0)

    # if np.random.uniform() > dqn.epsilon:
    if np.random.uniform() > dqn.epsilon:
        debug(f"Random action for exploration")
        action_index = np.random.randint(0, len(action_space))
        dqn.action = action_space[action_index]
        dqn.q_estimate = actions_tensor[action_index]
    else:
        debug(f"Action chosen by DQN for exploitation")
        dqn.action = action_space[actions_tensor.argmax()]
        dqn.q_estimate = actions_tensor.max()
