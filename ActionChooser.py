# -*- coding: utf-8 -*-
import numpy as np
import torch
from DebugPrint import *


def choose_action(dqn, action_space, device):
    actions_tensor = dqn(torch.tensor(dqn.curr_state).float().to(device))

    # if np.random.uniform() < 1:
    if np.random.uniform() > dqn.epsilon:
    # if np.random.uniform() > 0:
        debug(f"Random action for exploration")
        action_index = np.random.randint(0, len(action_space))
        dqn.action = action_space[action_index]
        # dqn.action = torch.randint(0, len(action_space), (1,)).item()
        dqn.q_estimate = actions_tensor[action_index]
    else:
        debug(f"Action chosen by DQN for exploitation")
        dqn.action = action_space[actions_tensor.argmax()]
        dqn.q_estimate = actions_tensor.max()
