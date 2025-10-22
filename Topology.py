# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from Parameters import (
    RL_N_STATES,
    RL_N_HIDDEN,
    RL_N_ACTIONS,
    SCENE_SCALE_X,
    SCENE_SCALE_Y,
    VEHICLE_SAFETY_DISTANCE,
    DIRECTION_H_RIGHT,
    DIRECTION_H_STEADY,
    DIRECTION_H_LEFT,
    DIRECTION_V_UP,
    DIRECTION_V_STEADY,
    DIRECTION_V_DOWN,
    BOUNDARY_POSITION_LIST,
    VEHICLE_OCCUR_PROB,
    # === 新增导入：信道模型参数 ===
    USE_UMI_NLOS_MODEL,
    ANTENNA_HEIGHT_BS,
)
from Classes import DQN, Vehicle
from logger import debug, debug_print


def formulate_global_list_dqn(dqn_list, device):
    """
    创建全局DQN列表 - 主要修改基站高度设置
    """
    # 共10条道路, 每条道路对应一个DQN智能体
    # === 修改：根据信道模型选择基站高度 ===
    if USE_UMI_NLOS_MODEL:
        base_station_height = ANTENNA_HEIGHT_BS  # 使用UMi模型的基站高度
        debug("Topology: Using UMi NLOS model base station height")
    else:
        from Parameters import BASE_STATION_HEIGHT
        base_station_height = BASE_STATION_HEIGHT  # 使用原模型高度
        debug("Topology: Using original model base station height")

    # 创建DQN列表 - 保持原有结构不变
    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=1,
            start_x=0,
            start_y=SCENE_SCALE_Y / 3,
            end_x=SCENE_SCALE_X / 3,
            end_y=SCENE_SCALE_Y / 3,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=2,
            start_x=SCENE_SCALE_X / 3,
            start_y=0,
            end_x=SCENE_SCALE_X / 3,
            end_y=SCENE_SCALE_Y / 3,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=3,
            start_x=SCENE_SCALE_X / 3,
            start_y=SCENE_SCALE_Y / 3,
            end_x=SCENE_SCALE_X / 3,
            end_y=2 * SCENE_SCALE_Y / 3,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=4,
            start_x=SCENE_SCALE_X / 3,
            start_y=2 * SCENE_SCALE_Y / 3,
            end_x=SCENE_SCALE_X / 3,
            end_y=SCENE_SCALE_Y,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=5,
            start_x=SCENE_SCALE_X / 3,
            start_y=2 * SCENE_SCALE_Y / 3,
            end_x=2 * SCENE_SCALE_X / 3,
            end_y=2 * SCENE_SCALE_Y / 3,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=6,
            start_x=2 * SCENE_SCALE_X / 3,
            start_y=SCENE_SCALE_Y / 3,
            end_x=2 * SCENE_SCALE_X / 3,
            end_y=2 * SCENE_SCALE_Y / 3,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=7,
            start_x=2 * SCENE_SCALE_X / 3,
            start_y=2 * SCENE_SCALE_Y / 3,
            end_x=SCENE_SCALE_X,
            end_y=2 * SCENE_SCALE_Y / 3,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=8,
            start_x=2 * SCENE_SCALE_X / 3,
            start_y=2 * SCENE_SCALE_Y / 3,
            end_x=2 * SCENE_SCALE_X / 3,
            end_y=SCENE_SCALE_Y,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=9,
            start_x=2 * SCENE_SCALE_X / 3,
            start_y=SCENE_SCALE_Y / 3,
            end_x=SCENE_SCALE_X,
            end_y=SCENE_SCALE_Y / 3,
        ).to(device)
    )

    dqn_list.append(
        DQN(
            RL_N_STATES,
            RL_N_HIDDEN,
            RL_N_ACTIONS,
            dqn_id=10,
            start_x=2 * SCENE_SCALE_X / 3,
            start_y=0,
            end_x=2 * SCENE_SCALE_X / 3,
            end_y=SCENE_SCALE_Y / 3,
        ).to(device)
    )

    debug(f"Created {len(dqn_list)} DQNs with base station height: {base_station_height}m")


def vehicle_movement(vehicle_id, vehicle_list):
    """
    车辆移动更新 - 完全保持原有逻辑不变
    """
    # 更新当前车辆位置
    if len(vehicle_list) > 0:
        while True:
            flag_vehicle_location_update_succeed = True
            temp_vehicle_list = deepcopy(vehicle_list)  # 将车辆列表备份为临时列表
            for vehicle in temp_vehicle_list:
                vehicle.move()  # 更新车辆位置

            # 检查更新后的临时车辆列表, 任意两辆同向车之间的距离必须大于等于安全距离, 否则重新更新车辆位置
            for first_vehicle in temp_vehicle_list:
                for second_vehicle in temp_vehicle_list:
                    if (
                        first_vehicle != second_vehicle
                        and first_vehicle.curr_dir == second_vehicle.curr_dir
                    ):  # 任意两辆同向车
                        if (
                            np.sqrt(
                                (first_vehicle.curr_loc[0] - second_vehicle.curr_loc[0])
                                ** 2
                                + (
                                    first_vehicle.curr_loc[1]
                                    - second_vehicle.curr_loc[1]
                                )
                                ** 2
                            )
                            < VEHICLE_SAFETY_DISTANCE
                        ):
                            debug(
                                f"Warning: vehicle {first_vehicle.id} and vehicle {second_vehicle.id} are too close."
                            )
                            flag_vehicle_location_update_succeed = False
                            break

                if not flag_vehicle_location_update_succeed:
                    break

            if not flag_vehicle_location_update_succeed:
                debug(f"Warning: vehicle location update failed.")
                temp_vehicle_list.clear()
                continue
            else:
                debug(f"Vehicle location update succeed.")
                break

        vehicle_list.clear()
        vehicle_list = deepcopy(temp_vehicle_list)
        debug(f"Vehicle list updated, length: {len(vehicle_list)}")

        # 将超出边界的车辆移除
        for vehicle in vehicle_list:
            if (
                vehicle.curr_loc[0] < 0
                or vehicle.curr_loc[0] > SCENE_SCALE_X
                or vehicle.curr_loc[1] < 0
                or vehicle.curr_loc[1] > SCENE_SCALE_Y
            ):
                debug(f"Vehicle {vehicle.id} is out of boundary, removed.")
                vehicle_list.remove(vehicle)

    # 根据特定概率添加车辆 - 保持原有逻辑不变
    for boundary_position in BOUNDARY_POSITION_LIST:
        horizontal, vertical = 0, 0
        if np.random.uniform() <= VEHICLE_OCCUR_PROB:
            if boundary_position == (0, SCENE_SCALE_Y / 3):  # 1
                horizontal, vertical = DIRECTION_H_RIGHT, DIRECTION_V_STEADY
            elif boundary_position == (SCENE_SCALE_X / 3, 0):  # 2
                horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_UP
            elif boundary_position == (SCENE_SCALE_X / 3, SCENE_SCALE_Y):  # 4
                horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_DOWN
            elif boundary_position == (SCENE_SCALE_X, 2 * SCENE_SCALE_Y / 3):  # 7
                horizontal, vertical = DIRECTION_H_LEFT, DIRECTION_V_STEADY
            elif boundary_position == (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y):  # 8
                horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_DOWN
            elif boundary_position == (SCENE_SCALE_X, SCENE_SCALE_Y / 3):  # 9
                horizontal, vertical = DIRECTION_H_LEFT, DIRECTION_V_STEADY
            elif boundary_position == (2 * SCENE_SCALE_X / 3, 0):  # 10
                horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_UP
            else:
                debug_print(f"Error: invalid boundary_position {boundary_position}")
                quit()

            vehicle_id += 1  # 更新全局车辆ID
            vehicle_list.append(
                Vehicle(
                    vehicle_id,
                    boundary_position[0],
                    boundary_position[1],
                    horizontal,
                    vertical,
                )
            )

    return vehicle_id, vehicle_list