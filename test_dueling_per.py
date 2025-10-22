# test_dueling_per.py - 修复版验证脚本
import torch
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def debug_print(msg):
    """简单的调试输出函数"""
    print(f"[DEBUG] {msg}")


def set_debug_mode(mode):
    """兼容性函数"""
    pass


def test_dueling_dqn_creation():
    """测试双头DQN创建和基本功能"""
    debug_print("=== 测试双头DQN创建 ===")

    try:
        # 动态导入，避免依赖问题
        from Parameters import RL_N_STATES, RL_N_HIDDEN, RL_N_ACTIONS, USE_DUELING_DQN

        if USE_DUELING_DQN:
            from Classes import DuelingDQN as DQNClass
        else:
            from Classes import DQN as DQNClass

        # 创建测试DQN
        test_dqn = DQNClass(
            n_states=RL_N_STATES,
            n_hidden=RL_N_HIDDEN,
            n_actions=RL_N_ACTIONS,
            dqn_id=99,
            start_x=0,
            start_y=0,
            end_x=100,
            end_y=100
        )

        # 测试前向传播
        test_input = torch.randn(1, RL_N_STATES)
        output = test_dqn(test_input)

        debug_print(f"  DQN创建成功 - 类型: {type(test_dqn).__name__}")
        debug_print(f"   输入维度: {test_input.shape}")
        debug_print(f"   输出维度: {output.shape}")

        # 如果是双头DQN，测试额外功能
        if USE_DUELING_DQN and hasattr(test_dqn, 'get_value_advantage'):
            value, advantages = test_dqn.get_value_advantage(test_input)
            debug_print(f"   状态价值 V(s): {value.item():.3f}")
            debug_print(f"   动作优势 A(s,a) 形状: {advantages.shape}")

        return True

    except Exception as e:
        debug_print(f"  DQN测试失败: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return False


def test_per_buffer():
    """测试优先级经验回放缓冲区"""
    debug_print("=== 测试PER缓冲区 ===")

    try:
        from PriorityReplayBuffer import PriorityReplayBuffer
        from Parameters import RL_N_STATES, RL_N_ACTIONS

        # 创建测试缓冲区
        buffer = PriorityReplayBuffer(capacity=100)

        # 添加测试经验
        for i in range(10):
            state = np.random.randn(RL_N_STATES).tolist()
            action = i % RL_N_ACTIONS
            reward = np.random.uniform(-1, 1)
            next_state = np.random.randn(RL_N_STATES).tolist()
            done = False

            buffer.add(state, action, reward, next_state, done)

        debug_print(f"  PER缓冲区创建成功")
        debug_print(f"   缓冲区大小: {len(buffer)}")

        # 测试采样
        batch, indices, weights = buffer.sample(batch_size=5)

        if batch is not None:
            debug_print(f"   采样批次大小: {len(batch)}")
            debug_print(f"   重要性权重范围: [{weights.min():.3f}, {weights.max():.3f}]")

            # 测试优先级更新
            td_errors = np.random.uniform(0, 1, size=len(indices))
            buffer.update_priorities(indices, td_errors)

            stats = buffer.get_statistics()
            debug_print(f"   统计信息: 大小={stats['buffer_size']}, 平均优先级={stats['avg_priority']:.3f}")

        return True

    except Exception as e:
        debug_print(f"  PER缓冲区测试失败: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return False


def test_parameters():
    """测试参数配置"""
    debug_print("=== 测试参数配置 ===")

    try:
        from Parameters import (
            USE_DUELING_DQN, USE_PRIORITY_REPLAY,
            RL_N_STATES, RL_N_HIDDEN, RL_N_ACTIONS,
            print_parameters
        )

        debug_print(f"  参数加载成功")
        debug_print(f"   使用双头DQN: {USE_DUELING_DQN}")
        debug_print(f"   使用优先级回放: {USE_PRIORITY_REPLAY}")
        debug_print(f"   状态维度: {RL_N_STATES}")
        debug_print(f"   隐藏层维度: {RL_N_HIDDEN}")
        debug_print(f"   动作空间: {RL_N_ACTIONS}")

        # 打印完整参数
        print_parameters()

        return True

    except Exception as e:
        debug_print(f"  参数测试失败: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return False


def test_topology():
    """测试拓扑创建"""
    debug_print("=== 测试拓扑创建 ===")

    try:
        from Topology import formulate_global_list_dqn
        from Parameters import global_dqn_list, USE_DUELING_DQN, RL_N_STATES

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建DQN列表
        formulate_global_list_dqn(global_dqn_list, device)

        debug_print(f"   拓扑创建成功")
        debug_print(f"   创建的DQN数量: {len(global_dqn_list)}")
        debug_print(f"   使用架构: {'DuelingDQN' if USE_DUELING_DQN else 'Traditional DQN'}")
        debug_print(f"   设备: {device}")

        # 验证第一个DQN
        if global_dqn_list:
            first_dqn = global_dqn_list[0]
            debug_print(f"   第一个DQN: {type(first_dqn).__name__} ID={first_dqn.dqn_id}")

            # 测试网络输出
            test_state = torch.randn(1, RL_N_STATES).to(device)
            with torch.no_grad():
                output = first_dqn(test_state)
                debug_print(f"   网络输出形状: {output.shape}")
                debug_print(f"   输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

        return True

    except Exception as e:
        debug_print(f"  拓扑测试失败: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return False


if __name__ == "__main__":
    debug_print("  开始基础集成验证...")

    success_count = 0
    total_tests = 4

    # 运行测试
    tests = [
        test_parameters,
        test_dueling_dqn_creation,
        test_per_buffer,
        test_topology
    ]

    for test_func in tests:
        try:
            if test_func():
                success_count += 1
        except Exception as e:
            debug_print(f"  测试 {test_func.__name__} 异常: {e}")

    # 输出结果
    debug_print("=" * 50)
    if success_count == total_tests:
        debug_print(f"  所有测试通过! ({success_count}/{total_tests})")
        debug_print("  基础集成完成，可以进入阶段2")
    else:
        debug_print(f"  部分测试通过 ({success_count}/{total_tests})")
        debug_print("  请检查失败的测试并修复")

        # 提供修复建议
        if success_count == 0:
            debug_print("\n 建议执行顺序:")
            debug_print("1. 先创建 PriorityReplayBuffer.py")
            debug_print("2. 然后修改 Parameters.py")
            debug_print("3. 接着修改 Classes.py")
            debug_print("4. 最后修改 Topology.py")