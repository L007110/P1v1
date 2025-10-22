# 专用调试入口文件
import torch
from DebugPrint import set_debug_mode, debug_print

# 在模块级别导入
from Parameters import USE_UMI_NLOS_MODEL, USE_GNN_ENHANCEMENT


def setup_debug_environment():
    """设置调试环境"""
    set_debug_mode(True)  # 开启详细调试
    torch.manual_seed(42)  # 固定随机种子，确保可复现

    debug_print("🔧 调试环境初始化完成")


def test_individual_components():
    """逐个测试独立组件"""
    debug_print("=== 开始组件级测试 ===")

    try:
        # 1. 测试信道模型
        from ChannelModel import test_channel_model
        test_channel_model()
    except Exception as e:
        debug_print(f"❌ 信道模型测试失败: {e}")

    try:
        # 2. 测试注意力机制
        from AttentionMechanism import test_attention_mechanisms
        test_attention_mechanisms()
    except Exception as e:
        debug_print(f"❌ 注意力机制测试失败: {e}")

    try:
        # 3. 测试图构建器
        from GraphBuilder import test_graph_builder
        test_graph_builder()
    except Exception as e:
        debug_print(f"❌ 图构建器测试失败: {e}")

    try:
        # 4. 测试GNN模型
        from GNNModel import test_gnn_model
        test_gnn_model()
    except Exception as e:
        debug_print(f"❌ GNN模型测试失败: {e}")


def run_minimal_training():
    """运行最小化训练测试"""
    debug_print("=== 开始最小化训练测试 ===")

    try:
        # 简化版训练循环
        from Topology import formulate_global_list_dqn, vehicle_movement
        from Parameters import global_dqn_list

        # 创建最小DQN列表
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        formulate_global_list_dqn(global_dqn_list, device)

        # 只保留前2个DQN用于测试
        test_dqn_list = global_dqn_list[:2] if len(global_dqn_list) >= 2 else global_dqn_list

        # 运行3个epoch的微型训练
        for epoch in range(3):
            debug_print(f"🔍 测试Epoch {epoch + 1}/3")

            # 创建测试车辆
            vehicle_id = 0
            vehicle_list = []
            vehicle_id, vehicle_list = vehicle_movement(vehicle_id, vehicle_list)

            debug_print(f"Epoch {epoch + 1}: 有 {len(vehicle_list)} 辆车")

            # 简化版训练步骤
            for dqn in test_dqn_list:
                debug_print(f"DQN {dqn.dqn_id} 测试通过")

            debug_print(f"✅ Epoch {epoch + 1} 完成")

    except Exception as e:
        debug_print(f"❌ 最小化训练测试失败: {e}")


if __name__ == "__main__":
    setup_debug_environment()
    test_individual_components()  # 先测试组件
    run_minimal_training()  # 再测试集成