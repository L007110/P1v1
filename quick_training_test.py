#  快速训练测试
import torch
from DebugPrint import debug_print


def quick_training_test():
    """快速训练测试"""
    debug_print("=== 快速训练测试 ===")

    try:
        from Main import rl
        from Parameters import global_dqn_list
        from Topology import formulate_global_list_dqn

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug_print(f"训练设备: {device}")

        # 初始化DQN
        formulate_global_list_dqn(global_dqn_list, device)
        debug_print(f"初始化了 {len(global_dqn_list)} 个DQN")

        # 测试GNN增强
        from GNNIntegration import global_gnn_manager
        if global_gnn_manager.use_gnn:
            debug_print("✅ GNN增强已启用")
            for dqn in global_dqn_list[:2]:  # 只增强前2个
                enhanced = global_gnn_manager.enhance_dqn_with_gnn(dqn)
                debug_print(f"  DQN {dqn.dqn_id} -> {enhanced.__class__.__name__}")
        else:
            debug_print("ℹ️ GNN增强未启用")

        debug_print("🎉 训练环境就绪！")
        return True

    except Exception as e:
        debug_print(f"❌ 训练测试失败: {e}")
        return False


if __name__ == "__main__":
    if quick_training_test():
        debug_print("🚀 现在可以运行: python Main.py")
    else:
        debug_print("❌ 需要检查训练配置")