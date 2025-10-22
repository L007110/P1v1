# 完整系统测试
from DebugPrint import debug_print


def test_complete_system():
    """测试完整系统功能"""
    debug_print("=== 完整系统功能测试 ===")

    try:
        # 测试所有核心模块
        from GNNModel import global_gnn_model, EnhancedHeteroGNN
        from GraphBuilder import global_graph_builder
        from GNNIntegration import global_gnn_manager
        from AttentionMechanism import global_attention_mechanism

        debug_print("✅ 所有核心模块导入成功")

        # 创建测试场景
        from Classes import DQN, Vehicle

        test_dqns = [
            DQN(10, 20, 5, 1, 0, 400, 400, 400),
            DQN(10, 20, 5, 2, 400, 400, 800, 400)
        ]

        test_vehicles = [
            Vehicle(1, 100, 400, 1, 0),
            Vehicle(2, 300, 400, -1, 0)
        ]

        # 测试图构建
        graph_data = global_graph_builder.build_dynamic_graph(test_dqns, test_vehicles, epoch=1)
        debug_print(f"✅ 图构建成功: {graph_data['metadata']}")

        # 测试GNN推理
        with torch.no_grad():
            # 全局Q值
            global_q = global_gnn_model(graph_data)
            debug_print(f"✅ 全局GNN推理: {global_q.shape}")

            # 单个DQN Q值
            single_q = global_gnn_model(graph_data, dqn_id=1)
            debug_print(f"✅ 单个DQN推理: {single_q.shape}")

        # 测试GNN集成管理器
        enhanced_dqn = global_gnn_manager.enhance_dqn_with_gnn(test_dqns[0])
        debug_print(f"✅ GNN集成: {enhanced_dqn.__class__.__name__}")

        debug_print("🎉 完整系统测试通过！")
        return True

    except Exception as e:
        debug_print(f"❌ 系统测试失败: {e}")
        import traceback
        debug_print(f"详细错误: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    import torch

    if test_complete_system():
        debug_print("🚀 系统完全正常！现在可以开始正式训练了！")
    else:
        debug_print("❌ 需要进一步调试")