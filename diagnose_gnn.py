# diagnose_gnn.py - GNN问题诊断
import sys
import subprocess
from DebugPrint import debug_print


def diagnose_gnn_issue():
    """诊断GNN问题"""
    debug_print("=== GNN问题诊断 ===")

    # 1. 检查Python环境
    debug_print(f"Python路径: {sys.executable}")
    debug_print(f"Python版本: {sys.version}")

    # 2. 检查SSL模块
    try:
        import ssl
        debug_print(f"✅ SSL模块: {ssl.OPENSSL_VERSION}")
    except ImportError as e:
        debug_print(f"❌ SSL模块导入失败: {e}")
        return False

    # 3. 检查torch_geometric
    try:
        import torch_geometric
        debug_print(f"✅ torch_geometric: {torch_geometric.__version__}")

        # 检查子模块
        from torch_geometric.nn import GATConv
        debug_print("✅ GATConv导入成功")

        return True
    except ImportError as e:
        debug_print(f"❌ torch_geometric导入失败: {e}")
        return False


def fix_suggestions():
    """提供修复建议"""
    debug_print("=== 修复建议 ===")

    suggestions = [
        "1. 运行: conda install openssl -y",
        "2. 运行: conda install python=3.8 -y",
        "3. 运行: pip install --upgrade torch-geometric",
        "4. 或者创建新环境: conda create -n v2x_gnn python=3.8",
        "5. 安装: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia",
        "6. 安装: pip install torch-geometric"
    ]

    for suggestion in suggestions:
        debug_print(suggestion)


if __name__ == "__main__":
    if diagnose_gnn_issue():
        debug_print("🎉 GNN环境正常！")
    else:
        fix_suggestions()