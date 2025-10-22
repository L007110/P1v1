# AttentionMechanism.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from logger import debug, debug_print
from Parameters import ATTENTION_HEADS

class MultiHeadAttention(nn.Module):
    """
    标准多头注意力机制
    支持自注意力和交叉注意力
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换矩阵
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        debug(f"MultiHeadAttention initialized: d_model={d_model}, heads={num_heads}")

    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k]

        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k] (if return_attention)
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)

        # 线性变换 + 分头
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力到value
        context = torch.matmul(attention_weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )

        # 输出变换
        output = self.w_o(context)

        if return_attention:
            return output, attention_weights
        return output


class HierarchicalAttention(nn.Module):
    """
    层次化注意力机制
    节点级 -> 边级 -> 图级 三层注意力
    """

    def __init__(self, node_dim, edge_dim, graph_dim, num_heads=4, dropout=0.1):
        super(HierarchicalAttention, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.graph_dim = graph_dim
        self.num_heads = num_heads

        # 节点级自注意力
        self.node_attention = MultiHeadAttention(node_dim, num_heads, dropout)

        # 边级注意力（节点到边）
        self.edge_attention = MultiHeadAttention(edge_dim, num_heads, dropout)

        # 图级注意力（边到图）
        self.graph_attention = MultiHeadAttention(graph_dim, num_heads, dropout)

        # 层归一化
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.graph_norm = nn.LayerNorm(graph_dim)

        # 特征融合
        self.fusion_layer = nn.Linear(node_dim + edge_dim + graph_dim, graph_dim)

        debug(f"HierarchicalAttention initialized: node_dim={node_dim}, "
              f"edge_dim={edge_dim}, graph_dim={graph_dim}")

    def forward(self, node_features, edge_features, graph_features,
                node_mask=None, edge_mask=None, return_attentions=False):
        """
        Args:
            node_features: [batch_size, num_nodes, node_dim]
            edge_features: [batch_size, num_edges, edge_dim]
            graph_features: [batch_size, num_graphs, graph_dim]
        """
        attentions = {}

        # 1. 节点级自注意力
        node_enhanced, node_attn = self.node_attention(
            node_features, node_features, node_features,
            mask=node_mask, return_attention=True
        )
        node_enhanced = self.node_norm(node_features + node_enhanced)
        attentions['node'] = node_attn

        # 2. 边级注意力
        edge_enhanced, edge_attn = self.edge_attention(
            edge_features, edge_features, edge_features,
            mask=edge_mask, return_attention=True
        )
        edge_enhanced = self.edge_norm(edge_features + edge_enhanced)
        attentions['edge'] = edge_attn

        # 3. 图级注意力
        graph_enhanced, graph_attn = self.graph_attention(
            graph_features, graph_features, graph_features,
            return_attention=True
        )
        graph_enhanced = self.graph_norm(graph_features + graph_enhanced)
        attentions['graph'] = graph_attn

        # 4. 层次特征融合
        # 池化节点特征到图级别
        node_pooled = torch.mean(node_enhanced, dim=1, keepdim=True)  # [batch_size, 1, node_dim]
        edge_pooled = torch.mean(edge_enhanced, dim=1, keepdim=True)  # [batch_size, 1, edge_dim]

        # 拼接所有层次特征
        fused_features = torch.cat([node_pooled, edge_pooled, graph_enhanced], dim=-1)
        final_output = self.fusion_layer(fused_features)

        if return_attentions:
            return final_output, attentions
        return final_output


class TemporalAttention(nn.Module):
    """
    时序注意力机制
    处理时间序列数据，支持位置编码
    """

    def __init__(self, d_model, num_heads, seq_len, dropout=0.1):
        super(TemporalAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len

        # 多头注意力
        self.temporal_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # 位置编码
        self.position_encoding = PositionalEncoding(d_model, seq_len, dropout)

        # 层归一化和前馈网络
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)

        debug(f"TemporalAttention initialized: d_model={d_model}, "
              f"heads={num_heads}, seq_len={seq_len}")

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: [batch_size, seq_len, d_model] 时序数据
        """
        # 位置编码
        x = self.position_encoding(x)

        # 时序自注意力
        if return_attention:
            temporal_out, attention_weights = self.temporal_attention(
                x, x, x, mask=mask, return_attention=True
            )
        else:
            temporal_out = self.temporal_attention(x, x, x, mask=mask)

        # 残差连接 + 层归一化
        x = self.norm1(x + temporal_out)

        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if return_attention:
            return x, attention_weights
        return x


class SpatialTemporalAttention(nn.Module):
    """
    时空融合注意力机制
    同时处理空间和图结构信息以及时序信息
    """

    def __init__(self, spatial_dim, temporal_dim, num_heads=4, seq_len=10, dropout=0.1):
        super(SpatialTemporalAttention, self).__init__()

        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.num_heads = num_heads

        # 空间注意力（图结构）
        self.spatial_attention = MultiHeadAttention(spatial_dim, num_heads, dropout)

        # 时序注意力
        self.temporal_attention = TemporalAttention(temporal_dim, num_heads, seq_len, dropout)

        # 时空交叉注意力
        self.cross_attention = MultiHeadAttention(spatial_dim, num_heads, dropout)

        # 特征融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, spatial_dim),
            nn.Sigmoid()
        )

        self.output_projection = nn.Linear(spatial_dim, spatial_dim)

        debug(f"SpatialTemporalAttention initialized: spatial_dim={spatial_dim}, "
              f"temporal_dim={temporal_dim}, heads={num_heads}")

    def forward(self, spatial_features, temporal_features, spatial_mask=None,
                temporal_mask=None, return_attentions=False):
        """
        Args:
            spatial_features: [batch_size, num_nodes, spatial_dim] 空间特征
            temporal_features: [batch_size, seq_len, temporal_dim] 时序特征
        """
        attentions = {}

        batch_size, num_nodes, spatial_dim = spatial_features.size()
        seq_len = temporal_features.size(1)

        # 1. 空间注意力
        spatial_enhanced, spatial_attn = self.spatial_attention(
            spatial_features, spatial_features, spatial_features,
            mask=spatial_mask, return_attention=True
        )
        attentions['spatial'] = spatial_attn

        # 2. 时序注意力
        temporal_enhanced, temporal_attn = self.temporal_attention(
            temporal_features, mask=temporal_mask, return_attention=True
        )
        attentions['temporal'] = temporal_attn

        # 3. 时空交叉注意力
        # 将时序特征扩展到空间维度
        temporal_expanded = temporal_enhanced.unsqueeze(1).expand(
            batch_size, num_nodes, seq_len, temporal_enhanced.size(-1)
        )
        temporal_expanded = temporal_expanded.reshape(batch_size * num_nodes, seq_len, -1)

        # 空间特征扩展
        spatial_expanded = spatial_enhanced.unsqueeze(2).expand(
            batch_size, num_nodes, seq_len, spatial_dim
        )
        spatial_expanded = spatial_expanded.reshape(batch_size * num_nodes, seq_len, spatial_dim)

        # 交叉注意力
        cross_output, cross_attn = self.cross_attention(
            spatial_expanded, temporal_expanded, temporal_expanded,
            return_attention=True
        )
        cross_output = cross_output.reshape(batch_size, num_nodes, seq_len, spatial_dim)
        attentions['cross'] = cross_attn

        # 4. 时空特征融合
        # 平均池化时序维度
        cross_pooled = torch.mean(cross_output, dim=2)  # [batch_size, num_nodes, spatial_dim]

        # 门控融合
        fusion_input = torch.cat([spatial_enhanced, cross_pooled], dim=-1)
        gate = self.fusion_gate(fusion_input)

        fused_features = gate * spatial_enhanced + (1 - gate) * cross_pooled
        output = self.output_projection(fused_features)

        if return_attentions:
            return output, attentions
        return output


class PositionalEncoding(nn.Module):
    """
    位置编码（Transformer风格）
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GraphAwareAttention(nn.Module):
    """
    图感知注意力机制
    专门为图结构数据设计的多头注意力
    """

    def __init__(self, d_model, num_heads, dropout=0.1, use_edge_weights=True):
        super(GraphAwareAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_edge_weights = use_edge_weights

        # 注意力参数
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # 图结构参数
        self.edge_projection = nn.Linear(1, num_heads) if use_edge_weights else None

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        debug(f"GraphAwareAttention initialized: d_model={d_model}, "
              f"heads={num_heads}, use_edge_weights={use_edge_weights}")

    def forward(self, node_features, edge_index, edge_weights=None, mask=None, return_attention=False):
        """
        Args:
            node_features: [num_nodes, d_model]
            edge_index: [2, num_edges]
            edge_weights: [num_edges] or None
        """
        num_nodes = node_features.size(0)

        # 线性变换
        Q = self.w_q(node_features).view(num_nodes, self.num_heads, self.d_k).transpose(0, 1)
        K = self.w_k(node_features).view(num_nodes, self.num_heads, self.d_k).transpose(0, 1)
        V = self.w_v(node_features).view(num_nodes, self.num_heads, self.d_k).transpose(0, 1)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [num_heads, num_nodes, num_nodes]

        # 应用图结构信息（边权重）
        if self.use_edge_weights and edge_weights is not None:
            # 创建邻接矩阵
            adj_matrix = torch.zeros(num_nodes, num_nodes, device=node_features.device)
            adj_matrix[edge_index[0], edge_index[1]] = edge_weights

            # 投影边权重到注意力头
            if self.edge_projection is not None:
                edge_bias = self.edge_projection(edge_weights.unsqueeze(-1))  # [num_edges, num_heads]
                edge_bias_matrix = torch.zeros(num_nodes, num_nodes, self.num_heads,
                                               device=node_features.device)
                edge_bias_matrix[edge_index[0], edge_index[1]] = edge_bias
                edge_bias_matrix = edge_bias_matrix.permute(2, 0, 1)  # [num_heads, num_nodes, num_nodes]

                scores = scores + edge_bias_matrix

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力
        context = torch.matmul(attention_weights, V)  # [num_heads, num_nodes, d_k]
        context = context.transpose(0, 1).contiguous().view(num_nodes, self.d_model)

        # 输出变换
        output = self.w_o(context)

        if return_attention:
            return output, attention_weights
        return output


# 全局注意力机制实例
global_attention_mechanism = HierarchicalAttention(
    node_dim=64,
    edge_dim=32,
    graph_dim=128,
    num_heads=ATTENTION_HEADS
)


def test_attention_mechanisms():
    """测试注意力机制"""
    debug_print("Testing Attention Mechanisms...")

    # 测试多头注意力
    batch_size, seq_len, d_model = 2, 5, 64
    multihead_attn = MultiHeadAttention(d_model, num_heads=4)

    x = torch.randn(batch_size, seq_len, d_model)
    output = multihead_attn(x, x, x)
    debug(f"MultiHeadAttention: input {x.shape}, output {output.shape}")

    # 测试层次化注意力
    node_features = torch.randn(batch_size, 10, 64)
    edge_features = torch.randn(batch_size, 15, 32)
    graph_features = torch.randn(batch_size, 1, 128)

    hierarchical_attn = HierarchicalAttention(64, 32, 128, num_heads=4)
    output, attentions = hierarchical_attn(node_features, edge_features, graph_features, return_attentions=True)
    debug(f"HierarchicalAttention: output {output.shape}")
    debug(f"Attention shapes: { {k: v.shape for k, v in attentions.items()} }")

    debug_print("Attention mechanisms test completed!")


if __name__ == "__main__":
    set_debug_mode(True)
    test_attention_mechanisms()