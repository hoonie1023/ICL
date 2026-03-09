#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV (Mixture-of-Interleaved Context Vectors) 核心模块实现
===========================================================

本文件实现两个核心组件：
1. Dual_MoICV_Layer:
   - 双路（Attention / FFN）并行的专家系统
   - 每路包含 4 个视觉专家、4 个文本专家、1 个通用专家（通用专家不参与路由，权重恒为 1）
   - 使用 Top-2 稀疏路由，从 8 个特化专家中选择并混合

2. MoICV_Loss:
   - 针对 MoICV 结构的正则化 Loss
   - 包含：
       * 负载均衡 Loss_Balance：约束 8 个特化专家的使用均衡
       * 正交解耦 Loss_Ortho：约束视觉 / 文本专家在表示空间上尽量正交

依赖：
    pip install torch

使用方式（示例）：
    from moicv_core import Dual_MoICV_Layer, MoICV_Loss
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dual_MoICV_Layer(nn.Module):
    """
    Dual-Pathway MoICV 层

    功能：
    - 输入查询特征向量 query_features（通常来自 Qwen2.5 的某一层隐藏状态或 pooled 表示）
    - 通过两个 Router 分别对 Attention 路和 FFN 路的 8 个特化专家进行 Top-2 稀疏路由
    - 对被路由到的视觉 / 文本专家进行加权求和，并额外加上通用专家，得到最终的上下文向量

    输出：
    - v_attn: [batch_size, attn_dim]  Attention 路混合后的上下文向量
    - v_ffn:  [batch_size, ffn_dim]   FFN 路混合后的上下文向量
    - logits_attn: [batch_size, 8]    Attention 路 8 个特化专家的路由 logits（用于 Loss）
    - logits_ffn:  [batch_size, 8]    FFN 路 8 个特化专家的路由 logits（用于 Loss）
    """

    def __init__(self, query_dim: int, attn_dim: int, ffn_dim: int) -> None:
        """
        初始化 Dual_MoICV_Layer。

        参数：
            query_dim: 输入查询向量的维度（Router 的输入维度）
            attn_dim:  Attention 路专家向量的维度
            ffn_dim:   FFN 路专家向量的维度
        """
        super().__init__()

        if query_dim <= 0 or attn_dim <= 0 or ffn_dim <= 0:
            raise ValueError(
                f"query_dim / attn_dim / ffn_dim 必须为正整数，"
                f"当前为 query_dim={query_dim}, attn_dim={attn_dim}, ffn_dim={ffn_dim}"
            )

        self.query_dim = query_dim
        self.attn_dim = attn_dim
        self.ffn_dim = ffn_dim

        # ==========================
        # 1. 定义 Attention 路专家
        # ==========================
        # 4 个视觉专家: [4, attn_dim]
        self.E_attn_vis = nn.Parameter(
            torch.empty(4, attn_dim)
        )
        # 4 个文本专家: [4, attn_dim]
        self.E_attn_text = nn.Parameter(
            torch.empty(4, attn_dim)
        )
        # 1 个通用专家: [1, attn_dim]，不参与路由，直接相加
        self.E_attn_general = nn.Parameter(
            torch.empty(1, attn_dim)
        )

        # ======================
        # 2. 定义 FFN 路专家
        # ======================
        self.E_ffn_vis = nn.Parameter(
            torch.empty(4, ffn_dim)
        )
        self.E_ffn_text = nn.Parameter(
            torch.empty(4, ffn_dim)
        )
        self.E_ffn_general = nn.Parameter(
            torch.empty(1, ffn_dim)
        )

        # ======================
        # 3. 双路 Router
        # ======================
        # 仅对 8 个特化专家（4 视觉 + 4 文本）进行路由
        self.router_attn = nn.Linear(query_dim, 8)
        self.router_ffn = nn.Linear(query_dim, 8)

        # 参数初始化（可以根据需要使用更复杂的初始化策略）
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """对专家向量和 Router 进行初始化。"""
        # 使用较小的均匀分布初始化专家参数，避免初始过大
        with torch.no_grad():
            for p in [
                self.E_attn_vis,
                self.E_attn_text,
                self.E_attn_general,
                self.E_ffn_vis,
                self.E_ffn_text,
                self.E_ffn_general,
            ]:
                nn.init.xavier_uniform_(p)
                # 为打破视觉 / 文本专家之间的完全对称性，在初始化时加入极小的高斯噪声，
                # 避免 Router 长时间处于完全对称的均匀状态。
                p.add_(0.0001 * torch.randn_like(p))

            # 对 Router 使用 Kaiming 或 Xavier 初始化，这里采用 Xavier
            nn.init.xavier_uniform_(self.router_attn.weight)
            nn.init.zeros_(self.router_attn.bias)
            nn.init.xavier_uniform_(self.router_ffn.weight)
            nn.init.zeros_(self.router_ffn.bias)

    @staticmethod
    def _top2_sparse_routing(logits: torch.Tensor) -> torch.Tensor:
        """
        对给定 logits 执行 Top-2 稀疏路由。

        步骤：
        1. 对每个样本的 8 个专家 logits 执行 topk(k=2)，得到 top2 的值和索引
        2. 构造一个全为 -inf 的张量，将 top2 位置的值填回（保持原值）
        3. 在专家维度上对 mask 后的 logits 执行 softmax，得到稀疏概率分布

        输入：
            logits: [batch_size, num_experts]
        输出：
            probs:  [batch_size, num_experts]，仅 top2 位置有非零概率
        """
        if logits.dim() != 2:
            raise ValueError(f"logits 维度必须为 2 (batch, num_experts)，当前为 {logits.shape}")

        batch_size, num_experts = logits.shape
        if num_experts < 2:
            raise ValueError("num_experts 必须至少为 2 才能进行 Top-2 路由")

        # 找到每个样本 top-2 的专家索引和值
        top_vals, top_idx = torch.topk(logits, k=2, dim=-1)  # [B, 2]

        # 构造全为 -inf 的张量，然后在 top2 位置填充值
        # 使用 logits 的 dtype 和 device，保证兼容性
        masked_logits = torch.full_like(logits, float("-inf"))
        # 利用 scatter_ 在指定索引位置写入 top-2 的值
        masked_logits.scatter_(dim=-1, index=top_idx, src=top_vals)

        # 对 mask 后的 logits 做 softmax，得到稀疏权重
        probs = F.softmax(masked_logits, dim=-1)
        return probs

    def _mix_experts(
        self,
        weights: torch.Tensor,
        E_vis: torch.Tensor,
        E_text: torch.Tensor,
        E_general: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据给定的路由权重与专家向量进行加权求和并加上通用专家。

        参数：
            weights:   [batch_size, 8] 的路由权重（前 4 维对应视觉专家，后 4 维对应文本专家）
            E_vis:     [4, dim]  视觉专家矩阵
            E_text:    [4, dim]  文本专家矩阵
            E_general: [1, dim]  通用专家向量（对 batch 维度广播相加）

        返回：
            v: [batch_size, dim] 加权后向量 + 通用专家
        """
        if weights.dim() != 2 or weights.size(1) != 8:
            raise ValueError(
                f"weights 形状必须为 [batch_size, 8]，当前为 {weights.shape}"
            )
        if E_vis.size(0) != 4 or E_text.size(0) != 4:
            raise ValueError(
                "E_vis 和 E_text 的第 0 维必须为 4（4 个专家），"
                f"当前为 E_vis={E_vis.shape}, E_text={E_text.shape}"
            )

        # 拆分出视觉 / 文本专家权重
        w_vis = weights[:, :4]   # [B, 4]
        w_text = weights[:, 4:]  # [B, 4]

        # 视觉专家加权求和: [B, 4] @ [4, dim] -> [B, dim]
        v_vis = torch.matmul(w_vis, E_vis)    # [B, dim]
        # 文本专家加权求和
        v_text = torch.matmul(w_text, E_text)  # [B, dim]

        # 通用专家：E_general: [1, dim]，在 batch 维度上广播
        # 注意：这里不对通用专家做缩放，权重恒为 1
        v_general = E_general  # [1, dim]，广播到 [B, dim]

        # 三者相加得到最终混合向量
        v = v_vis + v_text + v_general
        return v

    def forward(
        self, query_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。

        输入：
            query_features: [batch_size, query_dim]

        输出：
            v_attn:  [batch_size, attn_dim]
            v_ffn:   [batch_size, ffn_dim]
            logits_attn: [batch_size, 8]
            logits_ffn:  [batch_size, 8]
        """
        if query_features.dim() != 2 or query_features.size(-1) != self.query_dim:
            raise ValueError(
                f"query_features 形状必须为 [batch_size, {self.query_dim}]，"
                f"当前为 {query_features.shape}"
            )

        # 将 Router 与专家参数移动到与输入相同的 device / dtype（通常已经在同一设备上）
        # 这里不主动调用 to()，假设上层训练代码会将整个模块移动到相应设备

        # 1. 通过 Router 得到原始 logits（未归一化的路由分数）
        logits_attn = self.router_attn(query_features)  # [B, 8]
        logits_ffn = self.router_ffn(query_features)    # [B, 8]

        # 2. Top-2 稀疏路由，得到稀疏路由权重
        weights_attn = self._top2_sparse_routing(logits_attn)  # [B, 8]
        weights_ffn = self._top2_sparse_routing(logits_ffn)    # [B, 8]

        # 3. 基于权重混合视觉 / 文本专家，并加上通用专家
        v_attn = self._mix_experts(
            weights_attn,
            self.E_attn_vis,
            self.E_attn_text,
            self.E_attn_general,
        )  # [B, attn_dim]

        v_ffn = self._mix_experts(
            weights_ffn,
            self.E_ffn_vis,
            self.E_ffn_text,
            self.E_ffn_general,
        )  # [B, ffn_dim]

        return v_attn, v_ffn, logits_attn, logits_ffn


class MoICV_Loss(nn.Module):
    """
    MoICV 专用正则化 Loss 模块。

    包含两部分：
        1. Loss_Balance (负载均衡)：约束 8 个特化专家在 batch 内的平均激活率尽量均衡
        2. Loss_Ortho   (正交解耦)：约束视觉 / 文本专家在向量空间中尽量正交（低相关）
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.1) -> None:
        """
        初始化 MoICV_Loss。

        参数：
            alpha: Loss_Balance 的权重系数
            beta:  Loss_Ortho 的权重系数
        """
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def _balance_loss_single(logits: torch.Tensor) -> torch.Tensor:
        """
        对单一路由 logits 计算负载均衡 Loss。

        步骤：
        1. 对 logits 执行 softmax，得到每个样本对 8 个专家的概率分布 p_{b,e}
        2. 在 batch 维度上取平均，得到每个专家的平均激活率 mean_p_e
        3. 计算 mean_p_e 的方差，方差越大说明越不均衡 -> 惩罚越大
        """
        if logits.dim() != 2:
            raise ValueError(f"logits 维度必须为 2 (batch, num_experts)，当前为 {logits.shape}")

        probs = F.softmax(logits, dim=-1)  # [B, 8]
        mean_activation = probs.mean(dim=0)  # [8]

        # 无偏 / 有偏方差在这里影响不大，使用有偏方差 (unbiased=False)
        var = torch.var(mean_activation, unbiased=False)
        return var

    @staticmethod
    def _orthogonality_loss_pair(E_vis: torch.Tensor, E_text: torch.Tensor) -> torch.Tensor:
        """
        对给定的一对（视觉专家矩阵, 文本专家矩阵）计算正交解耦 Loss。

        思路：
        - 假设：
            E_vis:  [4, dim]
            E_text: [4, dim]
        - 先对每个向量进行 L2 归一化，得到单位向量
        - 计算所有视觉专家与所有文本专家之间的余弦相似度矩阵：
            sim = E_vis_norm @ E_text_norm^T  -> [4, 4]
        - 取 sim 的绝对值或平方并求和（或均值）
          这里使用平方后取均值：mean(sim^2)，对大相似度更敏感
        """
        if E_vis.dim() != 2 or E_text.dim() != 2:
            raise ValueError(
                f"E_vis 和 E_text 都必须为二维矩阵，当前为 E_vis={E_vis.shape}, E_text={E_text.shape}"
            )
        if E_vis.size(0) != 4 or E_text.size(0) != 4:
            raise ValueError(
                f"E_vis 和 E_text 的第 0 维必须为 4（4 个专家），当前为 E_vis={E_vis.shape}, E_text={E_text.shape}"
            )

        # L2 归一化，避免除以 0：添加一个极小 epsilon
        eps = 1e-8
        E_vis_norm = F.normalize(E_vis, p=2, dim=-1, eps=eps)   # [4, dim]
        E_text_norm = F.normalize(E_text, p=2, dim=-1, eps=eps)  # [4, dim]

        # 计算所有视觉专家与所有文本专家之间的余弦相似度
        sim = torch.matmul(E_vis_norm, E_text_norm.t())  # [4, 4]

        # 使用平方的方式更强惩罚高相似度
        loss = torch.mean(sim ** 2)
        return loss

    def forward(
        self,
        logits_attn: torch.Tensor,
        logits_ffn: torch.Tensor,
        moicv_layer: Dual_MoICV_Layer,
    ) -> torch.Tensor:
        """
        计算 MoICV 总正则化 Loss。

        参数：
            logits_attn: [batch_size, 8] Attention 路 Router 输出 logits
            logits_ffn:  [batch_size, 8] FFN 路 Router 输出 logits
            moicv_layer: Dual_MoICV_Layer 实例，用于访问专家参数

        返回：
            total_loss: 标量张量，等于 alpha * loss_balance + beta * loss_ortho
        """
        if not isinstance(moicv_layer, Dual_MoICV_Layer):
            raise TypeError(
                f"moicv_layer 必须是 Dual_MoICV_Layer 的实例，当前为 {type(moicv_layer)}"
            )

        # --------- 1. 负载均衡 Loss ---------
        loss_balance_attn = self._balance_loss_single(logits_attn)
        loss_balance_ffn = self._balance_loss_single(logits_ffn)
        loss_balance = 0.5 * (loss_balance_attn + loss_balance_ffn)

        # --------- 2. 正交解耦 Loss ---------
        # Attention 路专家
        E_attn_vis = moicv_layer.E_attn_vis
        E_attn_text = moicv_layer.E_attn_text
        # FFN 路专家
        E_ffn_vis = moicv_layer.E_ffn_vis
        E_ffn_text = moicv_layer.E_ffn_text

        loss_ortho_attn = self._orthogonality_loss_pair(E_attn_vis, E_attn_text)
        loss_ortho_ffn = self._orthogonality_loss_pair(E_ffn_vis, E_ffn_text)
        loss_ortho = 0.5 * (loss_ortho_attn + loss_ortho_ffn)

        # --------- Debug 探针：仅在前若干步打印一次 MoICV 各部分 Loss & 路由概率 ---------
        # 说明：
        #   1. 不使用 .item() 转成 Python float 再参与计算，防止打断计算图；
        #   2. 这里只在前 5 次调用时打印一次，避免刷屏；
        #   3. 同时打印 attn 路第一个样本的路由概率，帮助判断是否为绝对均匀分布。
        if not hasattr(self, "_debug_steps"):
            self._debug_steps = 0
        if self._debug_steps < 5:
            try:
                with torch.no_grad():
                    probs_attn = F.softmax(logits_attn, dim=-1)  # [B, 8]
                    first_probs = probs_attn[0].detach().cpu().tolist()
                print(
                    "[MoICV_Loss Debug] "
                    f"loss_balance_attn={loss_balance_attn.detach().cpu().item():.6f}, "
                    f"loss_balance_ffn={loss_balance_ffn.detach().cpu().item():.6f}, "
                    f"loss_ortho_attn={loss_ortho_attn.detach().cpu().item():.6f}, "
                    f"loss_ortho_ffn={loss_ortho_ffn.detach().cpu().item():.6f}, "
                    f"probs_attn[0]={first_probs}"
                )
            except Exception:
                # 避免任何打印错误影响前向传播
                pass
            self._debug_steps += 1

        # --------- 3. 总 Loss ---------
        total_loss = self.alpha * loss_balance + self.beta * loss_ortho
        return total_loss


if __name__ == "__main__":
    """
    简单自测：
    - 构造一个 Dual_MoICV_Layer 实例
    - 使用随机 query_features 跑一遍 forward
    - 使用 MoICV_Loss 计算一次 Loss
    - 打印各个输出的 shape 和 Loss 值，验证整体流程可以跑通
    """

    torch.manual_seed(0)

    batch_size = 16
    query_dim = 128
    attn_dim = 64
    ffn_dim = 96

    # 构造 MoICV 层
    moicv_layer = Dual_MoICV_Layer(
        query_dim=query_dim,
        attn_dim=attn_dim,
        ffn_dim=ffn_dim,
    )

    # 构造 Dummy 输入（模拟来自 Qwen2.5 的某层表示）
    query_features = torch.randn(batch_size, query_dim)

    # 前向传播
    v_attn, v_ffn, logits_attn, logits_ffn = moicv_layer(query_features)

    print("=== Dual_MoICV_Layer 前向传播结果 ===")
    print(f"query_features shape: {query_features.shape}")
    print(f"v_attn shape:         {v_attn.shape}")
    print(f"v_ffn shape:          {v_ffn.shape}")
    print(f"logits_attn shape:    {logits_attn.shape}")
    print(f"logits_ffn shape:     {logits_ffn.shape}")

    # 检查形状是否符合预期
    assert v_attn.shape == (batch_size, attn_dim)
    assert v_ffn.shape == (batch_size, ffn_dim)
    assert logits_attn.shape == (batch_size, 8)
    assert logits_ffn.shape == (batch_size, 8)

    # 构造 Loss 模块并计算正则化 Loss
    loss_fn = MoICV_Loss(alpha=0.1, beta=0.1)
    reg_loss = loss_fn(logits_attn, logits_ffn, moicv_layer)

    print("\n=== MoICV_Loss 计算结果 ===")
    print(f"Regularization loss: {reg_loss.item():.6f}")

    print("\n自测通过：Dual_MoICV_Layer 与 MoICV_Loss 可以正常前向计算。")


