#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV 注入机制 (Injection Mechanism) 与 Qwen2 封装
=================================================

本文件实现：
    - MoICV_Qwen_Wrapper: 将 Hugging Face 的 Qwen2 类 LLM 与 Dual_MoICV_Layer 整合
      * 通过 Hook 机制，在指定的 Attention 层和 FFN 层注入 MoICV 生成的上下文向量
      * 支持在 forward_with_moicv 中同时得到 LLM 的输出与 MoICV 的正则化 Loss

说明：
    - 代码以 Qwen2ForCausalLM 为典型目标，但对任何具有
        model.layers[i].self_attn
        model.layers[i].mlp
      结构的因果语言模型都适用。
    - 为了保证本文件可独立运行进行自测，__main__ 中使用一个简化版 Dummy 模型。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from moicv_core import Dual_MoICV_Layer, MoICV_Loss


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    返回方阵 x 的所有非对角线元素，展平为一维张量。

    该实现与 Barlow Twins 论文中使用的 off-diagonal 提取逻辑保持一致：
      - 先展平成一维，丢弃最后一个元素
      - 再 reshape 为 (n-1, n+1)，裁剪掉每行的首列
      - 最后再次展平得到 (n*n - n,) 个非对角线元素
    """
    n, m = x.shape
    assert n == m, "off_diagonal 只支持方阵输入"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class MoICV_Qwen_Wrapper(nn.Module):
    """
    将 Qwen2 类模型与 Dual_MoICV_Layer 结合的封装类。

    核心思想：
        - 在模型某一层的 Self-Attention 之前注入 v_attn（Attention 路上下文向量）
        - 在模型某一层的 MLP 之前注入 v_ffn（FFN 路上下文向量）
        - 两个注入均通过 forward_pre_hook 完成，不侵入原始模型代码
    """

    def __init__(
        self,
        llm_model: nn.Module,
        moicv_layer: Dual_MoICV_Layer,
        attn_inject_layer_idx: int = 0,
        ffn_inject_layer_idx: int = 15,
    ) -> None:
        """
        初始化 MoICV_Qwen_Wrapper。

        参数：
            llm_model:    已通过 Hugging Face 加载好的因果语言模型
                          典型示例：Qwen2ForCausalLM / Qwen2VLForConditionalGeneration 等
            moicv_layer:  事先实例化好的 Dual_MoICV_Layer
            attn_inject_layer_idx: 在第几层的 Attention 前注入 v_attn
            ffn_inject_layer_idx:  在第几层的 FFN(MLP) 前注入 v_ffn
        """
        super().__init__()

        if not isinstance(moicv_layer, Dual_MoICV_Layer):
            raise TypeError(
                f"moicv_layer 必须是 Dual_MoICV_Layer 的实例，当前类型：{type(moicv_layer)}"
            )

        self.llm_model = llm_model
        self.moicv_layer = moicv_layer
        self.attn_inject_layer_idx = int(attn_inject_layer_idx)
        self.ffn_inject_layer_idx = int(ffn_inject_layer_idx)

        # MoICV 注入强度的可学习缩放因子（分别用于 Attention 路与 FFN 路）
        self.gamma_attn = nn.Parameter(torch.tensor(2.0))
        self.gamma_ffn = nn.Parameter(torch.tensor(2.0))

        # 当前 batch 的 MoICV 向量缓存
        # 这些向量会在 forward_with_moicv 中被设置，在 Hook 中被读取和使用
        self.current_v_attn: Optional[torch.Tensor] = None  # [B, attn_dim]
        self.current_v_ffn: Optional[torch.Tensor] = None   # [B, ffn_dim]

        # 保存 Hook 句柄，以便需要时移除
        self._hook_handles = []

        # 检查模型结构并挂载 Hook
        self._setup_hooks()

    # ------------------------------------------------------------------
    # Hook 相关逻辑
    # ------------------------------------------------------------------
    def _get_layers_module(self) -> nn.Module:
        """
        尝试获取 LLM 的层列表容器 (通常为 model.layers)。
        对 Qwen2 / Qwen2.5-VL / GPT 系列模型，这通常是若干嵌套后的 *.model.layers。

        为了兼容多模态 Qwen2.5-VL，我们采用“防御式”层定位策略：
            1. 若模型具有典型的 VL 特征属性 visual / language_model，则优先在 language_model 下查找；
            2. 否则退回到通用的 llm_model.model.layers / llm_model.layers。
        """
        lm = self.llm_model

        # --- 1. Qwen2.5-VL 等多模态模型的优先路径 ---
        # 一般会同时包含 visual / language_model 等属性
        if hasattr(lm, "visual"):
            # 典型结构：llm_model.language_model.model.layers
            if hasattr(lm, "language_model"):
                inner = lm.language_model
                if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                    return inner.model.layers
                if hasattr(inner, "layers"):
                    return inner.layers

            # 有些实现可能直接将 text backbone 挂在 model.layers 下
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers

        # --- 2. 通用 Qwen2 / GPT 等纯文本模型 ---
        # 最常见结构：llm_model.model.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers

        # 退而求其次：直接在 llm_model 上寻找 layers
        if hasattr(lm, "layers"):
            return lm.layers

        raise AttributeError(
            "无法在给定 llm_model 中找到合适的 transformer 层列表。\n"
            "已尝试访问以下路径：\n"
            "  - model.language_model.model.layers\n"
            "  - model.language_model.layers\n"
            "  - model.model.layers\n"
            "  - model.layers\n"
            "请检查传入的模型是否为兼容的 Qwen2 / Qwen2.5-VL 结构。"
        )

    def _setup_hooks(self) -> None:
        """检查层索引并在指定层挂载前向 Hook。"""
        layers = self._get_layers_module()

        num_layers = len(layers)
        if not (0 <= self.attn_inject_layer_idx < num_layers):
            raise IndexError(
                f"attn_inject_layer_idx={self.attn_inject_layer_idx} 超出层数范围 [0, {num_layers - 1}]"
            )
        if not (0 <= self.ffn_inject_layer_idx < num_layers):
            raise IndexError(
                f"ffn_inject_layer_idx={self.ffn_inject_layer_idx} 超出层数范围 [0, {num_layers - 1}]"
            )

        # Attention 注入 Hook：优先挂载到具体的 self_attn 子模块，其次整个层
        attn_layer = layers[self.attn_inject_layer_idx]
        attn_module_for_hook = getattr(attn_layer, "self_attn", attn_layer)

        def attn_post_hook(module: nn.Module, inputs: tuple, outputs: Any) -> Any:
            """
            在 Self-Attention 输出端注入 v_attn，对应 M²IV 中的残差流注入：
                h_l = h_{l-1} + a_l + α * v^a + ...

            Qwen2 的 Attention 通常返回：
                - Tensor: attn_output
                - 或 Tuple: (attn_output, past_key_values, ...)
            """
            if self.current_v_attn is None:
                return outputs

            # 解析 Attention 的主输出张量 attn_output
            if isinstance(outputs, tuple):
                if len(outputs) == 0:
                    return outputs
                attn_output = outputs[0]
            else:
                attn_output = outputs

            if not isinstance(attn_output, torch.Tensor):
                return outputs

            # attn_output: [B, S, D]；current_v_attn: [B, D]
            if attn_output.dim() != 3:
                # 为了健壮性，如果形状异常则直接跳过注入
                return outputs

            v_attn = self.current_v_attn

            # 自动对齐 device / dtype
            if v_attn.device != attn_output.device:
                v_attn = v_attn.to(attn_output.device)
            if v_attn.dtype != attn_output.dtype:
                v_attn = v_attn.to(attn_output.dtype)

            # 在 seq_len 维度上广播： [B, 1, D]
            v_attn_expanded = v_attn.unsqueeze(1)
            if v_attn_expanded.size(-1) != attn_output.size(-1):
                # 维度不匹配则跳过注入，但不报错，以防中途改变 hidden_size
                return outputs

            attn_output = attn_output + self.gamma_attn * v_attn_expanded

            # 重新组装输出结构，保持与原始 outputs 的元组 / 张量形式一致
            if isinstance(outputs, tuple):
                new_outputs = (attn_output,) + tuple(outputs[1:])
                return new_outputs
            return attn_output

        handle_attn = attn_module_for_hook.register_forward_hook(attn_post_hook, with_kwargs=False)
        self._hook_handles.append(handle_attn)

        # FFN 注入 Hook：优先挂载到具体的 mlp 子模块，其次整个层
        ffn_layer = layers[self.ffn_inject_layer_idx]
        ffn_module_for_hook = getattr(ffn_layer, "mlp", ffn_layer)

        def ffn_post_hook(module: nn.Module, inputs: tuple, outputs: Any) -> Any:
            """
            在 MLP 输出端注入 v_ffn，对应 M²IV 中的残差流注入：
                h_l = ... + m_l + α * v^m

            输出通常为：
                - Tensor: ffn_output
                - 或 Tuple: (ffn_output, ...)
            """
            if self.current_v_ffn is None:
                return outputs

            if isinstance(outputs, tuple):
                if len(outputs) == 0:
                    return outputs
                ffn_output = outputs[0]
            else:
                ffn_output = outputs

            if not isinstance(ffn_output, torch.Tensor):
                return outputs

            if ffn_output.dim() != 3:
                return outputs

            v_ffn = self.current_v_ffn

            if v_ffn.device != ffn_output.device:
                v_ffn = v_ffn.to(ffn_output.device)
            if v_ffn.dtype != ffn_output.dtype:
                v_ffn = v_ffn.to(ffn_output.dtype)

            # [B, D] -> [B, 1, D] 广播到整个序列
            v_ffn_expanded = v_ffn.unsqueeze(1)
            if v_ffn_expanded.size(-1) != ffn_output.size(-1):
                return outputs

            ffn_output = ffn_output + self.gamma_ffn * v_ffn_expanded

            if isinstance(outputs, tuple):
                new_outputs = (ffn_output,) + tuple(outputs[1:])
                return new_outputs
            return ffn_output

        handle_ffn = ffn_module_for_hook.register_forward_hook(ffn_post_hook, with_kwargs=False)
        self._hook_handles.append(handle_ffn)

    def remove_hooks(self) -> None:
        """移除之前注册的所有 Hook（如需彻底卸载注入机制时调用）。"""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # 核心前向接口
    # ------------------------------------------------------------------
    def forward_with_moicv(
        self,
        query_features: torch.Tensor,
        compute_moicv_loss: bool = False,
        moicv_loss_fn: Optional[MoICV_Loss] = None,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        使用 MoICV 注入机制进行一次 LLM 前向推理。

        参数：
            query_features: [batch_size, query_dim]，用于 MoICV 路由的查询特征
            compute_moicv_loss: 是否同时计算 MoICV 正则化 Loss
            moicv_loss_fn:     可选的 MoICV_Loss 实例；若为 None 且 compute_moicv_loss=True，
                               将自动构造一个默认 MoICV_Loss(alpha=0.1, beta=0.1)
            **model_kwargs:    传递给 LLM 的标准输入参数，例如：
                               input_ids, attention_mask, position_ids, labels 等

        返回：
            一个字典，包含：
                - "llm_outputs": 原始 LLM 的输出（通常是 HuggingFace 的 ModelOutput）
                - "v_attn":      [B, attn_dim] MoICV Attention 向量
                - "v_ffn":       [B, ffn_dim]  MoICV FFN 向量
                - "logits_attn": [B, 8]       Attention 路路由 logits
                - "logits_ffn":  [B, 8]       FFN 路路由 logits
                - "moicv_loss":  (可选) MoICV 正则化 Loss（标量张量或 None）
        """
        if not isinstance(query_features, torch.Tensor):
            raise TypeError(
                f"query_features 必须是 torch.Tensor，当前类型：{type(query_features)}"
            )

        # 1. 通过 MoICV 层计算上下文向量和路由 logits
        v_attn, v_ffn, logits_attn, logits_ffn = self.moicv_layer(query_features)

        # 2. 基于 v_attn / v_ffn 计算 M²IV 动态协同损失 (Dynamic Synergistic Loss, L_syn)
        #    注意：这里在注入到主干模型之前，对两个分支的表征进行语义对齐约束。
        z_a = F.normalize(v_attn, p=2, dim=-1)
        z_m = F.normalize(v_ffn, p=2, dim=-1)

        # 拍平 batch / seq 维度，统一成 [N, D]
        z_a_flat = z_a.view(-1, z_a.size(-1))
        z_m_flat = z_m.view(-1, z_m.size(-1))

        # Cross-view correlation matrix: [D, D]
        M = torch.matmul(z_a_flat.T, z_m_flat) / z_a_flat.size(0)

        # Barlow Twins 风格的协同损失：对角线趋近于 1，非对角线趋近于 0
        on_diag = torch.diagonal(M).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(M).pow_(2).sum()
        syn_loss = on_diag + 1e-3 * off_diag

        # 3. 缓存 v_attn / v_ffn，供各自 Hook 在 LLM forward 过程中使用
        #    注意：必须确保 device 与 dtype 与模型对齐
        #    这里选择与模型第一个参数保持一致
        try:
            ref_param = next(self.llm_model.parameters())
            device = ref_param.device
            dtype = ref_param.dtype
        except StopIteration:
            # 非常规情况：模型没有参数（几乎不可能），则退回使用 query_features 的 device/dtype
            device = query_features.device
            dtype = query_features.dtype

        self.current_v_attn = v_attn.to(device=device, dtype=dtype)
        self.current_v_ffn = v_ffn.to(device=device, dtype=dtype)

        # 4. 调用底层 LLM 的 forward
        try:
            llm_outputs = self.llm_model(**model_kwargs)
        finally:
            # 无论 forward 是否抛异常，都要清空缓存，避免影响后续 batch
            self.current_v_attn = None
            self.current_v_ffn = None

        # 5.（可选）计算 MoICV 正则化 Loss
        moicv_loss_val = None
        if compute_moicv_loss:
            if moicv_loss_fn is None:
                moicv_loss_fn = MoICV_Loss(alpha=0.1, beta=0.1)
            if not isinstance(moicv_loss_fn, MoICV_Loss):
                raise TypeError(
                    f"moicv_loss_fn 必须是 MoICV_Loss 或其子类实例，当前类型：{type(moicv_loss_fn)}"
                )
            moicv_loss_val = moicv_loss_fn(
                logits_attn=logits_attn,
                logits_ffn=logits_ffn,
                moicv_layer=self.moicv_layer,
            )

        return {
            "llm_outputs": llm_outputs,
            "v_attn": v_attn,
            "v_ffn": v_ffn,
            "logits_attn": logits_attn,
            "logits_ffn": logits_ffn,
            "moicv_loss": moicv_loss_val,
            "syn_loss": syn_loss,
        }


# ======================================================================
# 自测代码：使用 Dummy 模型验证注入逻辑是否能跑通
# ======================================================================

if __name__ == "__main__":
    """
    说明：
        - 这里不会真正加载 Qwen2 模型，而是构造一个结构兼容的 Dummy LLM
        - 仅用于验证 Hook 注入与 forward_with_moicv 的基本流程是否通畅
    """

    class DummySelfAttn(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.proj = nn.Linear(hidden_size, hidden_size)

        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # 简单线性变换模拟 Attention
            return self.proj(hidden_states)

    class DummyMLP(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            x = self.fc1(hidden_states)
            x = self.act(x)
            x = self.fc2(x)
            return x

    class DummyLayer(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.self_attn = DummySelfAttn(hidden_size)
            self.mlp = DummyMLP(hidden_size)

        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            x = self.self_attn(hidden_states, *args, **kwargs)
            x = self.mlp(x)
            return x

    class DummyBackbone(nn.Module):
        def __init__(self, num_layers: int, hidden_size: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [DummyLayer(hidden_size) for _ in range(num_layers)]
            )

        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            x = hidden_states
            for layer in self.layers:
                x = layer(x, *args, **kwargs)
            return x

    class DummyLM(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.model = DummyBackbone(num_layers=num_layers, hidden_size=hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs: Any,
        ) -> Dict[str, torch.Tensor]:
            # attention_mask 在 Dummy 中不使用，仅保持接口兼容
            hidden_states = self.embed(input_ids)
            hidden_states = self.model(hidden_states)
            logits = self.lm_head(hidden_states)
            return {"logits": logits}

    # ---------------------- 开始自测 ----------------------
    torch.manual_seed(0)

    batch_size = 4
    seq_len = 8
    vocab_size = 100
    hidden_size = 32
    num_layers = 20

    # 1. 构造 Dummy LLM
    dummy_llm = DummyLM(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)

    # 2. 构造 Dual_MoICV_Layer
    query_dim = 16
    attn_dim = hidden_size   # 通常可与 hidden_size 对齐
    ffn_dim = hidden_size
    moicv_layer = Dual_MoICV_Layer(query_dim=query_dim, attn_dim=attn_dim, ffn_dim=ffn_dim)

    # 3. 构建 Wrapper，并指定注入层索引（0 层和 15 层）
    wrapper = MoICV_Qwen_Wrapper(
        llm_model=dummy_llm,
        moicv_layer=moicv_layer,
        attn_inject_layer_idx=0,
        ffn_inject_layer_idx=15,
    )

    # 4. 构造 Dummy 输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    query_features = torch.randn(batch_size, query_dim)

    # 5. 前向运行带 MoICV 注入的推理
    outputs = wrapper.forward_with_moicv(
        query_features=query_features,
        compute_moicv_loss=True,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    print("=== MoICV_Qwen_Wrapper 自测输出 ===")
    logits = outputs["llm_outputs"]["logits"]
    print(f"logits shape:       {logits.shape}")
    print(f"v_attn shape:       {outputs['v_attn'].shape}")
    print(f"v_ffn shape:        {outputs['v_ffn'].shape}")
    print(f"logits_attn shape:  {outputs['logits_attn'].shape}")
    print(f"logits_ffn shape:   {outputs['logits_ffn'].shape}")
    print(f"moicv_loss:         {outputs['moicv_loss'].item():.6f}")

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert outputs["v_attn"].shape[0] == batch_size
    assert outputs["v_ffn"].shape[0] == batch_size
    assert outputs["logits_attn"].shape == (batch_size, 8)
    assert outputs["logits_ffn"].shape == (batch_size, 8)

    print("\n自测通过：MoICV_Qwen_Wrapper 注入机制工作正常。")


