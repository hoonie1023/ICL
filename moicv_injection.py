#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV 注入机制 (Injection Mechanism) 与 Qwen2 封装
=================================================

单通路（single-pathway）架构：
  - Dual_MoICV_Layer 接收 query_features，输出：
        v_inject:        [B, H]    待注入的上下文向量
        routing_weights: [B, 8]    8 个专家的混合权重
  - 在指定的 transformer 层 (inject_layer_idx) 的 Self-Attention 输出端，
    对 hidden_states 执行一次残差注入：
        attn_output = attn_output + gamma_inject * v_inject
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from moicv_core import Dual_MoICV_Layer, MoICV_Loss


class MoICV_Qwen_Wrapper(nn.Module):
    """
    将 Qwen2 类模型与 Dual_MoICV_Layer 结合的封装类。

    单通路注入策略：
      - 仅在 inject_layer_idx 指定的 Self-Attention 输出端注入一次 v_inject。
    """

    def __init__(
        self,
        llm_model: nn.Module,
        moicv_layer: Dual_MoICV_Layer,
        inject_layer_idx: int = 15,
    ) -> None:
        super().__init__()

        if not isinstance(moicv_layer, Dual_MoICV_Layer):
            raise TypeError(
                f"moicv_layer 必须是 Dual_MoICV_Layer 的实例，当前类型：{type(moicv_layer)}"
            )

        self.llm_model = llm_model
        self.moicv_layer = moicv_layer
        self.inject_layer_idx = int(inject_layer_idx)

        # 单一注入强度因子 gamma_inject
        self.gamma_inject = nn.Parameter(torch.tensor(2.0))

        # 当前 batch 的待注入 MoICV 向量缓存
        self.current_v_inject: Optional[torch.Tensor] = None

        self._hook_handles = []
        self._setup_hooks()

    def _get_layers_module(self) -> nn.Module:
        lm = self.llm_model

        if hasattr(lm, "visual"):
            if hasattr(lm, "language_model"):
                inner = lm.language_model
                if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                    return inner.model.layers
                if hasattr(inner, "layers"):
                    return inner.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers

        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers

        raise AttributeError("无法在给定 llm_model 中找到合适的 transformer 层列表。")

    def _setup_hooks(self) -> None:
        layers = self._get_layers_module()

        num_layers = len(layers)
        if not (0 <= self.inject_layer_idx < num_layers):
            raise IndexError(
                f"inject_layer_idx={self.inject_layer_idx} 超出层数范围 [0, {num_layers - 1}]"
            )

        # 仅在指定层的 Self-Attention 输出端注册一个 Hook
        attn_layer = layers[self.inject_layer_idx]
        attn_module_for_hook = getattr(attn_layer, "self_attn", attn_layer)

        def attn_post_hook(module: nn.Module, inputs: tuple, outputs: Any) -> Any:
            if self.current_v_inject is None:
                return outputs

            if isinstance(outputs, tuple):
                if len(outputs) == 0:
                    return outputs
                attn_output = outputs[0]
            else:
                attn_output = outputs

            if not isinstance(attn_output, torch.Tensor):
                return outputs
            if attn_output.dim() != 3:
                return outputs

            v_inject = self.current_v_inject
            if v_inject.device != attn_output.device:
                v_inject = v_inject.to(attn_output.device)
            if v_inject.dtype != attn_output.dtype:
                v_inject = v_inject.to(attn_output.dtype)

            v_inject_expanded = v_inject.unsqueeze(1)
            if v_inject_expanded.size(-1) != attn_output.size(-1):
                return outputs

            attn_output = attn_output + self.gamma_inject * v_inject_expanded

            if isinstance(outputs, tuple):
                return (attn_output,) + tuple(outputs[1:])
            return attn_output

        handle_attn = attn_module_for_hook.register_forward_hook(attn_post_hook, with_kwargs=False)
        self._hook_handles.append(handle_attn)

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def forward_with_moicv(
        self,
        query_features: torch.Tensor,
        compute_moicv_loss: bool = False,
        moicv_loss_fn: Optional[MoICV_Loss] = None,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        if not isinstance(query_features, torch.Tensor):
            raise TypeError(
                f"query_features 必须是 torch.Tensor，当前类型：{type(query_features)}"
            )

        # 单通路 MoICV：直接获取 v_inject 与 routing_weights
        v_inject, routing_weights = self.moicv_layer(query_features)

        try:
            ref_param = next(self.llm_model.parameters())
            device = ref_param.device
            dtype = ref_param.dtype
        except StopIteration:
            device = query_features.device
            dtype = query_features.dtype

        self.current_v_inject = v_inject.to(device=device, dtype=dtype)

        try:
            llm_outputs = self.llm_model(**model_kwargs)
        finally:
            self.current_v_inject = None

        moicv_loss_val = None
        if compute_moicv_loss:
            if moicv_loss_fn is None:
                moicv_loss_fn = MoICV_Loss(alpha=0.1, beta=0.1)
            if not isinstance(moicv_loss_fn, MoICV_Loss):
                raise TypeError(
                    f"moicv_loss_fn 必须是 MoICV_Loss 或其子类实例，当前类型：{type(moicv_loss_fn)}"
                )
            # 新版正则接口：直接基于 routing_weights 与内部专家参数
            moicv_loss_val = moicv_loss_fn(
                routing_weights=routing_weights,
                moicv_layer=self.moicv_layer,
            )

        return {
            "llm_outputs": llm_outputs,
            "v_inject": v_inject,
            "routing_weights": routing_weights,
            "moicv_loss": moicv_loss_val,
        }

