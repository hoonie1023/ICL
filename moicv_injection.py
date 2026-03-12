#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV 注入机制 (Injection Mechanism) 与 Qwen2 封装
=================================================

本文件实现：
    - MoICV_Qwen_Wrapper: 将 Hugging Face 的 Qwen2 类 LLM 与 Dual_MoICV_Layer 整合
      * 通过 Hook 机制，在指定的 Attention 层和 FFN 层注入 MoICV 生成的上下文向量
      * 支持在 forward_with_moicv 中同时得到 LLM 的输出、MoICV 正则化 Loss 与协同损失
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
    """
    n, m = x.shape
    assert n == m, "off_diagonal 只支持方阵输入"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class MoICV_Qwen_Wrapper(nn.Module):
    """
    将 Qwen2 类模型与 Dual_MoICV_Layer 结合的封装类。
    """

    def __init__(
        self,
        llm_model: nn.Module,
        moicv_layer: Dual_MoICV_Layer,
        attn_inject_layer_idx: int = 0,
        ffn_inject_layer_idx: int = 15,
    ) -> None:
        super().__init__()

        if not isinstance(moicv_layer, Dual_MoICV_Layer):
            raise TypeError(
                f"moicv_layer 必须是 Dual_MoICV_Layer 的实例，当前类型：{type(moicv_layer)}"
            )

        self.llm_model = llm_model
        self.moicv_layer = moicv_layer
        self.attn_inject_layer_idx = int(attn_inject_layer_idx)
        self.ffn_inject_layer_idx = int(ffn_inject_layer_idx)

        # MoICV 注入强度的可学习缩放因子
        self.gamma_attn = nn.Parameter(torch.tensor(2.0))
        self.gamma_ffn = nn.Parameter(torch.tensor(2.0))

        self.current_v_attn: Optional[torch.Tensor] = None
        self.current_v_ffn: Optional[torch.Tensor] = None

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
        if not (0 <= self.attn_inject_layer_idx < num_layers):
            raise IndexError(
                f"attn_inject_layer_idx={self.attn_inject_layer_idx} 超出层数范围 [0, {num_layers - 1}]"
            )
        if not (0 <= self.ffn_inject_layer_idx < num_layers):
            raise IndexError(
                f"ffn_inject_layer_idx={self.ffn_inject_layer_idx} 超出层数范围 [0, {num_layers - 1}]"
            )

        attn_layer = layers[self.attn_inject_layer_idx]
        attn_module_for_hook = getattr(attn_layer, "self_attn", attn_layer)

        def attn_post_hook(module: nn.Module, inputs: tuple, outputs: Any) -> Any:
            if self.current_v_attn is None:
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

            v_attn = self.current_v_attn
            if v_attn.device != attn_output.device:
                v_attn = v_attn.to(attn_output.device)
            if v_attn.dtype != attn_output.dtype:
                v_attn = v_attn.to(attn_output.dtype)

            v_attn_expanded = v_attn.unsqueeze(1)
            if v_attn_expanded.size(-1) != attn_output.size(-1):
                return outputs

            attn_output = attn_output + self.gamma_attn * v_attn_expanded

            if isinstance(outputs, tuple):
                return (attn_output,) + tuple(outputs[1:])
            return attn_output

        handle_attn = attn_module_for_hook.register_forward_hook(attn_post_hook, with_kwargs=False)
        self._hook_handles.append(handle_attn)

        ffn_layer = layers[self.ffn_inject_layer_idx]
        ffn_module_for_hook = getattr(ffn_layer, "mlp", ffn_layer)

        def ffn_post_hook(module: nn.Module, inputs: tuple, outputs: Any) -> Any:
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

            v_ffn_expanded = v_ffn.unsqueeze(1)
            if v_ffn_expanded.size(-1) != ffn_output.size(-1):
                return outputs

            ffn_output = ffn_output + self.gamma_ffn * v_ffn_expanded

            if isinstance(outputs, tuple):
                return (ffn_output,) + tuple(outputs[1:])
            return ffn_output

        handle_ffn = ffn_module_for_hook.register_forward_hook(ffn_post_hook, with_kwargs=False)
        self._hook_handles.append(handle_ffn)

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

        v_attn, v_ffn, logits_attn, logits_ffn = self.moicv_layer(query_features)

        z_a = F.normalize(v_attn, p=2, dim=-1)
        z_m = F.normalize(v_ffn, p=2, dim=-1)
        z_a_flat = z_a.view(-1, z_a.size(-1))
        z_m_flat = z_m.view(-1, z_m.size(-1))
        M = torch.matmul(z_a_flat.T, z_m_flat) / z_a_flat.size(0)
        on_diag = torch.diagonal(M).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(M).pow_(2).sum()
        syn_loss = on_diag + 1e-3 * off_diag

        try:
            ref_param = next(self.llm_model.parameters())
            device = ref_param.device
            dtype = ref_param.dtype
        except StopIteration:
            device = query_features.device
            dtype = query_features.dtype

        self.current_v_attn = v_attn.to(device=device, dtype=dtype)
        self.current_v_ffn = v_ffn.to(device=device, dtype=dtype)

        try:
            llm_outputs = self.llm_model(**model_kwargs)
        finally:
            self.current_v_attn = None
            self.current_v_ffn = None

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

