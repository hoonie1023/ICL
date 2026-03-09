#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV + Qwen2.5-7B 训练流水线脚本
================================

功能概述：
---------
1. 加载本地 Qwen2.5-7B 模型，使用 4-bit (NF4) 量化 + device_map="auto" 做显存压缩
2. 冻结 LLM 所有原始参数，仅训练 MoICV 专家层 (Dual_MoICV_Layer)
3. 从 expert_init_tensors.pt 中加载 896 维的专家初始化向量，使用 1D 插值将维度映射到 hidden_size (例如 3584)
4. 构造 MoICV_Qwen_Wrapper，将 MoICV 注入到指定 Attention / FFN 层
5. 使用 mini_mixed_dataset 作为示例训练数据集，运行一个简单的训练 Loop

注意事项：
---------
- 本脚本仅演示完整训练管线的实现方式，不建议直接长时间大规模训练；
  你可以根据实际情况接入 Trainer / Accelerate / DeepSpeed 等更完善的训练框架。
- 请提前将 Qwen2.5-7B 的权重下载到本地，并在 MODEL_PATH 中填入对应路径。

依赖安装（示例）：
    pip install torch torchvision
    pip install transformers accelerate
    pip install datasets tqdm

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers import AutoTokenizer, set_seed

# 对于 Qwen2.5-VL-7B-Instruct（多模态 VL 模型），AutoModelForCausalLM 无法直接加载，
# 会报 Qwen2_5_VLConfig 不被支持的错误。因此这里优先尝试导入专门的 VL 类；
# 若本地 transformers 版本较老不包含该类，则回退为 AutoModelForCausalLM（此时需换成纯文本 Qwen2.5 模型）。
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:  # 回退方案：使用通用 CausalLM（仅适用于纯文本 Qwen2.5）
    from transformers import AutoModelForCausalLM as QwenVLModel
from datasets import load_from_disk

from moicv_core import Dual_MoICV_Layer, MoICV_Loss
from moicv_injection import MoICV_Qwen_Wrapper


# ===========================
# 离线 / 环境配置
# ===========================

"""针对远程 48G 显存服务器的离线 / 高配显存优化：
- 启用完全离线模式，避免任何网络访问；
- 不使用 4-bit 量化，直接用 FP16 提升精度与吞吐；
- 强制将所有计算固定在单块 GPU (cuda:0) 上，避免掉回 CPU。
"""

# 若你在完全离线环境下训练，以下环境变量阻止 Transformers / Datasets 访问互联网
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# 注意：不再在代码中强行改写 CUDA_VISIBLE_DEVICES，避免把本来可见的 GPU 隐藏掉。
# 如需限制可见 GPU，请在启动脚本前通过环境变量或启动命令自行设置。

# 为了避免 OpenMP 相关的多线程冲突，在多数服务器环境下推荐限制线程数
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")


# ===========================
# 全局训练配置（可按需修改）
# ===========================

@dataclass
class TrainConfig:
    # 本地 Qwen2.5-7B 模型路径（请根据实际情况修改）
    MODEL_PATH: str = r"/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct"

    # mini 数据集路径
    DATASET_PATH: str = "./mini_mixed_dataset"

    # 专家初始化权重路径
    EXPERT_INIT_PATH: str = "./expert_init_tensors.pt"

    # 训练产物输出目录（用于区分不同设备/实验的 MoICV 权重）
    # 建议 GPU 版本单独放到一个子目录中，避免与 CPU 版本产物混淆
    OUTPUT_DIR: str = "./outputs_gpu"

    # 训练相关
    # 针对 48G 显存：可以安全地将 batch size 提高到 8（甚至 16）
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 1
    MAX_SEQ_LEN: int = 512
    # 为提升数值稳定性，将学习率从 2e-4 降低到 1e-5
    LEARNING_RATE: float = 1e-5
    WEIGHT_DECAY: float = 0.0

    # MoICV Loss 权重（在 MoICV_Loss 中已经设置 alpha=0.1, beta=0.1，这里只需与 CE Loss 相加）
    MOICV_ALPHA: float = 0.1
    MOICV_BETA: float = 0.1

    # 注入层索引（示例：在第 0 层 Attention 和第 15 层 FFN 注入）
    ATTN_INJECT_LAYER_IDX: int = 0
    FFN_INJECT_LAYER_IDX: int = 15

    # 随机种子
    SEED: int = 42


CFG = TrainConfig()


def get_main_device() -> torch.device:
    """
    获取主设备。

    优先使用单块 GPU：cuda:0。
    如果当前进程里确实没有 CUDA，就自动退回 CPU，而不是直接抛异常。
    同时打印一行 debug 信息，方便你确认到底脚本里看到的 CUDA 状态是什么。
    """
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    print(f"[DEBUG] torch.cuda.is_available()={cuda_available}, device_count={cuda_count}")

    if cuda_available and cuda_count > 0:
        # 强制绑定到第 0 号 GPU
        return torch.device("cuda:0")

    print(
        "[WARN] 当前进程内未检测到可用的 CUDA GPU，训练将退回到 CPU。\n"
        "如果你确认有 GPU，请检查：Python 环境 / 容器 / 驱动 / CUDA 版本是否与本脚本一致。"
    )
    return torch.device("cpu")


# ===========================
# 数据集包装
# ===========================

class MixedQADataset(Dataset):
    """
    一个简单的 PyTorch Dataset 包装器，将 Hugging Face Dataset 封装为 torch Dataset。
    每个样本包含：
        - query:  原始问题 + 选项描述
        - label:  正确答案（字符串）
    """

    def __init__(self, hf_dataset):
        super().__init__()
        self.ds = hf_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        return {
            "query": ex["query"],
            "label": ex["label"],
        }


def build_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    """
    返回一个用于 DataLoader 的 collate_fn，用于将文本批处理为模型输入张量。

    这里采用最简单的语言建模训练目标：
        text = f"Q: {query}\nA: {label}"
        input_ids 作为同时的输入和标签（即预测整句）。
    在实际任务中，你可以根据需要设计更严格的 prompt 模板和 label mask。
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        for ex in batch:
            q = ex["query"]
            a = ex["label"]
            text = f"Q: {q}\nA: {a}"
            texts.append(text)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # 简单起见，将 labels 设为 input_ids（对所有 token 做 LM 训练）
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


# ===========================
# 专家权重加载与维度拉伸
# ===========================

def resize_expert_tensor(expert: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    将一个形状为 [num_experts, dim] 的专家张量沿最后一维拉伸到 target_dim。

    这里采用 1D 线性插值的方式进行“维度对齐”：
        - 可以把每个专家向量视为长度为 dim 的一维信号
        - 使用 F.interpolate 在信号长度维度上做线性插值，将 896 拉伸到 hidden_size (例如 3584)

    步骤示意：
        原始: expert 形状 [4, 896]
        视作: [4, 1, 896] （batch=4, channel=1, length=896）
        插值: F.interpolate(..., size=target_dim, mode="linear") -> [4, 1, target_dim]
        还原: squeeze(1) -> [4, target_dim]
    """
    if expert.dim() != 2:
        raise ValueError(f"expert 必须是二维张量 [num_experts, dim]，当前为 {expert.shape}")

    num_experts, dim = expert.shape
    if dim == target_dim:
        return expert.clone()

    # [N, D] -> [N, 1, D]，在长度维上做插值
    expert_1d = expert.unsqueeze(1)  # [N, 1, D]
    expert_resized = F.interpolate(
        expert_1d,
        size=target_dim,
        mode="linear",
        align_corners=False,
    )  # [N, 1, target_dim]
    expert_resized = expert_resized.squeeze(1)  # [N, target_dim]
    return expert_resized


def load_and_assign_experts(
    moicv_layer: Dual_MoICV_Layer,
    expert_init_path: str,
    hidden_size: int,
) -> None:
    """
    从 expert_init_tensors.pt 加载 896 维的初始化专家向量，
    并使用 resize_expert_tensor 将其拉伸到 hidden_size，然后赋值到 moicv_layer 中。

    同时，将各路通用专家初始化为对应路特化专家的平均值：
        E_attn_general = mean( concat(E_attn_vis, E_attn_text), dim=0 )
        E_ffn_general  = mean( concat(E_ffn_vis,  E_ffn_text),  dim=0 )
    """
    if not os.path.exists(expert_init_path):
        raise FileNotFoundError(f"找不到专家初始化文件：{expert_init_path}")

    print(f"[INFO] 正在加载专家初始化权重：{expert_init_path}")
    init_state = torch.load(expert_init_path, map_location="cpu")

    required_keys = {"E_attn_vis", "E_ffn_vis", "E_attn_text", "E_ffn_text"}
    if not required_keys.issubset(init_state.keys()):
        raise KeyError(
            f"expert_init_tensors.pt 中缺少必要键：{required_keys - set(init_state.keys())}"
        )

    # 从文件中取出 896 维的初始化张量（形状 [4, 896]）
    E_attn_vis_896: torch.Tensor = init_state["E_attn_vis"]  # [4, 896]
    E_ffn_vis_896: torch.Tensor = init_state["E_ffn_vis"]
    E_attn_text_896: torch.Tensor = init_state["E_attn_text"]
    E_ffn_text_896: torch.Tensor = init_state["E_ffn_text"]

    print(f"[INFO] 原始专家维度（期望 896）：")
    print(f"  - E_attn_vis:  {tuple(E_attn_vis_896.shape)}")
    print(f"  - E_ffn_vis:   {tuple(E_ffn_vis_896.shape)}")
    print(f"  - E_attn_text: {tuple(E_attn_text_896.shape)}")
    print(f"  - E_ffn_text:  {tuple(E_ffn_text_896.shape)}")

    # 使用 1D 线性插值将 896 -> hidden_size
    E_attn_vis_resized = resize_expert_tensor(E_attn_vis_896, hidden_size)
    E_ffn_vis_resized = resize_expert_tensor(E_ffn_vis_896, hidden_size)
    E_attn_text_resized = resize_expert_tensor(E_attn_text_896, hidden_size)
    E_ffn_text_resized = resize_expert_tensor(E_ffn_text_896, hidden_size)

    print(f"[INFO] 拉伸后的专家维度（对齐 hidden_size={hidden_size}）：")
    print(f"  - E_attn_vis:  {tuple(E_attn_vis_resized.shape)}")
    print(f"  - E_ffn_vis:   {tuple(E_ffn_vis_resized.shape)}")
    print(f"  - E_attn_text: {tuple(E_attn_text_resized.shape)}")
    print(f"  - E_ffn_text:  {tuple(E_ffn_text_resized.shape)}")

    with torch.no_grad():
        # 目标 dtype：保持与 MoICV 层参数一致（在高配版本中通常为 torch.bfloat16）
        target_dtype = moicv_layer.E_attn_vis.dtype

        # 将拉伸后的权重转换到目标 dtype，并拷贝到 MoICV 层中
        moicv_layer.E_attn_vis.data.copy_(E_attn_vis_resized.to(dtype=target_dtype))
        moicv_layer.E_ffn_vis.data.copy_(E_ffn_vis_resized.to(dtype=target_dtype))
        moicv_layer.E_attn_text.data.copy_(E_attn_text_resized.to(dtype=target_dtype))
        moicv_layer.E_ffn_text.data.copy_(E_ffn_text_resized.to(dtype=target_dtype))

        # 通用专家初始化为特化专家的平均值
        # Attention 路：拼接 8 个专家 -> 求均值 -> [1, hidden_size]
        attn_all = torch.cat(
            [moicv_layer.E_attn_vis.data, moicv_layer.E_attn_text.data],
            dim=0,
        )  # [8, hidden_size]
        E_attn_general = attn_all.mean(dim=0, keepdim=True)  # [1, hidden_size]

        # FFN 路同理
        ffn_all = torch.cat(
            [moicv_layer.E_ffn_vis.data, moicv_layer.E_ffn_text.data],
            dim=0,
        )  # [8, hidden_size]
        E_ffn_general = ffn_all.mean(dim=0, keepdim=True)  # [1, hidden_size]

        # 确保通用专家同样使用与 LLM 一致的精度（在高配版本中为 bfloat16）
        moicv_layer.E_attn_general.data.copy_(E_attn_general.to(dtype=target_dtype))
        moicv_layer.E_ffn_general.data.copy_(E_ffn_general.to(dtype=target_dtype))

    print("[INFO] 专家参数已成功加载并对齐到模型 hidden_size。")


# ===========================
# 主训练流程
# ===========================

def main() -> None:
    cfg = CFG

    # 设置随机种子，保证实验可复现
    set_seed(cfg.SEED)

    main_device = get_main_device()
    print(f"[INFO] 主设备（强制）：{main_device}")

    # -----------------------
    # 1. 加载高精度 LLM（FP16，强制）
    # -----------------------
    # 针对 48G 显存 / 老架构 GPU，我们直接使用 float16 精度，
    # 避免 bfloat16 探测与回退逻辑带来的不确定性。
    model_dtype = torch.float16
    print("[INFO] 固定使用 torch.float16 作为模型精度。")

    # 注意：对于 Qwen2.5-VL-7B-Instruct，这里使用 Qwen2_5_VLForConditionalGeneration（若可用），
    # 而不是 AutoModelForCausalLM，否则会触发 Qwen2_5_VLConfig 不被支持的报错。
    # 设备分配策略：
    #   - 若当前进程可见 CUDA，则将整个模型绑定到单块 GPU 上（cuda:0）；
    #   - 否则退回到 CPU。
    device_map = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    print(f"[INFO] 使用 device_map={device_map} 加载 Qwen2.5-VL 模型。")

    model = QwenVLModel.from_pretrained(
        cfg.MODEL_PATH,
        torch_dtype=model_dtype,  # 按你的要求，这里使用 torch_dtype 参数并固定为 float16
        device_map=device_map,
        trust_remote_code=True,  # Qwen2 通常需要
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 启用 gradient checkpointing，以节省显存
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # 关闭缓存以配合 checkpointing

    # 冻结 LLM 所有原始参数
    for param in model.parameters():
        param.requires_grad = False

    print("[INFO] LLM 参数已全部冻结，仅 MoICV 层会被训练。")

    # -----------------------
    # 2. 构建 MoICV 层并加载初始化专家
    # -----------------------
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError("model.config 中未找到 hidden_size 字段，请检查模型配置。")

    print(f"[INFO] 模型 hidden_size = {hidden_size}")

    # 我们使用 Embedding 的维度作为 query_dim（与 hidden_size 相同）
    query_dim = hidden_size

    moicv_layer = Dual_MoICV_Layer(
        query_dim=query_dim,
        attn_dim=hidden_size,
        ffn_dim=hidden_size,
    )

    # 为了提高 MoICV 参数与梯度的数值稳定性，我们将 MoICV 层**统一使用 float32 精度**，
    # 即便主干 LLM 仍然以 float16 运行。
    # 这样可以显著降低 Top-2 路由 / 正交正则带来的梯度爆炸风险。
    moicv_layer.to(device=main_device, dtype=torch.float32)

    # 从 896 维初始化权重文件中加载专家向量，并使用插值拉伸到 hidden_size
    load_and_assign_experts(
        moicv_layer=moicv_layer,
        expert_init_path=cfg.EXPERT_INIT_PATH,
        hidden_size=hidden_size,
    )

    # -----------------------
    # 3. 构建带注入能力的 Wrapper
    # -----------------------
    wrapper = MoICV_Qwen_Wrapper(
        llm_model=model,
        moicv_layer=moicv_layer,
        attn_inject_layer_idx=cfg.ATTN_INJECT_LAYER_IDX,
        ffn_inject_layer_idx=cfg.FFN_INJECT_LAYER_IDX,
    )

    # MoICV Loss，用于正交解耦 + 负载均衡
    moicv_loss_fn = MoICV_Loss(alpha=cfg.MOICV_ALPHA, beta=cfg.MOICV_BETA)

    # 只优化 MoICV 层的参数
    optimizer = torch.optim.AdamW(
        moicv_layer.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # -----------------------
    # 4. 准备数据集与 DataLoader
    # -----------------------
    print(f"[INFO] 正在加载训练数据集：{cfg.DATASET_PATH}")
    hf_dataset = load_from_disk(cfg.DATASET_PATH)
    train_dataset = MixedQADataset(hf_dataset)

    collate_fn = build_collate_fn(tokenizer, max_length=cfg.MAX_SEQ_LEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print(
        f"[INFO] 训练集样本数：{len(train_dataset)}，"
        f"batch_size={cfg.BATCH_SIZE}，num_batches={len(train_loader)}"
    )

    # -----------------------
    # 5. 训练 Loop
    # -----------------------
    wrapper.train()
    moicv_layer.train()

    for epoch in range(cfg.NUM_EPOCHS):
        # 每个 Epoch 开始时打印当前显存占用情况，方便监控 48G 显存利用率
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(
                f"[INFO][Epoch {epoch+1}] CUDA 显存占用："
                f"allocated={mem_allocated:.2f}GB, reserved={mem_reserved:.2f}GB"
            )

        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}", unit="batch")

        for step, batch in enumerate(epoch_bar):
            # 将输入显式移动到主设备（cuda:0），确保不会意外留在 CPU
            input_ids = batch["input_ids"].to(main_device)
            attention_mask = batch["attention_mask"].to(main_device)
            labels = batch["labels"].to(main_device)

            # 1) 计算 query_features：
            #    这里我们使用词嵌入的加权平均作为 MoICV 的查询特征：
            #        query_features = mean(Embedding(input_ids), dim=seq_len)
            #    其中 attention_mask 用于忽略 padding token。
            with torch.no_grad():
                # 获取输入嵌入：[B, S, H]
                embed_layer = wrapper.llm_model.get_input_embeddings()
                token_embeds = embed_layer(input_ids)  # 输出自动位于模型所在设备和 dtype（FP16）

                # 使用 attention_mask 计算加权平均（避免 padding 影响）
                # attention_mask: [B, S] -> [B, S, 1]
                mask = attention_mask.unsqueeze(-1)  # [B, S, 1]
                masked_embeds = token_embeds * mask  # [B, S, H]

                # 每个样本的有效 token 数
                lengths = mask.sum(dim=1).clamp(min=1)  # [B, 1]

                # 求平均：sum / length -> [B, H]
                query_features = masked_embeds.sum(dim=1) / lengths

                # Debug 探针：检查第一步的 query_features 是否出现全 0 / 极小值
                if epoch == 0 and step == 0:
                    try:
                        q_max = query_features.max().detach().cpu().item()
                        q_min = query_features.min().detach().cpu().item()
                        print(f"[Debug] query_features max: {q_max:.6e}, min: {q_min:.6e}")
                    except Exception:
                        pass

                # 将 Router 的查询特征提升到 float32 精度，以匹配 MoICV 层的计算精度，
                # 避免在 float16 下进行复杂的路由 / 正交运算导致数值溢出。
                query_features = query_features.to(dtype=torch.float32)

            # 2) 调用 wrapper.forward_with_moicv 完成：
            #    - MoICV: 生成 v_attn / v_ffn 并通过 Hook 注入指定层
            #    - LLM:   正常前向，返回 CE Loss
            outputs = wrapper.forward_with_moicv(
                query_features=query_features,
                compute_moicv_loss=True,
                moicv_loss_fn=moicv_loss_fn,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            llm_outputs = outputs["llm_outputs"]
            moicv_loss_val = outputs["moicv_loss"]

            # 3) 从 LLM 输出中获取交叉熵 Loss（CE Loss）
            if hasattr(llm_outputs, "loss") and llm_outputs.loss is not None:
                ce_loss = llm_outputs.loss
            elif isinstance(llm_outputs, dict) and "loss" in llm_outputs:
                ce_loss = llm_outputs["loss"]
            else:
                # 若模型未自动返回 loss，可手动计算：
                logits = llm_outputs["logits"] if isinstance(llm_outputs, dict) else llm_outputs.logits
                # shift logits 与 labels，典型 causal LM loss 计算方式
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=tokenizer.pad_token_id,
                )

            # moicv_loss_val 已经等于 alpha * loss_balance + beta * loss_ortho
            total_loss = ce_loss + moicv_loss_val

            optimizer.zero_grad()
            total_loss.backward()

            # NaN 探测器：一旦发现 loss 为 NaN，则跳过本次更新，避免污染参数。
            if torch.isnan(total_loss):
                print("[致命错误] 发现 NaN Loss，跳过本次 optimizer.step()！")
                optimizer.zero_grad()
                continue

            # 梯度裁剪：对 MoICV 层的梯度进行范数裁剪，防止梯度爆炸。
            torch.nn.utils.clip_grad_norm_(moicv_layer.parameters(), max_norm=1.0)

            optimizer.step()

            # 打印时提高 moicv_loss 的小数精度，避免真实值在 1e-4 量级时被四舍五入成 0.0000
            epoch_bar.set_postfix(
                {
                    "ce_loss": f"{ce_loss.item():.4f}",
                    "moicv_loss": f"{moicv_loss_val.item():.6f}",
                    "total_loss": f"{total_loss.item():.4f}",
                }
            )

        # 每个 epoch 结束后，保存一份 MoICV 权重快照，方便你分别评测 1/2/3 个 epoch 的效果。
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        epoch_ckpt = os.path.join(cfg.OUTPUT_DIR, f"moicv_epoch{epoch+1}_gpu.bin")
        torch.save(moicv_layer.state_dict(), epoch_ckpt)
        print(f"[INFO] Epoch {epoch+1} 结束，已保存 MoICV 权重到：{os.path.abspath(epoch_ckpt)}")

    # -----------------------
    # 6. 再额外保存一份“最终” MoICV 层参数
    # -----------------------
    final_path = os.path.join(cfg.OUTPUT_DIR, "moicv_trained_weights_gpu.bin")
    torch.save(moicv_layer.state_dict(), final_path)
    print(f"[INFO] 训练完成，最终 MoICV 参数已保存到（GPU 版本）：{os.path.abspath(final_path)}")


if __name__ == "__main__":
    main()


