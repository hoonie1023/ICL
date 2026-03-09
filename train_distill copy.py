#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV + Qwen2.5-VL-7B 自蒸馏训练脚本 (Contextual Self-Distillation)
====================================================================

核心思想：
---------
- Teacher：原始 Qwen2.5-VL-7B 模型（不注入 MoICV），使用 16-shot 多模态 Demonstrations + Target Query；
- Student：同一个 Qwen2.5-VL-7B Backbone，但注入 MoICV（Dual_MoICV_Layer），仅使用 0-shot Target Query；
- 蒸馏目标：让带 MoICV 的 Student 在 Target Query 的答案位置上，拟合 Teacher 的概率分布；
- 不再使用传统监督 CE Loss，仅使用 KL 自蒸馏 Loss + MoICV 正则 Loss（负载均衡 + 正交）。

显存与精度控制：
---------------
- 强制使用单卡 GPU：cuda:0（若不可用则退回 CPU）；
- Qwen2.5-VL Backbone: FP16 (torch.float16)；
- MoICV 层：FP32 (torch.float32) 以提高数值稳定性；
- 批大小 BATCH_SIZE = 1，使用 GRAD_ACCUM_STEPS 做梯度累加，避免 16-shot 爆显存。

依赖安装（示例）：
    pip install torch torchvision
    pip install transformers datasets
    pip install tqdm
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_from_disk
from tqdm.auto import tqdm

from transformers import AutoProcessor, set_seed

# Qwen2.5-VL 模型类：优先使用官方 VL 类，若版本过旧则退回 AutoModelForCausalLM
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    from transformers import AutoModelForCausalLM as QwenVLModel

from moicv_core import Dual_MoICV_Layer, MoICV_Loss
from moicv_injection import MoICV_Qwen_Wrapper


# ===========================
# 离线 / 环境配置
# ===========================

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# 为了避免 OpenMP 相关的多线程冲突，在多数服务器环境下推荐限制线程数
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")


# ===========================
# 全局训练配置
# ===========================

@dataclass
class DistillConfig:
    # 本地 Qwen2.5-VL-7B 模型路径
    MODEL_PATH: str = r"/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct"

    # 训练数据集（与之前相同）
    DATASET_PATH: str = "./mini_mixed_dataset"

    # 专家初始化权重
    EXPERT_INIT_PATH: str = "./expert_init_tensors.pt"

    # 训练产物输出目录
    OUTPUT_DIR: str = "./outputs_distill_gpu"

    # 训练相关
    BATCH_SIZE: int = 1  # 自蒸馏版本强制使用 batch_size=1
    NUM_EPOCHS: int = 1
    MAX_SEQ_LEN: int = 512

    # 自蒸馏相关
    NUM_SHOTS: int = 16          # 每个 Target Query 的 Demonstrations 数量
    TEMPERATURE: float = 1.0     # 蒸馏温度

    # MoICV 正则 Loss 权重
    MOICV_ALPHA: float = 0.1
    MOICV_BETA: float = 0.1

    # 学习率与梯度累加
    LEARNING_RATE: float = 1e-5
    WEIGHT_DECAY: float = 0.0
    GRAD_ACCUM_STEPS: int = 8    # 梯度累加步数

    # 注入层索引
    ATTN_INJECT_LAYER_IDX: int = 0
    FFN_INJECT_LAYER_IDX: int = 15

    # 随机种子
    SEED: int = 42


CFG = DistillConfig()


def get_main_device() -> torch.device:
    """
    获取主设备。

    优先使用单块 GPU：cuda:0。
    如果当前进程里确实没有 CUDA，就自动退回 CPU，并打印警告。
    """
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    print(f"[DEBUG][Distill] torch.cuda.is_available()={cuda_available}, device_count={cuda_count}")

    if cuda_available and cuda_count > 0:
        return torch.device("cuda:0")

    print(
        "[WARN][Distill] 当前进程内未检测到可用的 CUDA GPU，训练将退回到 CPU。\n"
        "如果你确认有 GPU，请检查：Python 环境 / 容器 / 驱动 / CUDA 版本是否与本脚本一致。"
    )
    return torch.device("cpu")


def build_qa_text(query: str, answer: str | None, with_vision_prefix: bool = True) -> str:
    """
    构造单条图文问答的文本模板。

    若 with_vision_prefix=True，则在最前面添加视觉占位符：
        <|vision_start|><|image_pad|><|vision_end|>
    然后接：
        Q: {query}\nA: {answer or "" }
    """
    lines: List[str] = []
    if with_vision_prefix:
        lines.append("<|vision_start|><|image_pad|><|vision_end|>")
    if answer is None:
        lines.append(f"Q: {query}\nA:")
    else:
        lines.append(f"Q: {query}\nA: {answer}")
    return "\n".join(lines)


def get_last_token_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    从模型输出 logits 中提取每个样本最后一个非 pad token 位置的 logits。

    logits: [B, S, V]
    attention_mask: [B, S]
    返回: [B, V]
    """
    B, S, V = logits.shape
    device = logits.device
    lengths = attention_mask.sum(dim=1)  # [B]
    last_indices = (lengths - 1).clamp(min=0)  # [B]
    batch_indices = torch.arange(B, device=device)
    final_logits = logits[batch_indices, last_indices]  # [B, V]
    return final_logits


def sample_demonstrations(
    dataset,
    target_idx: int,
    num_shots: int,
    rng: random.Random,
) -> List[int]:
    """
    从整个数据集中随机无放回采样 num_shots 个 Demonstrations 索引（排除 target_idx）。
    返回采样到的索引列表。
    """
    n = len(dataset)
    candidates = list(range(n))
    candidates.remove(target_idx)
    if num_shots >= len(candidates):
        return candidates
    return rng.sample(candidates, num_shots)


def main() -> None:
    cfg = CFG
    set_seed(cfg.SEED)
    rng = random.Random(cfg.SEED)

    device = get_main_device()
    print(f"[INFO][Distill] 主设备：{device}")

    # 1. 加载 Qwen2.5-VL Backbone (Teacher / Student 共用)
    model_dtype = torch.float16
    device_map = "cuda:0" if device.type == "cuda" else "cpu"
    print(f"[INFO][Distill] 正在加载 Qwen2.5-VL 模型：{cfg.MODEL_PATH}，device_map={device_map}")

    model = QwenVLModel.from_pretrained(
        cfg.MODEL_PATH,
        torch_dtype=model_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        cfg.MODEL_PATH,
        trust_remote_code=True,
    )
    model.eval()  # Teacher 使用 no_grad；Student 的梯度只在 MoICV 层

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError("model.config 中未找到 hidden_size 字段，请检查模型配置。")
    print(f"[INFO][Distill] 模型 hidden_size = {hidden_size}")

    # 2. 构建 MoICV 层（Student 专用）并初始化专家
    moicv_layer = Dual_MoICV_Layer(
        query_dim=hidden_size,
        attn_dim=hidden_size,
        ffn_dim=hidden_size,
    )
    # Student 的 MoICV 层使用 float32 精度，提高数值稳定性
    moicv_layer.to(device=device, dtype=torch.float32)

    # 从 896 维初始化权重文件中加载专家向量，并使用插值拉伸到 hidden_size
    from train import load_and_assign_experts  # 复用已有函数

    load_and_assign_experts(
        moicv_layer=moicv_layer,
        expert_init_path=cfg.EXPERT_INIT_PATH,
        hidden_size=hidden_size,
    )

    # 构建带注入能力的 Wrapper（Student）
    wrapper = MoICV_Qwen_Wrapper(
        llm_model=model,
        moicv_layer=moicv_layer,
        attn_inject_layer_idx=cfg.ATTN_INJECT_LAYER_IDX,
        ffn_inject_layer_idx=cfg.FFN_INJECT_LAYER_IDX,
    )

    # MoICV 正则 Loss
    moicv_loss_fn = MoICV_Loss(alpha=cfg.MOICV_ALPHA, beta=cfg.MOICV_BETA)

    # 只优化 MoICV 层参数
    optimizer = torch.optim.AdamW(
        moicv_layer.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # 3. 加载训练数据集（Teacher / Student 共用）
    print(f"[INFO][Distill] 正在加载训练数据集：{cfg.DATASET_PATH}")
    hf_dataset = load_from_disk(cfg.DATASET_PATH)
    n_samples = len(hf_dataset)
    indices = list(range(n_samples))
    print(f"[INFO][Distill] 训练样本数：{n_samples}")

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(cfg.NUM_EPOCHS):
        rng.shuffle(indices)
        epoch_bar = tqdm(indices, desc=f"[Distill] Epoch {epoch+1}/{cfg.NUM_EPOCHS}", unit="sample")

        for step_idx, idx in enumerate(epoch_bar):
            ex_target = hf_dataset[idx]

            # ========== 3.1 构造 Teacher 16-shot 输入 ==========
            # 采样 Demonstrations
            demo_indices = sample_demonstrations(hf_dataset, idx, cfg.NUM_SHOTS, rng)

            teacher_texts: List[str] = []
            teacher_images: List[Any] = []

            # 16 条 Demonstrations：图文 + 完整答案
            for did in demo_indices:
                ex_demo = hf_dataset[did]
                demo_text = build_qa_text(ex_demo["query"], ex_demo["label"], with_vision_prefix=True)
                teacher_texts.append(demo_text)
                teacher_images.append(ex_demo["image"])

            # 最后一条是 Target Query：图文 + 不含答案
            target_teacher_text = build_qa_text(ex_target["query"], None, with_vision_prefix=True)
            teacher_texts.append(target_teacher_text)
            teacher_images.append(ex_target["image"])

            # Teacher 前向（无梯度）
            with torch.no_grad():
                teacher_inputs = processor(
                    text=teacher_texts,
                    images=teacher_images,
                    return_tensors="pt",
                    padding=True,
                )
                teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

                teacher_outputs = model(**teacher_inputs)
                teacher_logits_all = teacher_outputs.logits  # [B,D,V]
                teacher_attn_mask = teacher_inputs["attention_mask"]

                # 取每个样本最后一个非 pad token 的 logits，这里我们只需要 batch 中最后一条（Target）
                final_logits = get_last_token_logits(teacher_logits_all, teacher_attn_mask)  # [B,V]
                teacher_logits = final_logits[-1:].contiguous()  # [1, V]

                # 温度平滑 + softmax -> 概率分布
                T = cfg.TEMPERATURE
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)  # [1,V]

            # 释放 Teacher 中间张量，缓解显存压力
            del teacher_outputs, teacher_logits_all, teacher_inputs, final_logits
            torch.cuda.empty_cache() if device.type == "cuda" else None

            # ========== 3.2 构造 Student 0-shot 输入（与 train.py / eval 融合的方式一致） ==========
            student_text = build_qa_text(ex_target["query"], None, with_vision_prefix=True)
            student_image = ex_target["image"]

            student_inputs = processor(
                text=[student_text],
                images=[student_image],
                return_tensors="pt",
                padding=True,
            )
            student_inputs = {k: v.to(device) for k, v in student_inputs.items()}

            input_ids = student_inputs["input_ids"]
            attention_mask = student_inputs["attention_mask"]
            pixel_values = student_inputs.get("pixel_values", None)
            image_grid_thw = student_inputs.get("image_grid_thw", None)

            # 计算 query_features（与 train.py 完全对齐）
            with torch.no_grad():
                embed_layer = wrapper.llm_model.get_input_embeddings()
                token_embeds = embed_layer(input_ids)          # [1,S,H]
                mask = attention_mask.unsqueeze(-1)            # [1,S,1]
                masked_embeds = token_embeds * mask            # [1,S,H]
                lengths = mask.sum(dim=1).clamp(min=1)         # [1,1]
                query_features = masked_embeds.sum(dim=1) / lengths  # [1,H]
                query_features = query_features.to(dtype=torch.float32)

            # 组装传入 Wrapper 的关键字参数
            wrapper_kwargs = dict(
                query_features=query_features,
                compute_moicv_loss=True,
                moicv_loss_fn=moicv_loss_fn,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if pixel_values is not None:
                wrapper_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                wrapper_kwargs["image_grid_thw"] = image_grid_thw

            # Student 前向（有梯度，仅在 MoICV 层）
            outputs = wrapper.forward_with_moicv(**wrapper_kwargs)
            llm_outputs = outputs["llm_outputs"]
            moicv_reg_loss = outputs["moicv_loss"]  # alpha * balance + beta * ortho

            # Student logits -> 取最后一个非 pad token 的 logits
            student_logits_all = llm_outputs["logits"] if isinstance(llm_outputs, dict) else llm_outputs.logits
            student_final_logits = get_last_token_logits(student_logits_all, attention_mask)  # [1,V]

            T = cfg.TEMPERATURE
            student_logprobs = F.log_softmax(student_final_logits / T, dim=-1)

            # 蒸馏 Loss：KL(student || teacher)，reduction = batchmean
            distill_loss = F.kl_div(
                student_logprobs,
                teacher_probs,
                reduction="batchmean",
            )

            # 总 Loss：自蒸馏 + MoICV 正则
            total_loss = distill_loss + moicv_reg_loss

            # 梯度累加：先将 Loss 归一化
            total_loss = total_loss / cfg.GRAD_ACCUM_STEPS

            # 反向传播前的 NaN 检测
            if torch.isnan(total_loss):
                print("[致命错误][Distill] 发现 NaN Loss，跳过本次样本！")
                optimizer.zero_grad()
                continue

            total_loss.backward()

            # 每 GRAD_ACCUM_STEPS 个样本执行一次优化步骤
            if (step_idx + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                # 梯度裁剪：防止 MoICV 层梯度爆炸
                torch.nn.utils.clip_grad_norm_(moicv_layer.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # 更新进度条
            epoch_bar.set_postfix(
                {
                    "distill_loss": f"{distill_loss.detach().cpu().item():.6f}",
                    "moicv_loss": f"{moicv_reg_loss.detach().cpu().item():.6f}",
                    "total_loss": f"{total_loss.detach().cpu().item():.6f}",
                    "g_step": global_step,
                }
            )

        # 每个 epoch 结束后，保存一份 MoICV 权重快照
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        epoch_ckpt = os.path.join(cfg.OUTPUT_DIR, f"moicv_distill_epoch{epoch+1}_gpu.bin")
        torch.save(moicv_layer.state_dict(), epoch_ckpt)
        print(f"[INFO][Distill] Epoch {epoch+1} 结束，已保存 MoICV Distill 权重到：{os.path.abspath(epoch_ckpt)}")

    # 训练结束后再额外保存一份“最终”权重
    final_path = os.path.join(cfg.OUTPUT_DIR, "moicv_distill_trained_gpu.bin")
    torch.save(moicv_layer.state_dict(), final_path)
    print(f"[INFO][Distill] 训练完成，自蒸馏版 MoICV 权重已保存到：{os.path.abspath(final_path)}")


if __name__ == "__main__":
    main()


