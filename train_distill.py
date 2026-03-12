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
from datasets import load_from_disk, Dataset
from PIL import Image as PILImage
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

    # 预打包的 M²IV 蒸馏数据集路径（由 build_m2iv_dataset.py 生成）
    DATASET_PATH: str = "/root/autodl-tmp/mmlm-icl/m2iv_distill_dataset"

    # 原始 A-OKVQA / CSQA 训练集 parquet 路径（用于根据索引还原图文内容）
    AOKVQA_TRAIN_PARQUET: str = "/root/autodl-tmp/mmlm-icl/train-00000-of-00002-c1d24de3bacb5e0c.parquet"
    CSQA_TRAIN_PARQUET: str = "/root/autodl-tmp/mmlm-icl/train-00000-of-00001.parquet"

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

    # 损失权重组
    # M²IV 基础权重组
    LAMBDA_MIM: float = 1.0      # 模仿损失 (自蒸馏 KL) 权重
    LAMBDA_SYN: float = 1.0      # 动态协同损失 (Synergistic Loss) 权重
    LAMBDA_SUP: float = 0.5      # 监督损失 (CE) 权重

    # MoE / MoICV 路由约束组（负载均衡 + 正交）
    MOICV_ALPHA: float = 2.0     # 负载均衡权重
    MOICV_BETA: float = 0.5      # 正交约束权重

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


def decode_image_field(image_field: Any) -> PILImage.Image:
    """
    将原始 parquet 中的 image 字段转换为 PIL.Image。
    支持：
      - dict(bytes=...)
      - 原始 bytes / bytearray
    解析失败时返回黑色占位图。
    """
    try:
        if isinstance(image_field, dict) and "bytes" in image_field:
            return PILImage.open(BytesIO(image_field["bytes"])).convert("RGB")
        if isinstance(image_field, (bytes, bytearray)):
            return PILImage.open(BytesIO(image_field)).convert("RGB")
    except Exception:
        pass
    return PILImage.new("RGB", (224, 224), color="black")


def extract_qa_from_raw(ex: Dict[str, Any], source: str) -> Tuple[str, str]:
    """
    根据原始 A-OKVQA / CSQA 样本结构，抽取 question 文本与 answer 文本。
    由于不同数据集 schema 不同，这里做了尽量合理的兼容：
      - A-OKVQA:
          question: ex["question"]
          answer:   若存在 choices + correct_choice_idx，则取对应选项文本；
                   否则若存在 direct_answers，则取第一个；
                   否则使用 "unknown"
      - CSQA:
          question: ex["question"]
          answer:   若存在 choices{"label","text"} + answerKey，则映射到对应 text；
                   否则若存在 "answer" 字段，则直接使用；
                   否则使用 "unknown"
    """
    q = ex.get("question", "")
    a = "unknown"

    if source == "aokvqa":
        choices = ex.get("choices", [])
        correct_idx = ex.get("correct_choice_idx", None)
        if isinstance(choices, list) and correct_idx is not None:
            try:
                ci = int(correct_idx)
                if 0 <= ci < len(choices):
                    a = str(choices[ci])
            except Exception:
                pass
        if a == "unknown":
            direct_answers = ex.get("direct_answers", None)
            if isinstance(direct_answers, list) and len(direct_answers) > 0:
                a = str(direct_answers[0])
    else:  # csqa
        raw_choices = ex.get("choices", {})
        answer_key = ex.get("answerKey", None)
        if isinstance(raw_choices, dict) and "label" in raw_choices and "text" in raw_choices and answer_key is not None:
            labels = list(raw_choices.get("label", []))
            texts = list(raw_choices.get("text", []))
            try:
                idx = labels.index(answer_key)
                if 0 <= idx < len(texts):
                    a = str(texts[idx])
            except ValueError:
                pass
        if a == "unknown" and "answer" in ex:
            a = str(ex["answer"])

    return q, a


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

    # 只优化 MoICV 层参数 + 注入强度因子 gamma_attn / gamma_ffn
    optimizer = torch.optim.AdamW(
        list(moicv_layer.parameters()) + [wrapper.gamma_attn, wrapper.gamma_ffn],
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # 3. 加载训练数据集（Teacher / Student 共用）
    print(f"[INFO][Distill] 正在加载预打包 M²IV 数据集：{cfg.DATASET_PATH}")
    hf_dataset = load_from_disk(cfg.DATASET_PATH)
    n_samples = len(hf_dataset)
    indices = list(range(n_samples))
    print(f"[INFO][Distill] M²IV 蒸馏样本数：{n_samples}")

    # 同时加载原始 A-OKVQA / CSQA 训练集，用于根据索引还原 target / demos 的图文内容
    print(f"[INFO][Distill] 从 parquet 加载原始 A-OKVQA 训练集：{cfg.AOKVQA_TRAIN_PARQUET}")
    ds_aokvqa_raw = Dataset.from_parquet(cfg.AOKVQA_TRAIN_PARQUET)
    print(f"[INFO][Distill] A-OKVQA 原始样本数：{len(ds_aokvqa_raw)}")

    print(f"[INFO][Distill] 从 parquet 加载原始 CSQA 训练集：{cfg.CSQA_TRAIN_PARQUET}")
    ds_csqa_raw = Dataset.from_parquet(cfg.CSQA_TRAIN_PARQUET)
    print(f"[INFO][Distill] CSQA 原始样本数：{len(ds_csqa_raw)}")

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(cfg.NUM_EPOCHS):
        rng.shuffle(indices)
        epoch_bar = tqdm(indices, desc=f"[Distill] Epoch {epoch+1}/{cfg.NUM_EPOCHS}", unit="sample")

        for step_idx, idx in enumerate(epoch_bar):
            ex_m2iv = hf_dataset[idx]
            source = ex_m2iv["source"]
            target_index = int(ex_m2iv["target_index"])
            demo_indices = [int(i) for i in ex_m2iv["demo_indices"]]

            # 选择对应的原始数据集
            if source == "aokvqa":
                ds_raw = ds_aokvqa_raw
            else:
                ds_raw = ds_csqa_raw

            # 恢复 Target 与 Demos 的原始样本
            target_raw = ds_raw[target_index]
            demos_raw = [ds_raw[i] for i in demo_indices]

            # ========== 3.1 构造 Teacher 16-shot 输入（直接使用预打包 demos） ==========
            teacher_texts: List[str] = []
            teacher_images: List[Any] = []

            # 16 条 Demonstrations：图文 + 完整答案
            for ex_demo in demos_raw:
                q_demo, a_demo = extract_qa_from_raw(ex_demo, source=source)
                demo_text = build_qa_text(q_demo, a_demo, with_vision_prefix=True)
                teacher_texts.append(demo_text)
                teacher_images.append(decode_image_field(ex_demo.get("image", None)))

            # 最后一条是 Target Query：图文 + 完整真实答案（用于序列级蒸馏）
            q_target, a_target = extract_qa_from_raw(target_raw, source=source)
            target_teacher_text = build_qa_text(q_target, a_target, with_vision_prefix=True)
            teacher_texts.append(target_teacher_text)
            teacher_images.append(decode_image_field(target_raw.get("image", None)))

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
                teacher_logits_all = teacher_outputs.logits  # [B,S,V]

                # 仅保留 batch 中最后一条（Target）的完整序列 logits
                teacher_logits_seq = teacher_logits_all[-1:, :, :]  # [1,S,V]
                # 提前提取出 target 的 mask，防止后续被 del 掉
                teacher_target_mask = teacher_inputs["attention_mask"][-1:].clone()

            # 释放 Teacher 中间张量，缓解显存压力
            del teacher_outputs, teacher_logits_all, teacher_inputs
            torch.cuda.empty_cache() if device.type == "cuda" else None

            # ========== 3.2 构造 Student 0-shot 输入（仅使用 target 图文 + 真实答案） ==========
            student_text = build_qa_text(q_target, a_target, with_vision_prefix=True)
            student_image = decode_image_field(target_raw.get("image", None))

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
            syn_loss = outputs.get("syn_loss", torch.tensor(0.0, device=device))

            student_logits_all = llm_outputs["logits"] if isinstance(llm_outputs, dict) else llm_outputs.logits
            student_logits_seq = student_logits_all  # [1,S,V]

            # ======== 1. 精准构造 Loss Mask (避开多模态 Tokenizer 陷阱) ========
            # 核心思路：分别过一次 processor 拿 prompt 的实际 token 长度，后面的自然就是答案
            prompt_text = build_qa_text(q_target, None, with_vision_prefix=True)
            with torch.no_grad():
                prompt_inputs = processor(
                    text=[prompt_text],
                    images=[student_image],
                    return_tensors="pt",
                )
            prompt_len = prompt_inputs["input_ids"].size(1)
            seq_len = input_ids.size(1)

            loss_mask = torch.zeros((1, seq_len), device=device, dtype=torch.float32)
            if seq_len > prompt_len:
                loss_mask[0, prompt_len:] = 1.0  # 答案部分标记为 1

            # 乘以 attention_mask 排除 pad 的影响
            loss_mask = loss_mask * attention_mask.float()  # [1, S]

            # ======== 2. 截取 Teacher Logits 消除 Batch Padding 差异 ========
            valid_len = int(teacher_target_mask.sum().item())            # 目标样本的无 pad 真实长度

            # 根据 Qwen 的 padding_side 取出有效 logits (通常 Qwen 是 left padding，但做一下兼容防御)
            if processor.tokenizer.padding_side == "left":
                clean_teacher_logits = teacher_logits_seq[:, -valid_len:, :]
            else:
                clean_teacher_logits = teacher_logits_seq[:, :valid_len, :]

            # 此时 clean_teacher_logits 的长度应该严格等于 student_logits_seq 的非 pad 长度
            # 为防止多模态 patch 导致的极微小舍入误差，做一次严格对齐防御
            min_len = min(clean_teacher_logits.size(1), student_logits_seq.size(1))
            clean_teacher_logits = clean_teacher_logits[:, :min_len, :]
            clean_student_logits = student_logits_seq[:, :min_len, :]
            clean_loss_mask = loss_mask[:, :min_len]

            # ======== 3. 序列级模仿损失：含 Shift 对齐 ========
            T = cfg.TEMPERATURE
            # Shift 操作：用 i 位置的输出对齐 i+1 位置的标签/mask
            shift_teacher_probs = F.softmax(clean_teacher_logits[:, :-1, :] / T, dim=-1)
            shift_student_logprobs = F.log_softmax(clean_student_logits[:, :-1, :] / T, dim=-1)
            shift_loss_mask = clean_loss_mask[:, 1:].contiguous()

            # 计算逐 token 的 KL 散度
            kl_per_token = F.kl_div(
                shift_student_logprobs,
                shift_teacher_probs,
                reduction="none",
            ).sum(dim=-1)  # [1, S-1]

            # 应用 Mask 并求均值
            masked_kl = kl_per_token * shift_loss_mask
            denom = shift_loss_mask.sum().clamp(min=1.0)
            distill_loss = masked_kl.sum() / denom

            # ======== 监督损失 (Supervised CE Loss) ========
            # 对 Student 的完整 logits 与 input_ids 做 causal LM 的 shift 计算交叉熵。
            shift_logits = student_logits_all[..., :-1, :].contiguous()   # [1,S-1,V]
            shift_labels = input_ids[..., 1:].contiguous()                # [1,S-1]

            # 计算 CE Loss 时忽略 padding token，对应 pad_token_id 由 processor.tokenizer 提供
            pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                # 若未定义 pad_token_id，则退回使用 -100 作为 ignore_index
                pad_token_id = -100

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=pad_token_id,
            )

            # ======== 融合多重损失 (Loss Integration) ========
            # 1. 模仿损失: distill_loss (self-distillation)
            # 2. 监督损失: ce_loss (supervised CE)
            # 3. 协同损失: syn_loss (M²IV Synergistic Loss)
            # 4. MoICV 路由正则: moicv_reg_loss (balance + ortho)
            total_loss = (
                cfg.LAMBDA_MIM * distill_loss
                + cfg.LAMBDA_SUP * ce_loss
                + cfg.LAMBDA_SYN * syn_loss
                + moicv_reg_loss
            )

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
                    "ce_loss": f"{ce_loss.detach().cpu().item():.6f}",
                    "syn_loss": f"{syn_loss.detach().cpu().item():.6f}",
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


