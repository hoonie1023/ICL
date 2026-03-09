#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估脚本：Baseline Qwen2.5-7B vs 注入 MoICV (Ours)
===================================================

快速验证版评估流水线，支持：
1. 加载 A-OKVQA & CSQA 的测试 split（前 500 条样本）
2. 串行对比：
   - Baseline：原始 Qwen2.5-7B
   - Ours：注入 MoICV 后的 Qwen2.5-7B
3. 使用统一 Prompt 做零样本多项选择推理：
   "Question: {q}\\nChoices:\\n(A) ...\\n(B) ...\\n...\\nAnswer:"
4. 通过 logits probe：
   - 取 "Answer:" 后第一个 token 的 logits
   - 在 'A','B','C','D' 四个候选上做 softmax，取 argmax 作为预测
   - 计算每个样本的 4 类分布熵（可扩展用于不确定性分析）
5. 在 MoICV 模式下统计路由分布（视觉/文本专家的激活比例）
6. 输出对比表 & 保存雷达图数据到 radar_plot_data.json

依赖示例：
    pip install torch torchvision
    pip install transformers datasets
    pip install pillow
    pip install tqdm

注意：
    - 本脚本假定你已经完成：
        * mini_mixed_dataset + test parquet 的准备
        * expert_init_tensors.pt 的生成（896 维）
        * moicv_trained_weights.bin 的训练（已对齐 hidden_size）
    - 为方便在 48G 显存服务器上运行，本脚本使用 BF16（若不支持则回退 FP16），并启用完全离线模式。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    set_seed,
)

# 与训练脚本一致：优先使用 Qwen2.5-VL 的专用类，若 transformers 版本较老则回退到 AutoModelForCausalLM
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    from transformers import AutoModelForCausalLM as QwenVLModel

from moicv_core import Dual_MoICV_Layer, MoICV_Loss
from moicv_injection import MoICV_Qwen_Wrapper


# ===========================
# 离线 / 环境配置
# ===========================

"""离线 & 高配显存优化：
- 启用 Hugging Face Hub 和 Datasets 的离线模式，防止任何网络访问；
- 默认使用 FP16，并强制绑定到单块 GPU（cuda:0），与训练脚本的高配版本保持一致。
"""

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@dataclass
class EvalConfig:
    # 本地 Qwen2.5-VL-7B 模型路径（与训练脚本一致，需按实际情况修改）
    MODEL_PATH: str = r"/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct"

    # 测试集 parquet 路径
    AOKVQA_TEST_PARQUET: str = r"/root/autodl-tmp/mmlm-icl/test-00000-of-00001-d306bf3ad53b6618.parquet"
    CSQA_TEST_PARQUET: str = r"/root/autodl-tmp/mmlm-icl/train-00000-of-00001.parquet"

    # 每个任务评估的样本数；若 <= 0，则使用整个测试集（即“全部评测”）
    NUM_SAMPLES_PER_TASK: int = -1

    # MoICV & 初始化权重
    EXPERT_INIT_PATH: str = "./expert_init_tensors.pt"
    # 使用 GPU 训练得到的 MoICV 权重，这里默认评测第 1 个 epoch 的结果
    MOICV_WEIGHTS_PATH: str = "/root/autodl-tmp/mmlm-icl/outputs_gpu/moicv_epoch1_gpu.bin"

    # 注入层索引（与训练时保持一致）
    ATTN_INJECT_LAYER_IDX: int = 0
    FFN_INJECT_LAYER_IDX: int = 15

    # 推理相关
    MAX_SEQ_LEN: int = 512
    BATCH_SIZE: int = 4  # 评估 batch size（可适当增大）
    SEED: int = 42


CFG = EvalConfig()


def get_main_device() -> torch.device:
    """
    获取主设备。

    为了与训练脚本在 vGPU 环境下的行为保持一致，这里**暴力锁定到 cuda:0**，
    如果当前进程内检测不到 CUDA，则退回到 CPU 并打印警告。
    """
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    print(f"[DEBUG][Eval] torch.cuda.is_available()={cuda_available}, device_count={cuda_count}")

    if cuda_available and cuda_count > 0:
        return torch.device("cuda:0")

    print(
        "[WARN][Eval] 当前进程内未检测到可用的 CUDA GPU，评估将退回到 CPU。\n"
        "如果你确认有 GPU，请检查：Python 环境 / 容器 / 驱动 / CUDA 版本是否与本脚本一致。"
    )
    return torch.device("cpu")


# ===========================
# 数据适配：A-OKVQA & CSQA
# ===========================

def load_aokvqa_test(cfg: EvalConfig) -> Dataset:
    """
    从 parquet 加载 A-OKVQA 测试集，并仅保留前 NUM_SAMPLES_PER_TASK 条样本。

    期望字段（与训练集类似）：
        - image: dict(bytes=...) 或原始 bytes
        - question: str
        - choices: List[str]
        - correct_choice_idx: int
    """
    if not os.path.exists(cfg.AOKVQA_TEST_PARQUET):
        raise FileNotFoundError(f"A-OKVQA 测试 parquet 不存在：{cfg.AOKVQA_TEST_PARQUET}")

    print(f"[INFO] 正在加载 A-OKVQA 测试集：{cfg.AOKVQA_TEST_PARQUET}")
    ds = Dataset.from_parquet(cfg.AOKVQA_TEST_PARQUET)
    print(f"[INFO] A-OKVQA 原始测试样本数：{len(ds)}")

    # 若 NUM_SAMPLES_PER_TASK > 0，则仅保留前 N 条样本；否则使用全量测试集
    if cfg.NUM_SAMPLES_PER_TASK > 0:
        n = min(cfg.NUM_SAMPLES_PER_TASK, len(ds))
        ds = ds.select(range(n))
        print(f"[INFO] A-OKVQA 采样后样本数：{len(ds)}")
    else:
        print("[INFO] A-OKVQA 使用全量测试集进行评估。")

    return ds


def load_csqa_test(cfg: EvalConfig) -> Dataset:
    """
    从 parquet 加载 CSQA 测试集，并仅保留前 NUM_SAMPLES_PER_TASK 条样本。

    期望字段（与 CommonsenseQA 类似）：
        - question: str
        - choices: dict(label=[A,B,C,D,...], text=[...])
        - answerKey: 正确选项字母
    """
    if not os.path.exists(cfg.CSQA_TEST_PARQUET):
        raise FileNotFoundError(f"CSQA 测试 parquet 不存在：{cfg.CSQA_TEST_PARQUET}")

    print(f"[INFO] 正在加载 CSQA 测试集：{cfg.CSQA_TEST_PARQUET}")
    ds = Dataset.from_parquet(cfg.CSQA_TEST_PARQUET)
    print(f"[INFO] CSQA 原始测试样本数：{len(ds)}")

    # 若 NUM_SAMPLES_PER_TASK > 0，则仅保留前 N 条样本；否则使用全量测试集
    if cfg.NUM_SAMPLES_PER_TASK > 0:
        n = min(cfg.NUM_SAMPLES_PER_TASK, len(ds))
        ds = ds.select(range(n))
        print(f"[INFO] CSQA 采样后样本数：{len(ds)}")
    else:
        print("[INFO] CSQA 使用全量测试集进行评估。")

    return ds


def aokvqa_example_to_common(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 A-OKVQA 的单条样本转换为统一格式：
        {
            "question": str,
            "choices": List[str],  # 文本选项
            "label_idx": int,      # 正确选项索引（0-based）
            "image": PIL.Image,
            "source": "aokvqa",
        }
    """
    question = ex.get("question", "")
    choices = ex.get("choices", [])
    if not isinstance(choices, list):
        choices = list(choices)

    # 正确答案索引
    label_idx = 0
    if "correct_choice_idx" in ex and ex["correct_choice_idx"] is not None:
        try:
            label_idx = int(ex["correct_choice_idx"])
        except Exception:
            label_idx = 0

    # 处理图像：支持 dict(bytes=...) 或直接 bytes
    image_data = ex.get("image", None)
    pil_image = PILImage.new("RGB", (224, 224), color="black")
    try:
        if isinstance(image_data, dict) and "bytes" in image_data:
            pil_image = PILImage.open(BytesIO(image_data["bytes"])).convert("RGB")
        elif isinstance(image_data, (bytes, bytearray)):
            pil_image = PILImage.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        # 若解析失败，则退回黑图，但保持流程继续
        pil_image = PILImage.new("RGB", (224, 224), color="black")

    return {
        "question": question,
        "choices": choices,
        "label_idx": label_idx,
        "image": pil_image,
        "source": "aokvqa",
    }


def csqa_example_to_common(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 CSQA 的单条样本转换为统一格式：
        {
            "question": str,
            "choices": List[str],  # 文本选项，如 ["(A) ...", "(B) ...", ...]
            "label_idx": int,      # 正确选项索引（0-based）
            "image": PIL.Image,    # 224x224 黑色占位图
            "source": "csqa",
        }
    """
    question = ex.get("question", "")
    raw_choices = ex.get("choices", {})
    choices_list: List[str] = []
    labels: List[str] = []

    # 解析 choices: dict(label=[...], text=[...])
    if isinstance(raw_choices, dict) and "label" in raw_choices and "text" in raw_choices:
        labels = list(raw_choices.get("label", []))
        texts = list(raw_choices.get("text", []))
        for lab, txt in zip(labels, texts):
            choices_list.append(f"{txt}")
    else:
        # 其他格式：作为简单列表处理
        if isinstance(raw_choices, list):
            choices_list = [str(c) for c in raw_choices]
        else:
            choices_list = [str(raw_choices)]

    # 正确答案字母
    answer_key = ex.get("answerKey", None)
    label_idx = 0
    if answer_key is not None and labels:
        try:
            idx = labels.index(answer_key)
            if 0 <= idx < len(choices_list):
                label_idx = idx
        except ValueError:
            label_idx = 0

    # 生成黑色占位图
    pil_image = PILImage.new("RGB", (224, 224), color="black")

    return {
        "question": question,
        "choices": choices_list,
        "label_idx": label_idx,
        "image": pil_image,
        "source": "csqa",
    }


def build_prompt(question: str, choices: List[str], with_vision_prefix: bool = False) -> str:
    """构造统一的多项选择 Prompt。"""
    lines: List[str] = []
    # 对于多模态任务（例如 A-OKVQA），在文本最前面加上视觉占位符，让 Qwen2.5-VL 知道有图像输入。
    if with_vision_prefix:
        lines.append("<|vision_start|><|image_pad|><|vision_end|>")

    lines.append(f"Question: {question}")
    lines.append("Choices:")
    letters = ["A", "B", "C", "D", "E", "F"]
    for i, choice in enumerate(choices):
        if i >= len(letters):
            break
        lines.append(f"({letters[i]}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


# ===========================
# MoICV 专家权重加载（与训练脚本逻辑保持一致）
# ===========================

def resize_expert_tensor(expert: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    将形状为 [num_experts, dim] 的专家张量沿最后一维拉伸到 target_dim。
    采用 1D 线性插值，与 train_pipeline.py 中的实现保持一致。
    """
    if expert.dim() != 2:
        raise ValueError(f"expert 必须是二维张量 [num_experts, dim]，当前为 {expert.shape}")

    num_experts, dim = expert.shape
    if dim == target_dim:
        return expert.clone()

    expert_1d = expert.unsqueeze(1)  # [N,1,D]
    expert_resized = F.interpolate(
        expert_1d,
        size=target_dim,
        mode="linear",
        align_corners=False,
    ).squeeze(1)  # [N, target_dim]
    return expert_resized


def load_and_assign_experts_for_eval(
    moicv_layer: Dual_MoICV_Layer,
    expert_init_path: str,
    moicv_weights_path: str,
    hidden_size: int,
) -> None:
    """
    在评估阶段加载 MoICV 专家权重：
        1. 先从 expert_init_tensors.pt 中加载 896 维的初始化向量，并拉伸到 hidden_size；
        2. 再从 moicv_trained_weights.bin 中加载微调后的 MoICV 层参数（覆盖初始值）。

    这样既保证维度对齐逻辑的一致性，也能正确加载训练好的参数。
    """
    if not os.path.exists(expert_init_path):
        raise FileNotFoundError(f"找不到专家初始化文件：{expert_init_path}")
    if not os.path.exists(moicv_weights_path):
        raise FileNotFoundError(f"找不到 MoICV 训练权重文件：{moicv_weights_path}")

    print(f"[INFO] (Eval) 加载专家初始化权重：{expert_init_path}")
    init_state = torch.load(expert_init_path, map_location="cpu")

    required_keys = {"E_attn_vis", "E_ffn_vis", "E_attn_text", "E_ffn_text"}
    if not required_keys.issubset(init_state.keys()):
        raise KeyError(f"expert_init_tensors.pt 中缺少必要键：{required_keys - set(init_state.keys())}")

    E_attn_vis_896: torch.Tensor = init_state["E_attn_vis"]
    E_ffn_vis_896: torch.Tensor = init_state["E_ffn_vis"]
    E_attn_text_896: torch.Tensor = init_state["E_attn_text"]
    E_ffn_text_896: torch.Tensor = init_state["E_ffn_text"]

    # 1D 插值到 hidden_size
    E_attn_vis_resized = resize_expert_tensor(E_attn_vis_896, hidden_size)
    E_ffn_vis_resized = resize_expert_tensor(E_ffn_vis_896, hidden_size)
    E_attn_text_resized = resize_expert_tensor(E_attn_text_896, hidden_size)
    E_ffn_text_resized = resize_expert_tensor(E_ffn_text_896, hidden_size)

    with torch.no_grad():
        target_dtype = moicv_layer.E_attn_vis.dtype
        moicv_layer.E_attn_vis.data.copy_(E_attn_vis_resized.to(dtype=target_dtype))
        moicv_layer.E_ffn_vis.data.copy_(E_ffn_vis_resized.to(dtype=target_dtype))
        moicv_layer.E_attn_text.data.copy_(E_attn_text_resized.to(dtype=target_dtype))
        moicv_layer.E_ffn_text.data.copy_(E_ffn_text_resized.to(dtype=target_dtype))

        attn_all = torch.cat(
            [moicv_layer.E_attn_vis.data, moicv_layer.E_attn_text.data],
            dim=0,
        )  # [8, hidden_size]
        ffn_all = torch.cat(
            [moicv_layer.E_ffn_vis.data, moicv_layer.E_ffn_text.data],
            dim=0,
        )
        E_attn_general = attn_all.mean(dim=0, keepdim=True)
        E_ffn_general = ffn_all.mean(dim=0, keepdim=True)

        moicv_layer.E_attn_general.data.copy_(E_attn_general.to(dtype=target_dtype))
        moicv_layer.E_ffn_general.data.copy_(E_ffn_general.to(dtype=target_dtype))

    # 然后加载训练后的 state_dict 覆盖上述初始化
    print(f"[INFO] (Eval) 加载 MoICV 训练权重：{moicv_weights_path}")
    trained_state = torch.load(moicv_weights_path, map_location="cpu")
    moicv_layer.load_state_dict(trained_state, strict=True)
    # 防御性措施：确保整个 MoICV 层的参数 dtype 与当前 target_dtype（通常与主模型一致）完全对齐
    moicv_layer.to(dtype=target_dtype)
    print("[INFO] (Eval) MoICV 权重加载完成，并已对齐 dtype。")


# ===========================
# Logits probe & 路由统计
# ===========================

def get_choice_token_ids(processor: AutoProcessor, letters: List[str]) -> List[int]:
    """
    获取选项字母的 token id 列表。
    为了兼容 BPE 分词，这里取 encode(letter, add_special_tokens=False) 的第一个 token 作为代表。
    """
    # 对于 Qwen2.5-VL，AutoProcessor 内部通常包含 tokenizer 属性
    tok = getattr(processor, "tokenizer", processor)

    ids: List[int] = []
    for ch in letters:
        toks = tok.encode(ch, add_special_tokens=False)
        if not toks:
            raise ValueError(f"无法为字母 '{ch}' 找到对应 token，请检查 tokenizer。")
        ids.append(toks[0])
    return ids


def logits_probe_batch(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    choice_token_ids: List[int],
) -> Tuple[List[int], List[float]]:
    """
    对一个 batch 的 logits 执行 logits probe：
    - 取每个样本最后一个非 padding token 位置的 logits（对应 "Answer:" 之后的下一个预测）
    - 在给定的 choice_token_ids 上做 softmax，取 argmax 作为预测选项索引
    - 计算 4 类分布的熵（自然对数）

    参数：
        logits: [B, S, V]
        attention_mask: [B, S]
        choice_token_ids: List[int]，例如 [id_A, id_B, id_C, id_D]

    返回：
        pred_indices: List[int]，长度为 B，表示预测选项索引（0-based）
        entropies:    List[float]，每个样本的 4 类分布熵
    """
    B, S, V = logits.shape
    device = logits.device

    # 找到每个样本最后一个有效位置 index（attention_mask==1 的最后一位）
    # 这里假设 padding=0, 非 pad=1
    lengths = attention_mask.sum(dim=1)  # [B]
    last_indices = (lengths - 1).clamp(min=0)  # [B]

    # 收集对应位置的 logits: [B, V]
    batch_indices = torch.arange(B, device=device)
    final_logits = logits[batch_indices, last_indices]  # [B, V]

    choice_ids_tensor = torch.tensor(choice_token_ids, device=device, dtype=torch.long)  # [C]
    # [B, C]
    choice_logits = final_logits[:, choice_ids_tensor]

    # softmax -> 概率分布
    probs = F.softmax(choice_logits, dim=-1)  # [B, C]
    pred_indices = torch.argmax(probs, dim=-1).tolist()

    # 信息熵：-sum p log p
    entropies_tensor = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # [B]
    entropies = entropies_tensor.tolist()

    return pred_indices, entropies


def accumulate_routing_stats(
    logits_attn: torch.Tensor,
    logits_ffn: torch.Tensor,
    counters_attn: List[int],
    counters_ffn: List[int],
) -> None:
    """
    累积 MoICV 的路由统计信息：
        - 对每个样本，分别在 attn / ffn 路上取 top-1 专家索引（0-7）
        - 在 counters 数组中增加计数
    """
    # [B, 8]
    top1_attn = torch.argmax(logits_attn, dim=-1)  # [B]
    top1_ffn = torch.argmax(logits_ffn, dim=-1)

    for idx in top1_attn.tolist():
        counters_attn[idx] += 1
    for idx in top1_ffn.tolist():
        counters_ffn[idx] += 1


def print_routing_report(task_name: str, counters_attn: List[int], counters_ffn: List[int], total_samples: int) -> None:
    """打印 MoICV 路由分布统计信息，帮助观察视觉专家是否更多被 A-OKVQA 激活。"""
    def _format_line(counters: List[int], label: str) -> str:
        parts = []
        for i, cnt in enumerate(counters):
            pct = 100.0 * cnt / max(total_samples, 1)
            # 前 4 个为视觉专家，后 4 个为文本专家
            typ = "V" if i < 4 else "T"
            parts.append(f"E{i}({typ}):{pct:.1f}%")
        return f"{label}: " + " | ".join(parts)

    print(f"\n[Routing Report] {task_name}")
    print(_format_line(counters_attn, "Attention 路 Top-1"))
    print(_format_line(counters_ffn, "FFN 路 Top-1"))


# ===========================
# 主评估流程
# ===========================

def evaluate_task(
    task_name: str,
    examples: List[Dict[str, Any]],
    model,
    processor: AutoProcessor,
    device: torch.device,
    use_moicv: bool,
    wrapper: MoICV_Qwen_Wrapper | None,
    moicv_loss_fn: MoICV_Loss | None,
    hidden_size: int,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    """
    对单个任务（A-OKVQA / CSQA）进行评估。

    参数：
        task_name: "A-OKVQA" 或 "CSQA"
        examples:  已转换为统一格式的样本列表
        model:     Baseline 模型（未注入 MoICV）
        tokenizer: 对应 tokenizer
        use_moicv: 是否启用 MoICV 注入
        wrapper:   若 use_moicv=True，则为 MoICV_Qwen_Wrapper 实例；否则为 None
        moicv_loss_fn: MoICV_Loss 实例（仅为保持接口统一，这里不使用 loss 值）
        hidden_size: LLM hidden_size
        cfg: EvalConfig
    """
    print(f"\n[INFO] 开始评估任务：{task_name} | USE_MOICV={use_moicv}")

    letters = ["A", "B", "C", "D"]
    choice_token_ids = get_choice_token_ids(processor, letters)

    total = len(examples)
    correct = 0
    all_entropies: List[float] = []

    # 路由统计（仅在 MoICV 模式下启用）
    attn_counters = [0] * 8
    ffn_counters = [0] * 8

    # 分 batch 推理
    for start in tqdm(range(0, total, cfg.BATCH_SIZE), desc=f"{task_name} Eval ({'MoICV' if use_moicv else 'Baseline'})"):
        end = min(start + cfg.BATCH_SIZE, total)
        batch_examples = examples[start:end]

        prompts = [
            build_prompt(
                ex["question"],
                ex["choices"],
                with_vision_prefix=(task_name == "A-OKVQA"),
            )
            for ex in batch_examples
        ]
        labels_idx = [int(ex["label_idx"]) for ex in batch_examples]

        # Qwen2.5-VL 多模态输入：A-OKVQA 传入图片，CSQA 仅传文本
        images = [ex["image"] for ex in batch_examples]

        if task_name == "A-OKVQA":
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        with torch.no_grad():
            if use_moicv:
                assert wrapper is not None

                # 与 train.py 对齐：使用文本词嵌入的加权平均作为 Router 的 query_features
                embed_layer = wrapper.llm_model.get_input_embeddings()
                token_embeds = embed_layer(input_ids)  # [B, S, H]
                mask = attention_mask.unsqueeze(-1)      # [B, S, 1]
                masked_embeds = token_embeds * mask      # [B, S, H]
                lengths = mask.sum(dim=1).clamp(min=1)   # [B, 1]
                query_features = masked_embeds.sum(dim=1) / lengths  # [B, H]
                # 为避免 dtype 不匹配（例如 Router 参数为 FP16 而 query 为 FP32）导致 matmul 报错，
                # 在评估阶段将 query_features 的 dtype 显式对齐到模型的 dtype。
                query_features = query_features.to(dtype=model.dtype)

                # 组装传入 Wrapper 的关键字参数（包含多模态输入）
                wrapper_kwargs = dict(
                    query_features=query_features,
                    compute_moicv_loss=False,
                    moicv_loss_fn=moicv_loss_fn,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                if pixel_values is not None:
                    wrapper_kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    wrapper_kwargs["image_grid_thw"] = image_grid_thw

                outputs = wrapper.forward_with_moicv(**wrapper_kwargs)
                llm_outputs = outputs["llm_outputs"]
                logits_attn = outputs["logits_attn"]
                logits_ffn = outputs["logits_ffn"]

                logits = llm_outputs["logits"] if isinstance(llm_outputs, dict) else llm_outputs.logits

                # 累积路由统计
                accumulate_routing_stats(
                    logits_attn=logits_attn,
                    logits_ffn=logits_ffn,
                    counters_attn=attn_counters,
                    counters_ffn=ffn_counters,
                )
            else:
                # Baseline：直接调用原始 LLM，多模态输入同样传入 pixel_values / image_grid_thw
                model_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                if pixel_values is not None:
                    model_kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    model_kwargs["image_grid_thw"] = image_grid_thw

                llm_outputs = model(**model_kwargs)
                logits = llm_outputs["logits"] if isinstance(llm_outputs, dict) else llm_outputs.logits

            # logits probe
            pred_indices, entropies = logits_probe_batch(
                logits=logits,
                attention_mask=attention_mask,
                choice_token_ids=choice_token_ids,
            )

        # 更新准确率与熵统计
        for p, t in zip(pred_indices, labels_idx):
            if p == t:
                correct += 1
        all_entropies.extend(entropies)

    acc = correct / max(total, 1)
    mean_entropy = sum(all_entropies) / max(len(all_entropies), 1)

    print(f"[RESULT] {task_name} | USE_MOICV={use_moicv} | Acc={acc:.4f} | MeanEntropy={mean_entropy:.4f}")

    if use_moicv:
        print_routing_report(
            task_name=task_name,
            counters_attn=attn_counters,
            counters_ffn=ffn_counters,
            total_samples=total,
        )

    return {
        "accuracy": acc,
        "mean_entropy": mean_entropy,
        "routing_attn": attn_counters if use_moicv else None,
        "routing_ffn": ffn_counters if use_moicv else None,
        "total": total,
    }


def main() -> None:
    cfg = CFG
    set_seed(cfg.SEED)

    device = get_main_device()
    print(f"[INFO] 评估主设备：{device}")

    # 1. 加载 LLM 模型和 tokenizer（与 train.py 高配版本对齐：固定 FP16 + 绑定单卡）
    model_dtype = torch.float16
    print("[INFO] 评估固定使用 torch.float16 作为模型精度。")

    device_map = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    print(f"[INFO] 正在加载 Qwen2.5-VL 模型：{cfg.MODEL_PATH}，device_map={device_map}")

    model = QwenVLModel.from_pretrained(
        cfg.MODEL_PATH,
        torch_dtype=model_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    # 使用 AutoProcessor 统一处理文本和图像输入（内部包含 tokenizer）
    processor = AutoProcessor.from_pretrained(
        cfg.MODEL_PATH,
        trust_remote_code=True,
    )

    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError("model.config 中未找到 hidden_size 字段，请检查模型配置。")
    print(f"[INFO] 模型 hidden_size = {hidden_size}")

    # 2. 准备 MoICV 层与 Wrapper（仅在 USE_MOICV=True 时真正使用）
    moicv_layer = Dual_MoICV_Layer(
        query_dim=hidden_size,
        attn_dim=hidden_size,
        ffn_dim=hidden_size,
    ).to(device=device, dtype=model_dtype)

    # 加载专家初始化 + 训练后权重
    load_and_assign_experts_for_eval(
        moicv_layer=moicv_layer,
        expert_init_path=cfg.EXPERT_INIT_PATH,
        moicv_weights_path=cfg.MOICV_WEIGHTS_PATH,
        hidden_size=hidden_size,
    )

    wrapper = MoICV_Qwen_Wrapper(
        llm_model=model,
        moicv_layer=moicv_layer,
        attn_inject_layer_idx=cfg.ATTN_INJECT_LAYER_IDX,
        ffn_inject_layer_idx=cfg.FFN_INJECT_LAYER_IDX,
    )
    wrapper.eval()

    moicv_loss_fn = MoICV_Loss(alpha=0.1, beta=0.1)  # 这里只是占位，不参与评估 loss

    # 3. 加载并适配测试数据
    ds_aokvqa_raw = load_aokvqa_test(cfg)
    ds_csqa_raw = load_csqa_test(cfg)

    aokvqa_examples = [aokvqa_example_to_common(ex) for ex in ds_aokvqa_raw]
    csqa_examples = [csqa_example_to_common(ex) for ex in ds_csqa_raw]

    # 5. 串行运行 Baseline 与 MoICV
    results = {
        "A-OKVQA": {"baseline": None, "moicv": None},
        "CSQA": {"baseline": None, "moicv": None},
    }

    # --- Baseline ---
    results["A-OKVQA"]["baseline"] = evaluate_task(
        task_name="A-OKVQA",
        examples=aokvqa_examples,
        model=model,
        processor=processor,
        device=device,
        use_moicv=False,
        wrapper=None,
        moicv_loss_fn=None,
        hidden_size=hidden_size,
        cfg=cfg,
    )
    results["CSQA"]["baseline"] = evaluate_task(
        task_name="CSQA",
        examples=csqa_examples,
        model=model,
        processor=processor,
        device=device,
        use_moicv=False,
        wrapper=None,
        moicv_loss_fn=None,
        hidden_size=hidden_size,
        cfg=cfg,
    )

    # --- Ours (MoICV 注入) ---
    results["A-OKVQA"]["moicv"] = evaluate_task(
        task_name="A-OKVQA",
        examples=aokvqa_examples,
        model=model,
        processor=processor,
        device=device,
        use_moicv=True,
        wrapper=wrapper,
        moicv_loss_fn=moicv_loss_fn,
        hidden_size=hidden_size,
        cfg=cfg,
    )
    results["CSQA"]["moicv"] = evaluate_task(
        task_name="CSQA",
        examples=csqa_examples,
        model=model,
        processor=processor,
        device=device,
        use_moicv=True,
        wrapper=wrapper,
        moicv_loss_fn=moicv_loss_fn,
        hidden_size=hidden_size,
        cfg=cfg,
    )

    # 6. 打印对比表 & 导出 radar_plot_data.json
    def _delta(b: float, m: float) -> float:
        return 100.0 * (m - b)

    a_base_acc = results["A-OKVQA"]["baseline"]["accuracy"]
    a_moicv_acc = results["A-OKVQA"]["moicv"]["accuracy"]
    c_base_acc = results["CSQA"]["baseline"]["accuracy"]
    c_moicv_acc = results["CSQA"]["moicv"]["accuracy"]

    print("\n================== 对比报告 ==================")
    print("| Task              | Metric   | Baseline (Acc) | MoICV (Acc) | Delta    |")
    print("| ----------------- | -------- | -------------- | ----------- | -------- |")
    print(
        f"| A-OKVQA (500)     | Accuracy | {a_base_acc*100:6.2f}%       | "
        f"{a_moicv_acc*100:6.2f}%    | { _delta(a_base_acc, a_moicv_acc):+6.2f}% |"
    )
    print(
        f"| CSQA (500)        | Accuracy | {c_base_acc*100:6.2f}%       | "
        f"{c_moicv_acc*100:6.2f}%    | { _delta(c_base_acc, c_moicv_acc):+6.2f}% |"
    )
    print("================================================\n")

    radar_data = {
        "A-OKVQA": {
            "baseline_acc": a_base_acc,
            "moicv_acc": a_moicv_acc,
            "delta": a_moicv_acc - a_base_acc,
        },
        "CSQA": {
            "baseline_acc": c_base_acc,
            "moicv_acc": c_moicv_acc,
            "delta": c_moicv_acc - c_base_acc,
        },
    }

    with open("radar_plot_data.json", "w", encoding="utf-8") as f:
        json.dump(radar_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 评估结果已保存到 radar_plot_data.json")


if __name__ == "__main__":
    main()


