#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Initialize MoICV expert vectors using Contrastive Activation Addition (CAA)
directly from Qwen2.5-VL-7B hidden states.

Pipeline:
  1. Load base Qwen2.5-VL-7B-Instruct and its AutoProcessor (same as train_distill.py).
  2. For each sample in the pre-built M²IV distillation dataset:
       - Build a 16-shot (teacher-style) multimodal prompt.
       - Build a 0-shot (student-style) prompt.
       - Run both through the backbone with output_hidden_states=True.
       - Extract last-token hidden state at TARGET_LAYER for each view.
       - Compute delta_h = h_16shot - h_0shot (CAA).
       - Accumulate into visual (A-OKVQA) or text (CSQA) pools.
  3. Run K-Means (K=4) on deltas for each pool.
  4. L2-normalize centroids and save as latent ICL expert vectors:
       {
         "E_vis":  [4, hidden_size],
         "E_text": [4, hidden_size],
       }
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from transformers import AutoProcessor, set_seed

# Reuse core config and prompt / data helpers from train_distill.py
from train_distill import (
    DistillConfig,
    build_qa_text,
    decode_image_field,
    extract_qa_from_raw,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    from transformers import AutoModelForCausalLM as QwenVLModel


# ===========================
# Global config
# ===========================

CFG = DistillConfig()

TARGET_LAYER = 15          # layer index for CAA extraction
MAX_SAMPLES = 500          # how many M²IV records to use for expert init
MODEL_DTYPE = torch.float16


def get_main_device() -> torch.device:
    """
    Prefer single GPU cuda:0; fall back to CPU if unavailable.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda:0")
    print("[WARN][init_experts] No CUDA GPU detected, falling back to CPU.")
    return torch.device("cpu")


def load_raw_parquet_datasets(cfg: DistillConfig) -> Tuple[Dataset, Dataset]:
    """
    Load raw A-OKVQA / CSQA train parquet datasets.
    """
    if not os.path.exists(cfg.AOKVQA_TRAIN_PARQUET):
        raise FileNotFoundError(f"A-OKVQA parquet not found: {cfg.AOKVQA_TRAIN_PARQUET}")
    if not os.path.exists(cfg.CSQA_TRAIN_PARQUET):
        raise FileNotFoundError(f"CSQA parquet not found: {cfg.CSQA_TRAIN_PARQUET}")

    print(f"[INFO][init_experts] Loading A-OKVQA train parquet: {cfg.AOKVQA_TRAIN_PARQUET}")
    ds_aokvqa = Dataset.from_parquet(cfg.AOKVQA_TRAIN_PARQUET)
    print(f"[INFO][init_experts] A-OKVQA train size: {len(ds_aokvqa)}")

    print(f"[INFO][init_experts] Loading CSQA train parquet: {cfg.CSQA_TRAIN_PARQUET}")
    ds_csqa = Dataset.from_parquet(cfg.CSQA_TRAIN_PARQUET)
    print(f"[INFO][init_experts] CSQA train size: {len(ds_csqa)}")

    return ds_aokvqa, ds_csqa


def get_last_token_hidden(
    hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    """
    Extract last non-pad token hidden state at a specific transformer layer.

    hidden_states: tuple of [B, S, H] tensors (one per layer or including embeddings).
    attention_mask: [B, S]
    layer_index: which entry in hidden_states to use (matches TARGET_LAYER assumption).
    """
    h_layer = hidden_states[layer_index]  # [B, S, H]
    B, S, H = h_layer.shape
    device = h_layer.device

    lengths = attention_mask.sum(dim=1)             # [B]
    last_indices = (lengths - 1).clamp(min=0)       # [B]
    batch_indices = torch.arange(B, device=device)  # [B]
    last_h = h_layer[batch_indices, last_indices]   # [B, H]
    return last_h


def run_kmeans_and_normalize(deltas: List[torch.Tensor], n_clusters: int) -> torch.Tensor:
    """
    Run K-Means on stacked delta vectors and L2-normalize centroids.
    """
    if len(deltas) == 0:
        raise RuntimeError("No delta vectors collected; cannot run K-Means.")

    mat = torch.stack(deltas, dim=0).cpu().numpy()  # [N, H]
    print(f"[INFO][init_experts] Running KMeans on {mat.shape[0]} vectors, dim={mat.shape[1]}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=CFG.SEED, n_init="auto")
    kmeans.fit(mat)
    centers = kmeans.cluster_centers_.astype(np.float32)  # [K, H]

    centers_t = torch.from_numpy(centers)  # [K, H]
    centers_t = F.normalize(centers_t, p=2, dim=-1)  # L2 normalize along hidden dim
    return centers_t


def main() -> None:
    cfg = CFG
    set_seed(cfg.SEED)
    rng = random.Random(cfg.SEED)

    device = get_main_device()
    print(f"[INFO][init_experts] Using device: {device}")

    # 1. Load backbone & processor (same as train_distill.py)
    model_path = cfg.MODEL_PATH
    device_map = "cuda:0" if device.type == "cuda" else "cpu"
    print(f"[INFO][init_experts] Loading Qwen2.5-VL model from {model_path}, device_map={device_map}")

    model = QwenVLModel.from_pretrained(
        model_path,
        torch_dtype=MODEL_DTYPE,
        device_map=device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError("model.config.hidden_size is missing; please check model config.")
    print(f"[INFO][init_experts] Model hidden_size = {hidden_size}")

    # 2. Load M²IV distillation dataset and raw parquets
    if not os.path.exists(cfg.DATASET_PATH):
        raise FileNotFoundError(f"M²IV distillation dataset not found at {cfg.DATASET_PATH}")
    print(f"[INFO][init_experts] Loading M²IV distillation dataset: {cfg.DATASET_PATH}")
    hf_dataset = load_from_disk(cfg.DATASET_PATH)
    n_samples = len(hf_dataset)
    print(f"[INFO][init_experts] Distillation records: {n_samples}")

    ds_aokvqa_raw, ds_csqa_raw = load_raw_parquet_datasets(cfg)

    # 3. CAA: collect delta_h for visual (A-OKVQA) and text (CSQA)
    delta_h_vis: List[torch.Tensor] = []
    delta_h_text: List[torch.Tensor] = []

    indices = list(range(n_samples))
    rng.shuffle(indices)
    limit = min(MAX_SAMPLES, n_samples)
    print(f"[INFO][init_experts] Sampling {limit} records for expert initialization.")

    for k in tqdm(indices[:limit], desc="[init_experts] CAA samples", unit="sample"):
        ex_m2iv: Dict[str, Any] = hf_dataset[k]
        source = ex_m2iv["source"]
        target_index = int(ex_m2iv["target_index"])
        demo_indices = [int(i) for i in ex_m2iv["demo_indices"]]

        # Choose raw dataset
        if source == "aokvqa":
            ds_raw = ds_aokvqa_raw
        else:
            ds_raw = ds_csqa_raw

        target_raw = ds_raw[target_index]
        demos_raw = [ds_raw[i] for i in demo_indices]

        # ----- 3.1 Build 16-shot teacher-style inputs -----
        teacher_texts: List[str] = []
        teacher_images: List[Any] = []

        # 16 demos: full QA
        for ex_demo in demos_raw:
            q_demo, a_demo = extract_qa_from_raw(ex_demo, source=source)
            demo_text = build_qa_text(q_demo, a_demo, with_vision_prefix=True)
            teacher_texts.append(demo_text)
            teacher_images.append(decode_image_field(ex_demo.get("image", None)))

        # Target: full QA as positive supervision context
        q_target, a_target = extract_qa_from_raw(target_raw, source=source)
        target_teacher_text = build_qa_text(q_target, a_target, with_vision_prefix=True)
        teacher_texts.append(target_teacher_text)
        teacher_images.append(decode_image_field(target_raw.get("image", None)))

        # 拼接成一个真正的 16-shot 长上下文；Batch Size = 1
        teacher_16shot_text = "\n\n".join(teacher_texts)

        with torch.no_grad():
            teacher_inputs = processor(
                text=[teacher_16shot_text],
                images=[teacher_images],  # 多图作为同一道题的上下文
                return_tensors="pt",
                padding=True,
            )
            teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

            teacher_outputs = model(**teacher_inputs, output_hidden_states=True)
            teacher_hidden_states = teacher_outputs.hidden_states  # tuple of [B,S,H]
            teacher_attn_mask = teacher_inputs["attention_mask"]

            # Batch Size 现为 1，直接在该样本上取最后一个非 pad token
            h_teacher_all = get_last_token_hidden(
                teacher_hidden_states,
                teacher_attn_mask,
                layer_index=TARGET_LAYER,
            )  # [B, H]
            h_teacher_last = h_teacher_all  # [1, H]

        # ----- 3.2 Build 0-shot student-style inputs -----
        student_text = build_qa_text(q_target, a_target, with_vision_prefix=True)
        student_image = decode_image_field(target_raw.get("image", None))

        with torch.no_grad():
            student_inputs = processor(
                text=[student_text],
                images=[student_image],
                return_tensors="pt",
                padding=True,
            )
            student_inputs = {k: v.to(device) for k, v in student_inputs.items()}

            student_outputs = model(**student_inputs, output_hidden_states=True)
            student_hidden_states = student_outputs.hidden_states  # tuple of [1,S,H]
            student_attn_mask = student_inputs["attention_mask"]

            h_student = get_last_token_hidden(
                student_hidden_states,
                student_attn_mask,
                layer_index=TARGET_LAYER,
            )  # [1, H]

        # ----- 3.3 CAA delta and accumulation -----
        # delta_h: [H]
        delta_h = (h_teacher_last[0] - h_student[0]).to(dtype=torch.float32).detach()

        if source == "aokvqa":
            delta_h_vis.append(delta_h)
        else:
            delta_h_text.append(delta_h)

        # OOM prevention: aggressively free memory
        del teacher_outputs, teacher_hidden_states, teacher_inputs
        del student_outputs, student_hidden_states, student_inputs
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print(
        f"[INFO][init_experts] Collected deltas: "
        f"visual={len(delta_h_vis)}, text={len(delta_h_text)}"
    )

    # 4. K-Means and normalization
    vis_centroids = run_kmeans_and_normalize(delta_h_vis, n_clusters=4)
    text_centroids = run_kmeans_and_normalize(delta_h_text, n_clusters=4)

    print(f"[INFO][init_experts] Visual centroids shape: {tuple(vis_centroids.shape)}")
    print(f"[INFO][init_experts] Text centroids shape:   {tuple(text_centroids.shape)}")

    # 5. Save experts
    expert_tensors = {
        "E_vis": vis_centroids,   # [4, H]
        "E_text": text_centroids, # [4, H]
    }

    save_path = cfg.EXPERT_INIT_PATH
    torch.save(expert_tensors, save_path)
    print(f"[INFO][init_experts] Expert vectors saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()

