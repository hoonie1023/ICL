#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the learned MoICV router strategy across domains and visualize it
as a heatmap (similar to RISER Figure 6).

For each domain (e.g., A-OKVQA, CSQA), we:
  1. Sample a small subset of M²IV distillation records.
  2. Rebuild the Student-style 0-shot prompt using Qwen2.5-VL-7B.
  3. Extract query_features from the input embeddings (same as train_distill.py).
  4. Pass query_features through the trained Dual_MoICV_Layer to obtain
     routing logits over 8 experts.
  5. Compute average softmax weights per expert per domain.

Finally, we plot a heatmap:
  - Y-axis: domains
  - X-axis: experts (Vis-1..Vis-4, Text-1..Text-4)
  - Values: average routing weight
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoProcessor, set_seed

from moicv_core import Dual_MoICV_Layer
from train import get_main_device, load_and_assign_experts
from train_distill import DistillConfig, build_qa_text, decode_image_field, extract_qa_from_raw

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    from transformers import AutoModelForCausalLM as QwenVLModel


CFG = DistillConfig()

# How many samples per domain to evaluate
SAMPLES_PER_DOMAIN = 150

# Path to trained MoICV router weights
ROUTER_CKPT_NAME = "moicv_distill_trained_gpu.bin"


def load_raw_parquet_datasets(cfg: DistillConfig) -> Dict[str, Dataset]:
    """Load raw A-OKVQA / CSQA parquet datasets."""
    if not os.path.exists(cfg.AOKVQA_TRAIN_PARQUET):
        raise FileNotFoundError(f"A-OKVQA parquet not found: {cfg.AOKVQA_TRAIN_PARQUET}")
    if not os.path.exists(cfg.CSQA_TRAIN_PARQUET):
        raise FileNotFoundError(f"CSQA parquet not found: {cfg.CSQA_TRAIN_PARQUET}")

    print(f"[INFO][EvalRouter] Loading A-OKVQA train parquet: {cfg.AOKVQA_TRAIN_PARQUET}")
    ds_aokvqa = Dataset.from_parquet(cfg.AOKVQA_TRAIN_PARQUET)
    print(f"[INFO][EvalRouter] A-OKVQA train size: {len(ds_aokvqa)}")

    print(f"[INFO][EvalRouter] Loading CSQA train parquet: {cfg.CSQA_TRAIN_PARQUET}")
    ds_csqa = Dataset.from_parquet(cfg.CSQA_TRAIN_PARQUET)
    print(f"[INFO][EvalRouter] CSQA train size: {len(ds_csqa)}")

    return {
        "aokvqa": ds_aokvqa,
        "csqa": ds_csqa,
    }


def build_query_features(
    model: QwenVLModel,
    processor: AutoProcessor,
    q_target: str,
    a_target: str,
    image_raw: Any,
    device: torch.device,
) -> torch.Tensor:
    """
    Rebuild Student-style 0-shot text+image input and compute query_features
    (mean pooled input embeddings), matching train_distill.py.
    """
    student_text = build_qa_text(q_target, a_target, with_vision_prefix=True)
    student_image = decode_image_field(image_raw)

    with torch.no_grad():
        student_inputs = processor(
            text=[student_text],
            images=[student_image],
            return_tensors="pt",
            padding=True,
        )
        student_inputs = {k: v.to(device) for k, v in student_inputs.items()}

        input_ids = student_inputs["input_ids"]           # [1,S]
        attention_mask = student_inputs["attention_mask"] # [1,S]

        embed_layer = model.get_input_embeddings()
        token_embeds = embed_layer(input_ids)             # [1,S,H]
        mask = attention_mask.unsqueeze(-1)               # [1,S,1]
        masked_embeds = token_embeds * mask               # [1,S,H]
        lengths = mask.sum(dim=1).clamp(min=1)            # [1,1]
        query_features = masked_embeds.sum(dim=1) / lengths  # [1,H]
        query_features = query_features.to(dtype=torch.float32)

    return query_features


def main() -> None:
    cfg = CFG
    set_seed(cfg.SEED)
    rng = random.Random(cfg.SEED)

    device = get_main_device()
    print(f"[INFO][EvalRouter] Using device: {device}")

    # 1. Load backbone and processor (frozen)
    model_path = cfg.MODEL_PATH
    device_map = "cuda:0" if device.type == "cuda" else "cpu"
    print(f"[INFO][EvalRouter] Loading Qwen2.5-VL from {model_path}, device_map={device_map}")

    model = QwenVLModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
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
    print(f"[INFO][EvalRouter] Model hidden_size = {hidden_size}")

    # 2. Load trained MoICV router (experts + router weights)
    moicv_layer = Dual_MoICV_Layer(
        query_dim=hidden_size,
        attn_dim=hidden_size,
        ffn_dim=hidden_size,
    )
    moicv_layer.to(device=device, dtype=torch.float32)

    # First initialize from expert_init_tensors.pt (handles both old/new formats),
    # then load the trained router checkpoint to restore full state.
    load_and_assign_experts(
        moicv_layer=moicv_layer,
        expert_init_path=cfg.EXPERT_INIT_PATH,
        hidden_size=hidden_size,
    )

    router_ckpt_path = os.path.join(cfg.OUTPUT_DIR, ROUTER_CKPT_NAME)
    if not os.path.exists(router_ckpt_path):
        raise FileNotFoundError(f"Router checkpoint not found: {router_ckpt_path}")
    state_dict = torch.load(router_ckpt_path, map_location="cpu")
    moicv_layer.load_state_dict(state_dict, strict=True)
    moicv_layer.eval()

    # 3. Load M²IV distillation dataset and raw parquets
    if not os.path.exists(cfg.DATASET_PATH):
        raise FileNotFoundError(f"M²IV distillation dataset not found at {cfg.DATASET_PATH}")
    print(f"[INFO][EvalRouter] Loading distillation dataset: {cfg.DATASET_PATH}")
    hf_dataset = load_from_disk(cfg.DATASET_PATH)
    n_samples = len(hf_dataset)
    print(f"[INFO][EvalRouter] Distill records: {n_samples}")

    raw_ds_map = load_raw_parquet_datasets(cfg)

    # Group indices by domain
    sources_all: List[str] = hf_dataset["source"]
    domain_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(sources_all):
        domain_to_indices.setdefault(str(s), []).append(i)

    domains = sorted(domain_to_indices.keys())
    print(
        "[INFO][EvalRouter] Domain sample counts: "
        + ", ".join(f"{d}={len(domain_to_indices[d])}" for d in domains)
    )

    # 4. For each domain, sample SAMPLES_PER_DOMAIN examples and collect router weights
    domain_weights: Dict[str, List[np.ndarray]] = {d: [] for d in domains}

    for domain in domains:
        idx_list = domain_to_indices[domain]
        if not idx_list:
            continue
        rng.shuffle(idx_list)
        picked = idx_list[: min(SAMPLES_PER_DOMAIN, len(idx_list))]

        print(f"[INFO][EvalRouter] Evaluating domain '{domain}' on {len(picked)} samples.")
        ds_raw = raw_ds_map.get(domain, None)
        if ds_raw is None:
            # Fallback: if new domain not in raw_ds_map, skip it gracefully
            print(f"[WARN][EvalRouter] No raw parquet mapping for domain '{domain}', skipping.")
            continue

        for idx in tqdm(picked, desc=f"[EvalRouter] {domain}", unit="sample"):
            ex_m2iv = hf_dataset[idx]
            target_index = int(ex_m2iv["target_index"])

            target_raw = ds_raw[target_index]
            q_target, a_target = extract_qa_from_raw(target_raw, source=domain)

            query_features = build_query_features(
                model=model,
                processor=processor,
                q_target=q_target,
                a_target=a_target,
                image_raw=target_raw.get("image", None),
                device=device,
            )  # [1,H]

            # 使用最新的 MoICV 双头接口：v_inject, routing_weights
            with torch.no_grad():
                v_inject, routing_weights = moicv_layer(query_features)
                weights = routing_weights[0].detach().cpu().numpy()  # [8]
            domain_weights[domain].append(weights)

    # 5. Aggregate to domain-wise average weights
    domain_names: List[str] = []
    avg_matrix: List[np.ndarray] = []
    for d in domains:
        w_list = domain_weights.get(d, [])
        if not w_list:
            continue
        mat = np.stack(w_list, axis=0)  # [N, 8]
        avg = mat.mean(axis=0)          # [8]
        domain_names.append(d)
        avg_matrix.append(avg)

    if not avg_matrix:
        raise RuntimeError("No router weights collected; please check dataset and configuration.")

    avg_matrix_np = np.stack(avg_matrix, axis=0)  # [D, 8]

    # 6. Plot heatmap
    expert_labels = [
        "Vis-Exp1",
        "Vis-Exp2",
        "Vis-Exp3",
        "Vis-Exp4",
        "Text-Exp1",
        "Text-Exp2",
        "Text-Exp3",
        "Text-Exp4",
    ]

    plt.figure(figsize=(10, 4 + 0.4 * len(domain_names)))
    sns.set(style="whitegrid")

    ax = sns.heatmap(
        avg_matrix_np,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticks=np.arange(len(expert_labels)) + 0.5,
        yticks=np.arange(len(domain_names)) + 0.5,
        cbar_kws={"label": "Average Routing Weight"},
    )

    ax.set_xticklabels(expert_labels, rotation=45, ha="right")
    ax.set_yticklabels(domain_names, rotation=0)
    ax.set_xlabel("Experts (4 Visual, 4 Text)")
    ax.set_ylabel("Benchmark Domains")
    ax.set_title("MoICV Router Strategy Heatmap")

    plt.tight_layout()
    out_path = "router_strategy_heatmap.png"
    plt.savefig(out_path, dpi=300)
    print(f"[INFO][EvalRouter] Heatmap saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()

