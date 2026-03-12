#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised Fine-Tuning (SFT) of the MoICV Router via Oracle Grid Search.

Phase 1: Oracle Label Construction (Grid Search)
  - For a small mixed-domain subset (A-OKVQA + CSQA),
    we run a Teacher 16-shot forward pass to get target distributions,
    and a Student 0-shot forward with manually injected single-expert vectors.
  - For each sample, we choose the expert whose one-hot activation
    yields the lowest KL divergence to the Teacher.
  - We store (query_features, oracle_target_weights) pairs.

Phase 2: Router SFT
  - Using the oracle dataset of size ~200, we train the MoICV router
    to regress its routing_weights to these oracle one-hot targets
    with an MSE loss.
  - The trained router weights are saved to moicv_router_sft.pth
    and later used as warm-start initialization in distillation.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoProcessor, set_seed

from moicv_core import Dual_MoICV_Layer
from moicv_injection import MoICV_Qwen_Wrapper
from train import get_main_device, load_and_assign_experts
from train_distill import (
    DistillConfig,
    build_qa_text,
    decode_image_field,
    extract_qa_from_raw,
    get_last_token_logits,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    from transformers import AutoModelForCausalLM as QwenVLModel


CFG = DistillConfig()

# Number of oracle samples per domain
N_ORACLE_AOKVQA = 100
N_ORACLE_CSQA = 100

# Router SFT hyperparameters
ROUTER_LR = 1e-3
ROUTER_EPOCHS = 15


def load_raw_parquet_datasets(cfg: DistillConfig) -> Tuple[Dataset, Dataset]:
    """Load raw A-OKVQA / CSQA train parquet datasets."""
    if not os.path.exists(cfg.AOKVQA_TRAIN_PARQUET):
        raise FileNotFoundError(f"A-OKVQA parquet not found: {cfg.AOKVQA_TRAIN_PARQUET}")
    if not os.path.exists(cfg.CSQA_TRAIN_PARQUET):
        raise FileNotFoundError(f"CSQA parquet not found: {cfg.CSQA_TRAIN_PARQUET}")

    print(f"[INFO][RouterSFT] Loading A-OKVQA train parquet: {cfg.AOKVQA_TRAIN_PARQUET}")
    ds_aokvqa = Dataset.from_parquet(cfg.AOKVQA_TRAIN_PARQUET)
    print(f"[INFO][RouterSFT] A-OKVQA train size: {len(ds_aokvqa)}")

    print(f"[INFO][RouterSFT] Loading CSQA train parquet: {cfg.CSQA_TRAIN_PARQUET}")
    ds_csqa = Dataset.from_parquet(cfg.CSQA_TRAIN_PARQUET)
    print(f"[INFO][RouterSFT] CSQA train size: {len(ds_csqa)}")

    return ds_aokvqa, ds_csqa


def build_student_inputs_and_query_features(
    model: QwenVLModel,
    processor: AutoProcessor,
    q_target: str,
    a_target: str,
    image_raw: Any,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Build Student-style 0-shot input (target only, with answer) and compute query_features
    via mean pooling of input embeddings, matching train_distill.py.
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

    return student_inputs, query_features


def main() -> None:
    cfg = CFG
    set_seed(cfg.SEED)
    rng = random.Random(cfg.SEED)

    device = get_main_device()
    print(f"[INFO][RouterSFT] Using device: {device}")

    # 1. Load Qwen2.5-VL backbone and processor
    model_path = cfg.MODEL_PATH
    device_map = "cuda:0" if device.type == "cuda" else "cpu"
    print(f"[INFO][RouterSFT] Loading Qwen2.5-VL from {model_path}, device_map={device_map}")

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
    print(f"[INFO][RouterSFT] Model hidden_size = {hidden_size}")

    # 2. Construct MoICV layer and load frozen experts
    moicv_layer = Dual_MoICV_Layer(
        query_dim=hidden_size,
        attn_dim=hidden_size,
        ffn_dim=hidden_size,
    )
    moicv_layer.to(device=device, dtype=torch.float32)

    load_and_assign_experts(
        moicv_layer=moicv_layer,
        expert_init_path=cfg.EXPERT_INIT_PATH,
        hidden_size=hidden_size,
    )

    # Router parameters will be trained in Phase 2; for Phase 1 we only need experts.

    # 3. Build wrapper to install MoICV hook on Qwen2.5-VL
    wrapper = MoICV_Qwen_Wrapper(
        llm_model=model,
        moicv_layer=moicv_layer,
        inject_layer_idx=15,
    )

    # 4. Load M²IV distillation dataset and raw parquets
    if not os.path.exists(cfg.DATASET_PATH):
        raise FileNotFoundError(f"M²IV distillation dataset not found at {cfg.DATASET_PATH}")
    print(f"[INFO][RouterSFT] Loading distillation dataset: {cfg.DATASET_PATH}")
    hf_dataset = load_from_disk(cfg.DATASET_PATH)
    n_samples = len(hf_dataset)
    print(f"[INFO][RouterSFT] Distillation records: {n_samples}")

    ds_aokvqa_raw, ds_csqa_raw = load_raw_parquet_datasets(cfg)

    # Group indices by domain
    sources_all: List[str] = hf_dataset["source"]
    aok_indices: List[int] = []
    csqa_indices: List[int] = []
    for i, s in enumerate(sources_all):
        s_str = str(s)
        if s_str == "aokvqa":
            aok_indices.append(i)
        elif s_str == "csqa":
            csqa_indices.append(i)

    rng.shuffle(aok_indices)
    rng.shuffle(csqa_indices)
    aok_selected = aok_indices[: min(N_ORACLE_AOKVQA, len(aok_indices))]
    csqa_selected = csqa_indices[: min(N_ORACLE_CSQA, len(csqa_indices))]

    print(
        f"[INFO][RouterSFT] Selected {len(aok_selected)} A-OKVQA and "
        f"{len(csqa_selected)} CSQA samples for oracle construction."
    )

    # ---------------
    # Phase 1: Oracle construction via grid search
    # ---------------
    oracle_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (query_features[H], oracle_weights[8])

    # 8 attention experts = 4 visual + 4 text
    experts_attn: torch.Tensor = torch.cat(
        [moicv_layer.E_attn_vis.detach(), moicv_layer.E_attn_text.detach()],
        dim=0,
    )  # [8, H]

    def process_one_index(idx: int, source: str) -> None:
        nonlocal oracle_pairs

        ex_m2iv = hf_dataset[idx]
        target_index = int(ex_m2iv["target_index"])
        demo_indices = [int(i) for i in ex_m2iv["demo_indices"]]

        ds_raw = ds_aokvqa_raw if source == "aokvqa" else ds_csqa_raw

        # Restore raw samples
        target_raw = ds_raw[target_index]
        demos_raw = [ds_raw[i] for i in demo_indices]

        # ----- Teacher 16-shot -----
        teacher_texts: List[str] = []
        teacher_images: List[Any] = []

        for ex_demo in demos_raw:
            q_demo, a_demo = extract_qa_from_raw(ex_demo, source=source)
            demo_text = build_qa_text(q_demo, a_demo, with_vision_prefix=True)
            teacher_texts.append(demo_text)
            teacher_images.append(decode_image_field(ex_demo.get("image", None)))

        q_target, a_target = extract_qa_from_raw(target_raw, source=source)
        target_teacher_text = build_qa_text(q_target, a_target, with_vision_prefix=True)
        teacher_texts.append(target_teacher_text)
        teacher_images.append(decode_image_field(target_raw.get("image", None)))

        teacher_16shot_text = "\n\n".join(teacher_texts)

        with torch.no_grad():
            teacher_inputs = processor(
                text=[teacher_16shot_text],
                images=[teacher_images],
                return_tensors="pt",
                padding=True,
            )
            teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

            teacher_outputs = model(**teacher_inputs)
            teacher_logits_all = teacher_outputs.logits  # [1,S,V]
            teacher_attn_mask = teacher_inputs["attention_mask"]

            # 仅使用答案最后一个 token 的预测分布作为蒸馏目标
            final_logits = get_last_token_logits(teacher_logits_all, teacher_attn_mask)  # [1,V]
            teacher_logits = final_logits  # [1,V]

            T = cfg.TEMPERATURE
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)  # [1,V]

        # Free teacher intermediates
        del teacher_outputs, teacher_logits_all, teacher_inputs, final_logits
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # ----- Student 0-shot baseline inputs & query_features -----
        student_inputs, query_features = build_student_inputs_and_query_features(
            model=model,
            processor=processor,
            q_target=q_target,
            a_target=a_target,
            image_raw=target_raw.get("image", None),
            device=device,
        )  # student_inputs on device, query_features [1,H] float32

        # ----- Grid search over 8 experts -----
        kl_values: List[float] = []

        # Precompute parameter device/dtype for manual v_inject
        try:
            ref_param = next(model.parameters())
            model_device = ref_param.device
            model_dtype = ref_param.dtype
        except StopIteration:
            model_device = device
            model_dtype = torch.float16

        for e_idx in range(8):
            expert_vec = experts_attn[e_idx]  # [H]
            test_v = expert_vec.to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,H]

            # Manually set current_v_inject so that hook uses this vector
            wrapper.current_v_inject = test_v.to(device=model_device, dtype=model_dtype)

            with torch.no_grad():
                student_outputs = model(**student_inputs)
                student_logits_all = student_outputs.logits  # [1,S,V]
                student_attn_mask = student_inputs["attention_mask"]

                student_final_logits = get_last_token_logits(
                    student_logits_all, student_attn_mask
                )  # [1,V]
                T = cfg.TEMPERATURE
                student_logprobs = F.log_softmax(student_final_logits / T, dim=-1)  # [1,V]

                kl = F.kl_div(
                    student_logprobs,
                    teacher_probs,
                    reduction="batchmean",
                )
                kl_values.append(float(kl.detach().cpu().item()))

            # Clear for next expert
            wrapper.current_v_inject = None
            del student_outputs, student_logits_all, student_attn_mask, student_final_logits
            torch.cuda.empty_cache() if device.type == "cuda" else None

        if not kl_values:
            return

        best_expert = int(min(range(8), key=lambda i: kl_values[i]))

        # Oracle one-hot routing target
        oracle_weights = torch.zeros(8, dtype=torch.float32)
        oracle_weights[best_expert] = 1.0

        # Store query_features as [H]
        oracle_pairs.append((query_features[0].detach().cpu(), oracle_weights))

        # Free per-sample student inputs
        del student_inputs, query_features
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Process selected samples for both domains
    print("[INFO][RouterSFT] Phase 1: Constructing oracle routing targets (grid search).")

    for idx in tqdm(aok_selected, desc="[RouterSFT] A-OKVQA Oracle", unit="sample"):
        process_one_index(idx, source="aokvqa")

    for idx in tqdm(csqa_selected, desc="[RouterSFT] CSQA Oracle", unit="sample"):
        process_one_index(idx, source="csqa")

    num_oracles = len(oracle_pairs)
    if num_oracles == 0:
        raise RuntimeError("No oracle pairs were constructed; please check dataset and configuration.")

    print(f"[INFO][RouterSFT] Total oracle pairs constructed: {num_oracles}")

    # ---------------
    # Phase 2: Router SFT (MSE regression to oracle weights)
    # ---------------
    print("[INFO][RouterSFT] Phase 2: Supervised fine-tuning of the router.")

    # Stack queries and targets into tensors
    all_queries = torch.stack([q for (q, _w) in oracle_pairs], dim=0).to(
        device=device, dtype=torch.float32
    )  # [N,H]
    all_targets = torch.stack([w for (_q, w) in oracle_pairs], dim=0).to(
        device=device, dtype=torch.float32
    )  # [N,8]

    # Optimizer only over router parameters (experts are frozen by load_and_assign_experts)
    trainable_params = list(filter(lambda p: p.requires_grad, moicv_layer.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=ROUTER_LR,
        weight_decay=0.0,
    )

    moicv_layer.train()

    for epoch in range(ROUTER_EPOCHS):
        idxs = list(range(num_oracles))
        rng.shuffle(idxs)

        epoch_loss = 0.0
        for i in tqdm(idxs, desc=f"[RouterSFT] Epoch {epoch+1}/{ROUTER_EPOCHS}", unit="sample"):
            q = all_queries[i : i + 1]      # [1,H]
            target = all_targets[i : i + 1] # [1,8]

            v_inject, routing_weights = moicv_layer(q)  # [1,H], [1,8]

            loss = F.mse_loss(routing_weights, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())

        avg_loss = epoch_loss / float(num_oracles)
        print(f"[INFO][RouterSFT] Epoch {epoch+1}/{ROUTER_EPOCHS} - avg MSE loss: {avg_loss:.6f}")

    # Save trained router weights
    ckpt_path = "moicv_router_sft.pth"
    torch.save(moicv_layer.state_dict(), ckpt_path)
    print(f"[INFO][RouterSFT] Router SFT checkpoint saved to: {os.path.abspath(ckpt_path)}")


if __name__ == "__main__":
    main()

