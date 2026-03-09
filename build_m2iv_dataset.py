#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 M²IV 风格蒸馏数据集的预处理脚本
===================================

目标：
  - 对 A-OKVQA 和 CSQA 原始训练集进行特征提取与聚类；
  - 使用 K=2000 的 K-means，将每个簇的质心最近样本作为 Target Query（约 2000 条）；
  - 将当前任务剩余的所有样本组成一个全局 Support Set；
  - 对每个 Target，从全局 Support Set 无放回随机采样 16 条 Demonstrations，并做一次顺序打乱增强；
  - 最终得到 4000 条 A-OKVQA + 4000 条 CSQA，总计 8000 条蒸馏训练样本；
  - 使用 HuggingFace Datasets 保存到 ./m2iv_distill_dataset 目录。

数据格式设计：
  每条样本为一个字典：
    {
      "target": <原始样本字典>,          # 包含 question / image / choices / answer 等字段
      "demos": [<demo_sample_1>, ..., <demo_sample_16>],  # 同簇内采样 16 条样本字典
      "source": "aokvqa" 或 "csqa"
    }

依赖安装（示例）：
    pip install torch torchvision
    pip install transformers datasets scikit-learn tqdm pillow
"""

from __future__ import annotations

import os
import random
import sys
import builtins
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from datasets import Dataset
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


# 在 Windows 控制台等非 UTF-8 环境下，直接 print 中文可能触发 UnicodeEncodeError。
# 这里对内置 print 做一个轻量级包装，避免因编码问题中断脚本执行。
_original_print = builtins.print


def _safe_print(*args: Any, **kwargs: Any) -> None:
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_args = []
        for a in args:
            s = str(a)
            safe_args.append(s.encode(enc, errors="ignore").decode(enc))
        _original_print(*safe_args, **kwargs)


builtins.print = _safe_print


@dataclass
class M2IVConfig:
    # A-OKVQA 训练集 parquet 路径（请按实际环境修改）
    AOKVQA_TRAIN_PARQUET: str = r"D:\mmlm-icl\train-00000-of-00002-c1d24de3bacb5e0c.parquet"

    # CSQA 训练集 parquet 路径
    CSQA_TRAIN_PARQUET: str = r"D:\mmlm-icl\train-00000-of-00001.parquet"

    # 聚类簇数
    N_CLUSTERS_AOKVQA: int = 2000
    N_CLUSTERS_CSQA: int = 2000

    # 每个 Target Query 的 Demonstration 数量
    NUM_DEMOS: int = 16

    # 特征提取模型
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # 输出路径
    OUTPUT_DIR: str = "./m2iv_distill_dataset"

    # 随机种子
    SEED: int = 42


CFG = M2IVConfig()


def get_device() -> torch.device:
    """自动选择 GPU / CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_image_field(image_field: Any) -> PILImage.Image:
    """
    将 A-OKVQA 中的 image 字段转换为 PIL.Image。
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


def extract_features_aokvqa(
    ds: Dataset,
    processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    对 A-OKVQA 训练集提取多模态特征。

    策略：
      - 图像特征：CLIP image_embeds
      - 文本特征：CLIP text_embeds（使用 question 文本）
      - 最终特征：L2 归一化后的 [image_embeds || text_embeds] 拼接向量，维度约 1024。
    返回：
      features: [N, D] 的 NumPy 数组
    """
    n = len(ds)
    print(f"[A-OKVQA] 开始特征提取，总样本数：{n}")

    all_feats: List[np.ndarray] = []

    for start in tqdm(range(0, n, batch_size), desc="A-OKVQA CLIP 特征", unit="batch"):
        end = min(start + batch_size, n)
        batch = ds[start:end]  # HuggingFace Dataset 切片返回的是列名->列表的字典

        images_raw = batch["image"]
        questions = batch["question"]

        images = [decode_image_field(img) for img in images_raw]
        texts = [q if q is not None else "" for q in questions]

        # 图像特征
        img_inputs = processor(images=images, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

        # 文本特征
        txt_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

        with torch.no_grad():
            img_embeds = clip_model.get_image_features(**img_inputs)   # [B, 512]
            txt_embeds = clip_model.get_text_features(**txt_inputs)    # [B, 512]

            # L2 归一化
            img_embeds = F.normalize(img_embeds, p=2, dim=-1)
            txt_embeds = F.normalize(txt_embeds, p=2, dim=-1)

            feats = torch.cat([img_embeds, txt_embeds], dim=-1)        # [B, 1024]

        all_feats.append(feats.cpu().numpy())

    features = np.concatenate(all_feats, axis=0)
    print(f"[A-OKVQA] 特征提取完成，形状：{features.shape}")
    return features


def extract_features_csqa(
    ds: Dataset,
    processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    对 CSQA 训练集提取文本特征。

    策略：
      - 仅使用 question 文本
      - 文本特征：CLIP text_embeds
      - 最终特征：L2 归一化后的 text_embeds，维度约 512。
    返回：
      features: [N, D] 的 NumPy 数组
    """
    n = len(ds)
    print(f"[CSQA] 开始特征提取，总样本数：{n}")

    all_feats: List[np.ndarray] = []

    for start in tqdm(range(0, n, batch_size), desc="CSQA CLIP 特征", unit="batch"):
        end = min(start + batch_size, n)
        batch = ds[start:end]  # dict: 列名 -> 列表

        questions = batch["question"]
        texts = [q if q is not None else "" for q in questions]

        txt_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

        with torch.no_grad():
            txt_embeds = clip_model.get_text_features(**txt_inputs)  # [B, 512]
            txt_embeds = F.normalize(txt_embeds, p=2, dim=-1)

        all_feats.append(txt_embeds.cpu().numpy())

    features = np.concatenate(all_feats, axis=0)
    print(f"[CSQA] 特征提取完成，形状：{features.shape}")
    return features


def run_kmeans(features: np.ndarray, n_clusters: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对给定特征执行 KMeans 聚类，返回：
      - labels: 每个样本的簇编号 [N]
      - centers: 每个簇的中心 [K, D]
    """
    print(f"[KMeans] 开始聚类：N={features.shape[0]}, D={features.shape[1]}, K={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    kmeans.fit(features)
    labels = kmeans.labels_.astype(np.int32)
    centers = kmeans.cluster_centers_.astype(np.float32)
    print("[KMeans] 聚类完成。")
    return labels, centers


def build_cluster_queries_and_demos(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    num_demos: int,
    source_name: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    给定聚类结果，按照重新梳理后的 M²IV 机制构造蒸馏数据：
      1) 对每个簇，选取簇中心最近的 1 个样本作为 Target Query（target_indices）；
      2) 将当前任务剩余的所有样本组成全局 Support Set（support_set_indices）；
      3) 对每个 Target，从全局 Support Set 无放回随机采样 num_demos 条 Demonstrations，
         并额外构造一次顺序打乱的增强版本。

    为了保持 HuggingFace Dataset 的 schema 简洁，我们只存储样本索引，后续训练脚本可以再去原始
    A-OKVQA / CSQA 数据集中根据索引取出具体的图文内容。

    返回：
      records: List[{"target_index": int, "demo_indices": List[int], "source": str}]
    """
    n_clusters = centers.shape[0]
    n_samples = features.shape[0]

    print(f"[{source_name}] 使用簇中心选择 Target Queries...")

    # 1. 为每个簇选取距离簇中心最近的样本，组成 target_indices
    target_indices: List[int] = []
    for c in tqdm(range(n_clusters), desc=f"{source_name} Clusters", unit="cluster"):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            # 允许极少数空簇，直接跳过
            continue

        feats_c = features[idxs]  # [Nc, D]
        centroid = centers[c]  # [D]
        dists = np.linalg.norm(feats_c - centroid[None, :], axis=1)  # [Nc]
        min_pos = int(np.argmin(dists))
        target_global_idx = int(idxs[min_pos])
        target_indices.append(target_global_idx)

    # 2. 构建全局 Support Set：当前任务中除 Targets 外的所有样本
    all_task_indices = np.arange(n_samples, dtype=np.int64)
    target_set = set(int(i) for i in target_indices)
    support_set_indices: List[int] = [
        int(idx) for idx in all_task_indices if int(idx) not in target_set
    ]

    if len(support_set_indices) < num_demos:
        raise ValueError(
            f"[{source_name}] Support Set 大小为 {len(support_set_indices)}，"
            f"小于所需的 Demonstrations 数量 num_demos={num_demos}"
        )

    print(
        f"[{source_name}] 共选出 {len(target_indices)} 条 Target Queries，"
        f"Support Set 大小为 {len(support_set_indices)}。"
    )

    # 3. 为每个 Target 构造 16-shot 上下文及其 Shuffle 增强
    records: List[Dict[str, Any]] = []
    print(f"[{source_name}] 为每个 Target 构造 {num_demos}-shot 上下文与打乱增强...")

    for target_idx in tqdm(target_indices, desc=f"{source_name} Targets", unit="target"):
        # 从全局 Support Set 无放回随机采样 num_demos 条（对单个 target 而言）
        demo_indices = rng.sample(support_set_indices, num_demos)
        demo_indices_list = [int(i) for i in demo_indices]

        # 基础数据：原顺序
        records.append(
            {
                "target_index": int(target_idx),
                "demo_indices": demo_indices_list,
                "source": source_name,
            }
        )

        # Shuffle 数据增强：打乱 demo_indices 顺序
        demos_shuffled = list(demo_indices_list)
        rng.shuffle(demos_shuffled)
        records.append(
            {
                "target_index": int(target_idx),
                "demo_indices": demos_shuffled,
                "source": source_name,
            }
        )

    print(f"[{source_name}] 构造完成，共生成 {len(records)} 条样本。")
    return records


def main() -> None:
    cfg = CFG
    rng = random.Random(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    device = get_device()
    print(f"[INFO] 当前设备：{device}")

    # 1. 加载 CLIP 模型与 Processor
    print(f"[INFO] 加载 CLIP 模型：{cfg.CLIP_MODEL_NAME}")
    processor = CLIPProcessor.from_pretrained(cfg.CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(cfg.CLIP_MODEL_NAME)
    clip_model.to(device)
    clip_model.eval()

    # 2. 加载 A-OKVQA 与 CSQA 训练集
    if not os.path.exists(cfg.AOKVQA_TRAIN_PARQUET):
        raise FileNotFoundError(f"A-OKVQA 训练 parquet 不存在：{cfg.AOKVQA_TRAIN_PARQUET}")
    if not os.path.exists(cfg.CSQA_TRAIN_PARQUET):
        raise FileNotFoundError(f"CSQA 训练 parquet 不存在：{cfg.CSQA_TRAIN_PARQUET}")

    print(f"[INFO] 从 parquet 加载 A-OKVQA 训练集：{cfg.AOKVQA_TRAIN_PARQUET}")
    ds_aokvqa = Dataset.from_parquet(cfg.AOKVQA_TRAIN_PARQUET)
    print(f"[INFO] A-OKVQA 训练样本数：{len(ds_aokvqa)}")

    print(f"[INFO] 从 parquet 加载 CSQA 训练集：{cfg.CSQA_TRAIN_PARQUET}")
    ds_csqa = Dataset.from_parquet(cfg.CSQA_TRAIN_PARQUET)
    print(f"[INFO] CSQA 训练样本数：{len(ds_csqa)}")

    # 3. 特征提取
    feats_aokvqa = extract_features_aokvqa(ds_aokvqa, processor, clip_model, device)
    feats_csqa = extract_features_csqa(ds_csqa, processor, clip_model, device)

    # 4. KMeans 聚类
    labels_aokvqa, centers_aokvqa = run_kmeans(
        feats_aokvqa,
        n_clusters=cfg.N_CLUSTERS_AOKVQA,
        seed=cfg.SEED,
    )
    labels_csqa, centers_csqa = run_kmeans(
        feats_csqa,
        n_clusters=cfg.N_CLUSTERS_CSQA,
        seed=cfg.SEED,
    )

    # 5. 构建 Target Query + Demonstrations（以及顺序打乱增强）
    records_aokvqa = build_cluster_queries_and_demos(
        features=feats_aokvqa,
        labels=labels_aokvqa,
        centers=centers_aokvqa,
        num_demos=cfg.NUM_DEMOS,
        source_name="aokvqa",
        rng=rng,
    )
    records_csqa = build_cluster_queries_and_demos(
        features=feats_csqa,
        labels=labels_csqa,
        centers=centers_csqa,
        num_demos=cfg.NUM_DEMOS,
        source_name="csqa",
        rng=rng,
    )

    # 6. 合并两个任务的样本，并整体打乱
    all_records = records_aokvqa + records_csqa
    print(f"[ALL] 合并后总样本数：{len(all_records)}")
    rng.shuffle(all_records)

    # 7. 构建 HuggingFace Dataset 并保存
    print(f"[INFO] 正在构建 HuggingFace Dataset 并保存到：{cfg.OUTPUT_DIR}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    final_ds = Dataset.from_list(all_records)
    final_ds.save_to_disk(cfg.OUTPUT_DIR)

    print("[INFO] M²IV 风格蒸馏数据集构建完成！")
    print(f"[INFO] 可使用以下代码加载：")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk(r\"{cfg.OUTPUT_DIR}\")")


if __name__ == "__main__":
    main()


