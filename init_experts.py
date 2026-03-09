#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV 专家向量冷启动初始化脚本
================================

功能概述：
1. 从本地 Hugging Face Dataset (`./mini_mixed_dataset`) 加载 1000 条数据
2. 使用文本模型 Sentence-BERT 提取 query 文本特征：[1000, 384]
3. 使用 CLIP 提取视觉特征（仅对 A-OKVQA 样本）：约 [500, 512]
4. 分别对文本 / 视觉特征做 K-means 聚类 (K=4)，获得簇中心
5. 使用随机线性投影将簇中心映射到大模型 hidden 维度 (TARGET_ATTN_DIM / TARGET_FFN_DIM)
6. 将初始化好的 4 个专家张量保存到 `expert_init_tensors.pt`

依赖安装（请先执行）：
    pip install torch torchvision
    pip install datasets
    pip install sentence-transformers
    pip install transformers
    pip install scikit-learn
    pip install tqdm
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPProcessor


# ===========================
# 全局配置（可按需修改）
# ===========================

# 本地数据集路径（默认当前目录下的 mini_mixed_dataset）
DATASET_PATH = "./mini_mixed_dataset"

# 目标高维空间维度（通常与大模型 hidden_size 对齐）
TARGET_ATTN_DIM = 896
TARGET_FFN_DIM = 896

# 批大小（用于特征提取阶段的 mini-batch 处理）
TEXT_BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 32


def get_device() -> torch.device:
    """自动选择可用设备：优先 CUDA，其次 MPS（Apple），最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(dataset_path: str):
    """从磁盘加载 Hugging Face Dataset。"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在：{dataset_path}")
    print(f"[INFO] 正在从磁盘加载数据集：{dataset_path}")
    ds = load_from_disk(dataset_path)
    print(f"[INFO] 数据集加载完成，样本数：{len(ds)}，字段：{ds.column_names}")
    return ds


def load_text_model(device: torch.device) -> SentenceTransformer:
    """
    加载 Sentence-BERT 文本特征提取模型：
        sentence-transformers/all-MiniLM-L6-v2
    输出维度固定为 384。
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[INFO] 正在加载文本模型：{model_name}")
    text_model = SentenceTransformer(model_name, device=str(device))
    print("[INFO] 文本模型加载完成。")
    return text_model


def load_clip_model(device: torch.device):
    """
    加载 CLIP 模型及 Processor：
        openai/clip-vit-base-patch32

    说明：
        为了获得 512 维图像特征，这里使用 CLIPVisionModelWithProjection，
        直接读取 image_embeds（已经过投影，维度为 512）。
    """
    model_name = "openai/clip-vit-base-patch32"
    print(f"[INFO] 正在加载 CLIP 模型与 Processor：{model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    clip_model.to(device)
    clip_model.eval()
    print("[INFO] CLIP 模型加载完成。")
    return processor, clip_model


def extract_text_features(
    ds,
    text_model: SentenceTransformer,
    batch_size: int = TEXT_BATCH_SIZE,
) -> torch.Tensor:
    """
    提取所有样本的 query 文本特征。

    参数：
        ds: Hugging Face Dataset，需包含 "query" 字段
        text_model: SentenceTransformer 模型
        batch_size: 文本特征提取的批大小

    返回：
        text_feats: [N, 384] 的 torch.Tensor
    """
    if "query" not in ds.column_names:
        raise KeyError("数据集中缺少 'query' 字段，无法提取文本特征。")

    all_queries: List[str] = ds["query"]
    num_samples = len(all_queries)
    print(f"[INFO] 开始提取文本特征，共 {num_samples} 条样本。")

    all_feats: List[torch.Tensor] = []
    for start in tqdm(range(0, num_samples, batch_size), desc="文本特征提取", unit="batch"):
        end = min(start + batch_size, num_samples)
        batch_queries = all_queries[start:end]

        # SentenceTransformer 自带批处理与 no_grad
        batch_emb = text_model.encode(
            batch_queries,
            convert_to_tensor=True,
            show_progress_bar=False,
        )  # [B, 384]
        all_feats.append(batch_emb.cpu())

    text_feats = torch.cat(all_feats, dim=0)  # [N, 384]
    print(f"[INFO] 文本特征提取完成，形状：{tuple(text_feats.shape)}")
    return text_feats


def extract_image_features(
    ds,
    processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
    batch_size: int = IMAGE_BATCH_SIZE,
) -> torch.Tensor:
    """
    提取视觉特征，仅对 dataset_source == 'aokvqa' 的样本进行处理。

    参数：
        ds: Hugging Face Dataset，需包含 "image" 与 "dataset_source"
        processor: CLIPProcessor
        clip_model: CLIPModel
        device: 计算设备
        batch_size: 图像特征提取的批大小

    返回：
        image_feats: [M, 512] 的 torch.Tensor（M 为 A-OKVQA 样本数）
    """
    required_cols = {"image", "dataset_source"}
    if not required_cols.issubset(set(ds.column_names)):
        raise KeyError(f"数据集中缺少字段：{required_cols - set(ds.column_names)}")

    print("[INFO] 正在筛选 dataset_source == 'aokvqa' 的样本用于视觉特征提取。")
    # 使用 filter 得到 A-OKVQA 子集
    ds_aokvqa = ds.filter(lambda ex: ex.get("dataset_source", "") == "aokvqa")
    num_samples = len(ds_aokvqa)
    if num_samples == 0:
        raise RuntimeError("筛选后没有任何 'aokvqa' 样本，无法提取视觉特征。")

    print(f"[INFO] A-OKVQA 子集样本数：{num_samples}")

    all_images = ds_aokvqa["image"]  # 这里应为 PIL.Image 对象列表

    all_feats: List[torch.Tensor] = []
    print("[INFO] 开始提取视觉特征（CLIP 图像嵌入）。")
    for start in tqdm(range(0, num_samples, batch_size), desc="视觉特征提取", unit="batch"):
        end = min(start + batch_size, num_samples)
        batch_images = all_images[start:end]

        # 通过 Processor 做预处理
        inputs = processor(
            images=batch_images,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = clip_model(pixel_values=pixel_values)
            # 使用 image_embeds 作为 512 维视觉特征
            # 形状：[B, 512]
            image_embeds = outputs.image_embeds

        all_feats.append(image_embeds.cpu())

    image_feats = torch.cat(all_feats, dim=0)
    print(f"[INFO] 视觉特征提取完成，形状：{tuple(image_feats.shape)}")
    return image_feats


def run_kmeans(features: torch.Tensor, n_clusters: int = 4, random_state: int = 42) -> torch.Tensor:
    """
    对给定特征执行 K-means 聚类，并返回簇中心。

    参数：
        features: [N, D] 的特征张量
        n_clusters: 聚类簇数
        random_state: 随机种子，保证可复现

    返回：
        centroids: [n_clusters, D] 的 torch.Tensor
    """
    if features.dim() != 2:
        raise ValueError(f"features 必须是二维张量 [N, D]，当前为 {features.shape}")

    print(f"[INFO] 开始 K-means 聚类，簇数 = {n_clusters}，特征维度 = {features.size(1)}")
    # 转为 numpy 以便 sklearn 使用
    feats_np = features.numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(feats_np)

    centroids_np = kmeans.cluster_centers_  # [K, D]
    centroids = torch.from_numpy(centroids_np).float()
    print(f"[INFO] K-means 聚类完成，簇中心形状：{tuple(centroids.shape)}")
    return centroids


def project_centroids_to_experts(
    text_centroids: torch.Tensor,
    vis_centroids: torch.Tensor,
) -> dict:
    """
    使用随机线性映射将文本 / 视觉簇中心投影到目标高维空间，生成 4 组专家张量。

    输入：
        text_centroids: [4, 384]
        vis_centroids:  [4, 512]

    输出字典：
        {
            "E_attn_vis":  [4, TARGET_ATTN_DIM],
            "E_ffn_vis":   [4, TARGET_FFN_DIM],
            "E_attn_text": [4, TARGET_ATTN_DIM],
            "E_ffn_text":  [4, TARGET_FFN_DIM],
        }
    """
    if text_centroids.size(0) != 4 or vis_centroids.size(0) != 4:
        raise ValueError(
            f"text_centroids / vis_centroids 的第 0 维必须为 4，"
            f"当前为 text={text_centroids.shape}, vis={vis_centroids.shape}"
        )
    if text_centroids.size(1) != 384:
        print(
            f"[WARN] 文本特征维度为 {text_centroids.size(1)}，"
            "理论上 all-MiniLM-L6-v2 输出应为 384，请确认模型与特征维度一致。"
        )
    if vis_centroids.size(1) != 512:
        print(
            f"[WARN] 视觉特征维度为 {vis_centroids.size(1)}，"
            "理论上 CLIP 图像嵌入为 512，请确认模型与特征维度一致。"
        )

    # 定义 4 个随机线性映射层（不带 bias）
    proj_vis_attn = nn.Linear(vis_centroids.size(1), TARGET_ATTN_DIM, bias=False)
    proj_vis_ffn = nn.Linear(vis_centroids.size(1), TARGET_FFN_DIM, bias=False)
    proj_text_attn = nn.Linear(text_centroids.size(1), TARGET_ATTN_DIM, bias=False)
    proj_text_ffn = nn.Linear(text_centroids.size(1), TARGET_FFN_DIM, bias=False)

    # 将线性层暂时放在 CPU 上即可（输入数据在 CPU）
    with torch.no_grad():
        E_attn_vis = proj_vis_attn(vis_centroids)   # [4, TARGET_ATTN_DIM]
        E_ffn_vis = proj_vis_ffn(vis_centroids)     # [4, TARGET_FFN_DIM]
        E_attn_text = proj_text_attn(text_centroids)  # [4, TARGET_ATTN_DIM]
        E_ffn_text = proj_text_ffn(text_centroids)    # [4, TARGET_FFN_DIM]

    print("[INFO] 随机高维投影完成，各专家张量形状：")
    print(f"  - E_attn_vis:  {tuple(E_attn_vis.shape)}")
    print(f"  - E_ffn_vis:   {tuple(E_ffn_vis.shape)}")
    print(f"  - E_attn_text: {tuple(E_attn_text.shape)}")
    print(f"  - E_ffn_text:  {tuple(E_ffn_text.shape)}")

    return {
        "E_attn_vis": E_attn_vis,
        "E_ffn_vis": E_ffn_vis,
        "E_attn_text": E_attn_text,
        "E_ffn_text": E_ffn_text,
    }


def main() -> None:
    # 1. 设备选择
    device = get_device()
    print(f"[INFO] 当前使用设备：{device}")

    # 2. 加载数据集
    ds = load_dataset(DATASET_PATH)

    # 3. 加载特征提取模型
    text_model = load_text_model(device)
    clip_processor, clip_model = load_clip_model(device)

    # 4. 提取文本特征
    text_features = extract_text_features(ds, text_model, batch_size=TEXT_BATCH_SIZE)

    # 5. 提取视觉特征（仅 A-OKVQA）
    image_features = extract_image_features(
        ds,
        processor=clip_processor,
        clip_model=clip_model,
        device=device,
        batch_size=IMAGE_BATCH_SIZE,
    )

    # 6. 对文本/视觉特征做 K-means 聚类，得到簇中心
    text_centroids = run_kmeans(text_features, n_clusters=4, random_state=42)
    vis_centroids = run_kmeans(image_features, n_clusters=4, random_state=42)

    # 7. 随机高维投影，生成专家向量
    expert_tensors = project_centroids_to_experts(
        text_centroids=text_centroids,
        vis_centroids=vis_centroids,
    )

    # 8. 保存到本地文件
    save_path = "expert_init_tensors.pt"
    torch.save(expert_tensors, save_path)
    print(f"[INFO] 专家初始化张量已保存到：{os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()


