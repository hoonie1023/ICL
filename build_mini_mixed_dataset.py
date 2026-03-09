#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态小型混合测试集预处理脚本

功能概述：
1. 使用 datasets.Dataset.from_parquet 加载本地 A-OKVQA 和 CSQA 的 .parquet 文件
2. 从每个数据集中随机抽取 SAMPLE_SIZE 条数据
3. 将两者统一为 {"image": PIL.Image, "query": str, "label": str} 格式
   - A-OKVQA：使用真实图片
   - CSQA：生成 224x224 黑色占位图
4. 合并、打乱，保存为可用 load_from_disk 直接加载的本地数据集

依赖（先安装）：
    pip install datasets pillow pyarrow numpy
"""

import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from datasets import Dataset, concatenate_datasets, load_from_disk

# ==================== 配置区域（可根据实际情况修改） ====================

# 1. A-OKVQA parquet 文件路径（请按需修改）
AOKVQA_PARQUET_PATH = r"D:\mmlm-icl\train-00000-of-00002-c1d24de3bacb5e0c.parquet"

# 2. CSQA parquet 文件路径（请按需修改）
CSQA_PARQUET_PATH = r"D:\mmlm-icl\train-00000-of-00001.parquet"

# 3. 输出数据集保存目录（将创建并覆盖同名目录）
OUTPUT_DIR = r"D:\mmlm-icl\mini_mixed_dataset"

# 4. 每个数据集采样数量
SAMPLE_SIZE = 500

# 5. 随机种子（保证可复现）
SEED = 42

# 6. 黑色占位图的尺寸
DUMMY_IMAGE_SIZE = (224, 224)


def set_random_seed(seed: int) -> None:
    """设置 Python 内建 random 与 numpy 的随机种子，保证采样可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def load_and_sample_parquet(parquet_path: str, sample_size: int, seed: int) -> Dataset:
    """
    从 parquet 加载为 Dataset，并随机抽样 sample_size 条数据。
    如果数据量小于 sample_size，则全部保留并打乱。
    """
    print(f"\n[加载] 从 parquet 加载数据集：{parquet_path}")
    ds = Dataset.from_parquet(parquet_path)
    print(f"[信息] 原始样本数：{len(ds)}")

    if len(ds) <= sample_size:
        print(f"[采样] 数据量 ({len(ds)}) 小于等于采样数量 ({sample_size})，将全部保留并打乱。")
        return ds.shuffle(seed=seed)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=sample_size, replace=False)
    sampled = ds.select(indices)
    print(f"[采样] 实际采样条数：{len(sampled)}")
    return sampled


def ensure_pil_image(image_obj):
    """
    将各种可能形式的图像字段，统一转换为 PIL.Image.Image 对象。

    可能形式：
      - 已经是 PIL.Image.Image
      - 字节（bytes 或 {"bytes": ...}）
      - 路径字符串（本地路径）
      - 其他异常情况则返回黑色占位图
    """
    if isinstance(image_obj, PILImage.Image):
        return image_obj

    if isinstance(image_obj, dict) and "bytes" in image_obj:
        try:
            return PILImage.open(BytesIO(image_obj["bytes"])).convert("RGB")
        except Exception:
            pass

    if isinstance(image_obj, (bytes, bytearray)):
        try:
            return PILImage.open(BytesIO(image_obj)).convert("RGB")
        except Exception:
            pass

    if isinstance(image_obj, str):
        try:
            return PILImage.open(image_obj).convert("RGB")
        except Exception:
            pass

    return PILImage.new("RGB", DUMMY_IMAGE_SIZE, color="black")


def make_dummy_black_image(size=(224, 224)) -> PILImage.Image:
    """创建一张给定尺寸的全黑占位图。"""
    return PILImage.new("RGB", size, color="black")


def build_aokvqa_query_and_label(example):
    """
    根据 A-OKVQA 样本构造 query 和 label。

    兼容常见字段：
      - question: 问题文本
      - choices: 选项（通常是 list[str]）
      - correct_answer / answer / multiple_choice_answer
      - correct_choice_idx + choices
    """
    question = example.get("question", "")

    raw_choices = example.get("choices", [])
    choices_str = ""
    if isinstance(raw_choices, list):
        labeled = []
        for i, c in enumerate(raw_choices):
            label_char = chr(ord("A") + i)
            labeled.append(f"({label_char}) {c}")
        choices_str = " | ".join(labeled)
    elif isinstance(raw_choices, dict):
        choices_str = " | ".join(f"{k}: {v}" for k, v in raw_choices.items())
    else:
        choices_str = str(raw_choices)

    query = f"{question} Choices: {choices_str}"

    answer = None
    if "correct_answer" in example and example["correct_answer"] is not None:
        answer = example["correct_answer"]
    elif "answer" in example and example["answer"] is not None:
        answer = example["answer"]
    elif "multiple_choice_answer" in example and example["multiple_choice_answer"] is not None:
        answer = example["multiple_choice_answer"]
    elif "correct_choice_idx" in example and isinstance(raw_choices, list):
        try:
            idx = int(example["correct_choice_idx"])
            if 0 <= idx < len(raw_choices):
                answer = raw_choices[idx]
        except Exception:
            pass

    if answer is None:
        answer = "unknown"

    label = str(answer)
    return query, label


def process_aokvqa_example(example):
    """将 A-OKVQA 的单条样本转换为统一格式。"""
    try:
        image_field = example.get("image", None)
        image = ensure_pil_image(image_field)
        query, label = build_aokvqa_query_and_label(example)
        return {
            "image": image,
            "query": query,
            "label": label,
            "dataset_source": "aokvqa",
        }
    except Exception as e:
        print(f"[警告] 处理 A-OKVQA 样本出错，将使用占位数据。错误信息：{e}")
        return {
            "image": make_dummy_black_image(DUMMY_IMAGE_SIZE),
            "query": "error",
            "label": "error",
            "dataset_source": "aokvqa",
        }


def build_csqa_query_and_label(example):
    """
    根据 CSQA 样本构造 query 和 label。

    兼容典型 CommonsenseQA 格式：
      - question: 问题文本
      - choices: 可能是 dict，如 {"label": [...], "text": [...]} 或其他形式
      - answer / answerKey: 正确答案（字母标签或文本）
    """
    question = example.get("question", "")

    raw_choices = example.get("choices", {})
    choices_display = []
    answer_text = None

    if isinstance(raw_choices, dict) and "label" in raw_choices and "text" in raw_choices:
        labels = raw_choices.get("label", [])
        texts = raw_choices.get("text", [])
        for lab, txt in zip(labels, texts):
            choices_display.append(f"({lab}) {txt}")

        answer_key = example.get("answerKey", None)
        if answer_key is not None and answer_key in labels:
            idx = labels.index(answer_key)
            if 0 <= idx < len(texts):
                answer_text = texts[idx]
    elif isinstance(raw_choices, dict):
        for k, v in raw_choices.items():
            choices_display.append(f"{k}: {v}")
    elif isinstance(raw_choices, list):
        for i, c in enumerate(raw_choices):
            label_char = chr(ord("A") + i)
            choices_display.append(f"({label_char}) {c}")
    else:
        if raw_choices:
            choices_display.append(str(raw_choices))

    choices_str = " | ".join(choices_display)
    query = f"{question} Choices: {choices_str}"

    if answer_text is not None:
        label = str(answer_text)
    else:
        answer = None
        if "answer" in example and example["answer"] is not None:
            answer = example["answer"]
        elif "answerKey" in example and example["answerKey"] is not None:
            answer = example["answerKey"]
        elif "label" in example and example["label"] is not None:
            answer = example["label"]

        if answer is None:
            answer = "unknown"

        label = str(answer)

    return query, label


def process_csqa_example(example):
    """将 CSQA 的单条样本转换为统一格式。"""
    try:
        image = make_dummy_black_image(DUMMY_IMAGE_SIZE)
        query, label = build_csqa_query_and_label(example)
        return {
            "image": image,
            "query": query,
            "label": label,
            "dataset_source": "csqa",
        }
    except Exception as e:
        print(f"[警告] 处理 CSQA 样本出错，将使用占位数据。错误信息：{e}")
        return {
            "image": make_dummy_black_image(DUMMY_IMAGE_SIZE),
            "query": "error",
            "label": "error",
            "dataset_source": "csqa",
        }


def main() -> None:
    print("=" * 70)
    print("Step 0: 设置随机种子")
    print("=" * 70)
    set_random_seed(SEED)

    # Step 1: 加载与采样
    print("\n" + "=" * 70)
    print("Step 1: 加载 A-OKVQA 与 CSQA，并随机采样")
    print("=" * 70)

    aokvqa_sampled = load_and_sample_parquet(AOKVQA_PARQUET_PATH, SAMPLE_SIZE, SEED)
    csqa_sampled = load_and_sample_parquet(CSQA_PARQUET_PATH, SAMPLE_SIZE, SEED)

    print("\n[A-OKVQA] 样本字段：", aokvqa_sampled[0].keys())
    print("[CSQA] 样本字段：", csqa_sampled[0].keys())

    # Step 2: 格式对齐
    print("\n" + "=" * 70)
    print("Step 2: 将两个数据集统一为 {image, query, label} 格式")
    print("=" * 70)

    print("[处理] A-OKVQA 样本 -> 统一格式")
    aokvqa_processed = aokvqa_sampled.map(
        process_aokvqa_example,
        remove_columns=aokvqa_sampled.column_names,
    )
    print(f"[完成] A-OKVQA 处理后样本数：{len(aokvqa_processed)}")

    print("\n[处理] CSQA 样本 -> 统一格式")
    csqa_processed = csqa_sampled.map(
        process_csqa_example,
        remove_columns=csqa_sampled.column_names,
    )
    print(f"[完成] CSQA 处理后样本数：{len(csqa_processed)}")

    print("\n[检查] 处理后的 A-OKVQA 示例：")
    sa = aokvqa_processed[0]
    print("  - image 类型：", type(sa["image"]))
    print("  - image 尺寸：", getattr(sa["image"], "size", None))
    print("  - query 示例：", sa["query"][:100], "...")
    print("  - label：", sa["label"])

    print("\n[检查] 处理后的 CSQA 示例：")
    sc = csqa_processed[0]
    print("  - image 类型：", type(sc["image"]))
    print("  - image 尺寸：", getattr(sc["image"], "size", None))
    print("  - query 示例：", sc["query"][:100], "...")
    print("  - label：", sc["label"])

    # Step 3: 混合、打乱与保存
    print("\n" + "=" * 70)
    print("Step 3: 混合两个数据集，打乱顺序，并保存到本地")
    print("=" * 70)

    print("[合并] A-OKVQA + CSQA")
    mixed = concatenate_datasets([aokvqa_processed, csqa_processed])
    print(f"[信息] 合并后总样本数：{len(mixed)}")

    print("[打乱] 打乱混合数据集顺序")
    mixed = mixed.shuffle(seed=SEED)

    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        print(f"[提示] 输出目录已存在，将覆盖：{OUTPUT_DIR}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[保存] 保存混合数据集到磁盘：{OUTPUT_DIR}")
    mixed.save_to_disk(OUTPUT_DIR)
    print("[完成] save_to_disk 已完成。")

    # 可选：验证 load_from_disk
    print("\n[验证] 使用 load_from_disk 重新加载数据集（可选）")
    try:
        reloaded = load_from_disk(OUTPUT_DIR)
        print("  - 重新加载样本数：", len(reloaded))
        print("  - 字段名：", reloaded.column_names)

        a_count = sum(1 for x in reloaded if x.get("dataset_source") == "aokvqa")
        c_count = sum(1 for x in reloaded if x.get("dataset_source") == "csqa")
        print("  - A-OKVQA 样本数：", a_count)
        print("  - CSQA 样本数：", c_count)
    except Exception as e:
        print("[警告] 验证加载失败，但数据本身已保存。错误信息：", e)

    print("\n" + "=" * 70)
    print("全部预处理步骤完成！")
    print("=" * 70)
    print("你可以后续这样加载数据集：")
    print("    from datasets import load_from_disk")
    print(f"    dataset = load_from_disk(r\"{OUTPUT_DIR}\")")


if __name__ == "__main__":
    main()


