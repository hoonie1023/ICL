#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV + Qwen2.5-VL-7B 自蒸馏训练脚本（10 Epoch 版本）
====================================================

说明：
- 不修改原始 `train_distill.py` 的逻辑；
- 通过覆盖默认配置，将训练轮数改为 10，并单独使用新的输出目录。
"""

from __future__ import annotations

from train_distill import CFG, DistillConfig, main as distill_main


def main() -> None:
    """
    启动 10 个 Epoch 的自蒸馏训练。
    """
    # 修改全局配置（不改源码，只改默认值）
    DistillConfig.NUM_EPOCHS = 10
    DistillConfig.OUTPUT_DIR = "./outputs_distill_gpu_10ep"

    # 同时修改已经实例化的全局 CFG，确保 main() 里读取到的是新值
    CFG.NUM_EPOCHS = 10
    CFG.OUTPUT_DIR = "./outputs_distill_gpu_10ep"

    distill_main()


if __name__ == "__main__":
    main()


