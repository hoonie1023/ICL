#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoICV 自蒸馏版 Benchmark 评测脚本
================================

说明：
- 不修改原始 `eval_benchmarks_split.py`；
- 直接复用其评测逻辑，只是切换为自蒸馏训练得到的 MoICV 权重。
"""

from __future__ import annotations

from eval_benchmarks_split import CFG, EvalConfig, main as eval_main


def main() -> None:
    """
    使用自蒸馏训练得到的 MoICV 权重进行 Benchmark 评测。
    """
    # ===== 根据你的实际训练情况，选择合适的权重路径 =====
    # 1）如果用原始 `train_distill.py`（1 epoch）：
    #    weights_path = "/root/autodl-tmp/mmlm-icl/outputs_distill_gpu/moicv_distill_trained_gpu.bin"
    #
    # 2）如果用 `train_distill_10ep.py`（10 epoch）：
    #    建议使用最终权重：
    #    weights_path = "/root/autodl-tmp/mmlm-icl/outputs_distill_gpu_10ep/moicv_distill_trained_gpu.bin"
    #
    # 默认先指向 10 epoch 的最终权重；如有需要，你可以直接改下面这一行。
    weights_path = "/root/autodl-tmp/mmlm-icl/outputs_distill_gpu_10ep/moicv_distill_trained_gpu.bin"

    # 同时修改配置类和全局 CFG，确保内部统一读取新的路径
    EvalConfig.MOICV_WEIGHTS_PATH = weights_path
    CFG.MOICV_WEIGHTS_PATH = weights_path

    print(f"[Eval-Distill] 使用自蒸馏 MoICV 权重：{weights_path}")

    # 直接调用原始评测入口
    eval_main()


if __name__ == "__main__":
    main()


