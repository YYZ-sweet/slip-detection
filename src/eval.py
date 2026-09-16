"""
eval.py — 独立评估脚本
======================

为什么单独拆一个 eval.py？
  训练时 eval 是为了"挑最优权重"（用验证集）；
  论文里报指标要的是"可复现的客观评估"（用测试集，且不参与任何调参）。
  把评估独立出来，别人拿到你的 best.pt 就能一键复现论文里的数字。

用法：
  # 评估某个实验
  python src/eval.py --run experiments/baseline_GRU

  # 指定阈值（不用 checkpoint 里存的）
  python src/eval.py --run experiments/baseline_GRU --threshold 0.35

  # 在另一个 processed_data 上评估（跨物体泛化实验用）
  python src/eval.py --run experiments/baseline_GRU --processed_dir ./processed_data_brush

  # 同时输出每个物体的分项指标（data_dir 里有几个物体就报几个）
  python src/eval.py --run experiments/baseline_GRU --per_object
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_npz_splits
from train import (build_model, resolve_device, predict_proba, compute_metrics)


def parse_args():
    p = argparse.ArgumentParser(description="滑移检测评估脚本")
    p.add_argument("--run", type=str, required=True,
                   help="实验目录（含 best.pt），如 experiments/baseline_GRU")
    p.add_argument("--processed_dir", type=str, default=None,
                   help="覆盖数据目录（默认用 checkpoint 里记录的配置）")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"], help="评估哪个划分")
    p.add_argument("--threshold", type=float, default=None,
                   help="决策阈值（默认用 checkpoint 里存的最优阈值）")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--out", type=str, default=None,
                   help="结果 JSON 输出路径（默认写到 run 目录下 eval_<split>.json）")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = os.path.join(args.run, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    threshold = args.threshold if args.threshold is not None else ckpt.get("threshold", 0.5)

    device = resolve_device(cfg["train"]["device"])
    print(f"载入: {ckpt_path}")
    print(f"  实验      : {cfg['experiment']['name']}")
    print(f"  模型      : {cfg['model']['type']}")
    print(f"  频带模式  : {cfg['data']['freq']['mode']}")
    print(f"  阈值      : {threshold:.4f}")
    print(f"  设备      : {device}")

    # ---------- 数据 ----------
    processed_dir = args.processed_dir or cfg["data"]["processed_dir"]
    print(f"  数据目录  : {processed_dir}")

    splits, meta = load_npz_splits(processed_dir)
    X, y = splits[args.split]
    print(f"  评估划分  : {args.split} ({len(y)} 样本, Slip%={y.mean():.1%})")

    # 手工频带特征（与训练时保持一致）
    f_cfg = cfg["data"]["freq"]
    if f_cfg.get("enabled") and f_cfg.get("mode") == "manual":
        from data import add_freq_features
        X, _ = add_freq_features(X, mode="manual",
                                 edges_hz=f_cfg["edges_hz"],
                                 fs=f_cfg.get("fs", 180.0))
        print(f"  频带特征已拼接 → {X.shape[-1]} 维")

    # ---------- 模型 ----------
    input_size = X.shape[-1]
    model = build_model(cfg, input_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---------- 推理 ----------
    p = predict_proba(model, X, device, args.batch_size)
    y_pred = (p >= threshold).astype(int)
    metrics = compute_metrics(y, y_pred)

    print(f"\n{'='*64}")
    print(f"评估结果（{args.split} set, threshold={threshold:.3f}）")
    print(f"{'='*64}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  MCC      : {metrics['mcc']:.4f}")
    print(f"  漏检率 FNR        : {metrics['fnr']:.4f}")
    print(f"  Slip 类 Precision : {metrics['slip_precision']:.4f}")
    print(f"  Slip 类 Recall    : {metrics['slip_recall']:.4f}")
    print(f"  混淆矩阵:\n{np.array(metrics['confusion_matrix'])}")

    # ---------- 落盘 ----------
    metrics.update({
        "run_name": cfg["experiment"]["name"],
        "split": args.split,
        "processed_dir": processed_dir,
        "threshold": float(threshold),
    })
    out_path = args.out or os.path.join(args.run, f"eval_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
