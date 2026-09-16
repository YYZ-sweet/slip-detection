"""
rebuild_summary.py — 从各实验目录重建 experiments/summary.csv
=============================================================

用途：summary.csv 有时会丢（误删、切机器、clone 新仓库），
      但每个实验目录下的 metrics.json + config.yaml 都还在。
      这个脚本据此重建汇总表，不用重跑实验。

用法：
  python src/rebuild_summary.py
  python src/rebuild_summary.py --experiments_dir ./experiments
"""

import os
import csv
import json
import argparse

import yaml

FIELDS = ["run_name", "model", "freq_mode", "loss", "class_weight",
          "lr", "epochs_ran", "threshold", "acc", "f1", "mcc", "fnr",
          "params", "time_sec"]


def count_epochs(history_path):
    """history.csv 的数据行数 = 实际训练轮数。"""
    if not os.path.exists(history_path):
        return 0
    with open(history_path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def collect(experiments_dir):
    rows = []
    for name in sorted(os.listdir(experiments_dir)):
        run_dir = os.path.join(experiments_dir, name)
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        cfg = {}
        cfg_path = os.path.join(run_dir, "config.yaml")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        train_cfg = cfg.get("train", {})
        rows.append({
            "run_name": m.get("run_name", name),
            "model": m.get("model_type", ""),
            "freq_mode": m.get("freq_mode", ""),
            "loss": train_cfg.get("loss", "ce"),
            "class_weight": train_cfg.get("class_weight", False),
            "lr": train_cfg.get("lr", ""),
            "epochs_ran": count_epochs(os.path.join(run_dir, "history.csv")),
            "threshold": round(float(m.get("threshold", 0.0)), 4),
            "acc": round(float(m.get("accuracy", 0.0)), 4),
            "f1": round(float(m.get("f1", 0.0)), 4),
            "mcc": round(float(m.get("mcc", 0.0)), 4),
            "fnr": round(float(m.get("fnr", 0.0)), 4),
            "params": int(m.get("n_params", 0)),
            "time_sec": round(float(m.get("train_time_sec", 0.0)), 1),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description="重建 experiments/summary.csv")
    p.add_argument("--experiments_dir", type=str, default="./experiments")
    args = p.parse_args()

    rows = collect(args.experiments_dir)
    if not rows:
        print(f"在 {args.experiments_dir} 下没找到任何含 metrics.json 的实验目录")
        return

    out_path = os.path.join(args.experiments_dir, "summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"已重建 {out_path}，共 {len(rows)} 行\n")
    header = f"{'run_name':<20} {'freq':<10} {'acc':>7} {'f1':>7} {'mcc':>7} {'fnr':>7} {'params':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run_name']:<20} {r['freq_mode']:<10} "
              f"{r['acc']:>7.4f} {r['f1']:>7.4f} {r['mcc']:>7.4f} "
              f"{r['fnr']:>7.4f} {r['params']:>8}")


if __name__ == "__main__":
    main()
