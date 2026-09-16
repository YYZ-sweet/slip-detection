"""
train.py — 统一训练入口（从你暑假 train.py 改造而来）
=====================================================

相比原版的改进：
  1. 支持 YAML 配置 + 命令行覆盖（不再改源码跑实验）
  2. 支持 3 种模型：GRU / LSTM / Transformer
  3. 支持 3 种频带模式：none / manual / learnable
  4. 支持 3 种类别不平衡处理：SMOTE(预处理阶段) / class_weight / focal_loss
  5. 每次实验自动落盘到 experiments/<name>/：
     config.yaml（配置快照）+ history.csv（训练曲线）+ metrics.json（测试指标）+ best.pt（权重）
  6. 自动追加一行到 experiments/summary.csv，方便多组实验横向对比

用法：
  # 用配置文件
  python src/train.py --config configs/baseline.yaml

  # 配置文件 + 命令行覆盖
  python src/train.py --config configs/freq_manual.yaml --lr 0.001 --epochs 50

  # 完全命令行（不用配置文件）
  python src/train.py --name quick_gru --model GRU --freq_mode none
"""

import os
import sys
import csv
import json
import copy
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
)

# 保证无论从哪里运行，都能 import 到同目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_npz_splits, add_freq_features, TactileWindowDataset
from freq_modules import FocalLoss, LearnableBandEnergy, tune_threshold, FS

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================
# 1. 配置加载
# ============================================================

DEFAULT_CONFIG = {
    "experiment": {"name": "run", "tags": []},
    "data": {
        "processed_dir": "./processed_data",
        "freq": {"enabled": False, "mode": "none", "n_bands": 4,
                 "edges_hz": [0.0, 5.0, 15.0, 40.0, 90.0], "fs": FS, "tau": 1.5},
    },
    "model": {
        "type": "GRU", "input_size": 54, "hidden_size": 64, "num_layers": 1,
        "bidirectional": False, "dropout": 0.3, "num_classes": 2,
        # Transformer 专用
        "d_model": 64, "nhead": 4, "num_encoder_layers": 2,
    },
    "train": {
        "lr": 0.003, "batch_size": 256, "epochs": 30,
        "early_stop_patience": 7, "seed": 42, "device": "auto",
        "loss": "ce", "focal_gamma": 2.0, "focal_alpha": 0.75,
        "class_weight": False,
    },
    "eval": {"batch_size": 1024, "tune_threshold": True, "min_f1_ratio": 0.95},
    "output": {"results_dir": "./experiments"},
}


def deep_update(base, new):
    """递归合并 dict（new 覆盖 base）。"""
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_args():
    p = argparse.ArgumentParser(description="滑移检测训练脚本")
    p.add_argument("--config", type=str, default=None,
                   help="YAML 配置文件路径（如 configs/baseline.yaml）")
    # 常用覆盖项
    p.add_argument("--name", type=str, default=None, help="实验名（决定输出目录）")
    p.add_argument("--model", type=str, default=None,
                   choices=["GRU", "LSTM", "Transformer"], help="模型类型")
    p.add_argument("--freq_mode", type=str, default=None,
                   choices=["none", "manual", "learnable"], help="频带模式")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--loss", type=str, default=None,
                   choices=["ce", "focal"], help="损失函数")
    p.add_argument("--class_weight", action="store_true",
                   help="使用类别加权 CrossEntropy")
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--no_threshold_tune", action="store_true",
                   help="关闭阈值调优（用 0.5 硬阈值）")
    return p.parse_args()


def load_config(args):
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if args.config:
        if yaml is None:
            raise ImportError("需要 pyyaml：pip install pyyaml")
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        deep_update(cfg, user_cfg)
        stem = Path(args.config).stem
        cfg["experiment"]["name"] = cfg["experiment"].get("name") or stem

    # 命令行覆盖
    if args.name is not None:
        cfg["experiment"]["name"] = args.name
    if args.model is not None:
        cfg["model"]["type"] = args.model
    if args.freq_mode is not None:
        cfg["data"]["freq"]["enabled"] = (args.freq_mode != "none")
        cfg["data"]["freq"]["mode"] = args.freq_mode
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.loss is not None:
        cfg["train"]["loss"] = args.loss
    if args.class_weight:
        cfg["train"]["class_weight"] = True
    if args.processed_dir is not None:
        cfg["data"]["processed_dir"] = args.processed_dir
    if args.no_threshold_tune:
        cfg["eval"]["tune_threshold"] = False

    # 兼容旧 yaml：顶层 freq 字段写入 data.freq
    if "freq" in cfg and isinstance(cfg["freq"], dict):
        deep_update(cfg["data"]["freq"], cfg.pop("freq"))

    return cfg


# ============================================================
# 2. 模型定义
# ============================================================

class RecurrentClassifier(nn.Module):
    """GRU / LSTM 分类器，可选在 forward 里挂可学习频带模块。"""

    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=1,
                 dropout=0.3, rnn_type="GRU", bidirectional=False, freq_module=None):
        super().__init__()
        self.rnn_type = rnn_type
        self.freq_module = freq_module  # None 或 LearnableBandEnergy

        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        feat_dim = hidden_size * (2 if bidirectional else 1)

        # 因果决策：只看当前时刻（最后一个时间步）——可在线部署
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C)
        if self.freq_module is not None:
            band = self.freq_module(x)          # (B, T, K*C)
            x = torch.cat([x, band], dim=-1)    # (B, T, C + K*C)
        out, _ = self.rnn(x)
        out = out[:, -1, :]                     # 因果：取最后一步
        return self.head(out)


class TransformerClassifier(nn.Module):
    """轻量 Transformer 编码器分类器（对照组用）。"""

    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2,
                 num_classes=2, dropout=0.1, max_len=512, freq_module=None):
        super().__init__()
        self.freq_module = freq_module
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if self.freq_module is not None:
            band = self.freq_module(x)
            x = torch.cat([x, band], dim=-1)

        h = self.input_proj(x)
        T = h.size(1)
        h = h + self.pos_embed[:, :T, :]
        h = self.encoder(h)
        h = self.norm(h).mean(dim=1)   # 平均池化
        return self.head(h)


def build_model(cfg, input_size, original_input_size=None):
    """根据配置构建模型。

    Args:
        input_size: 模型实际接收的输入维度（含频带扩展）。
        original_input_size: 原始时域特征维度（54），可学习频带模块内部用。
    """
    m_cfg = cfg["model"]
    f_cfg = cfg["data"]["freq"]
    model_type = m_cfg["type"]
    original_input_size = original_input_size or input_size

    freq_module = None
    if f_cfg.get("enabled") and f_cfg.get("mode") == "learnable":
        freq_module = LearnableBandEnergy(
            n_bands=f_cfg["n_bands"],
            n_ch=original_input_size,
            fs=f_cfg.get("fs", FS),
            tau=f_cfg.get("tau", 1.5),
        )

    if model_type in ("GRU", "LSTM"):
        model = RecurrentClassifier(
            input_size=input_size,
            hidden_size=m_cfg["hidden_size"],
            num_classes=m_cfg["num_classes"],
            num_layers=m_cfg["num_layers"],
            dropout=m_cfg["dropout"],
            rnn_type=model_type,
            bidirectional=m_cfg["bidirectional"],
            freq_module=freq_module,
        )
    elif model_type == "Transformer":
        model = TransformerClassifier(
            input_size=input_size,
            d_model=m_cfg["d_model"],
            nhead=m_cfg["nhead"],
            num_layers=m_cfg["num_encoder_layers"],
            num_classes=m_cfg["num_classes"],
            dropout=m_cfg["dropout"],
            freq_module=freq_module,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    return model


# ============================================================
# 3. 工具函数
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_criterion(cfg, y_train):
    """按配置返回损失函数。"""
    t_cfg = cfg["train"]
    loss_name = t_cfg.get("loss", "ce")

    if loss_name == "focal":
        print("  损失函数: FocalLoss")
        return FocalLoss(gamma=t_cfg["focal_gamma"], alpha=t_cfg["focal_alpha"])

    if t_cfg.get("class_weight"):
        counts = np.bincount(y_train, minlength=2).astype(np.float64)
        w = counts.sum() / (2.0 * np.maximum(counts, 1.0))
        weight = torch.tensor(w, dtype=torch.float32)
        print(f"  损失函数: CrossEntropyLoss(class_weight={w.round(3).tolist()})")
        return nn.CrossEntropyLoss(weight=weight)

    print("  损失函数: CrossEntropyLoss")
    return nn.CrossEntropyLoss()


@torch.no_grad()
def predict_proba(model, X, device, batch_size=1024):
    """批量推理，返回正类(Slip)概率。"""
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def compute_metrics(y_true, y_pred):
    """计算全套二分类指标。"""
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    slip_recall = recall_score(y_true, y_pred, average="binary",
                               pos_label=1, zero_division=0)
    slip_prec = precision_score(y_true, y_pred, average="binary",
                                pos_label=1, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec_w),
        "recall": float(rec_w),
        "f1": float(f1_w),
        "mcc": float(mcc),
        "slip_precision": float(slip_prec),
        "slip_recall": float(slip_recall),
        "fnr": float(1.0 - slip_recall),          # 漏检率（安全关键指标）
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# 4. 训练循环
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, X_val, y_val, criterion, device, batch_size=1024):
    model.eval()
    total_loss, correct = 0.0, 0
    n = len(X_val)

    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X_val[i:i + batch_size]).to(device)
        yb = torch.from_numpy(y_val[i:i + batch_size]).to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()

    return total_loss / max(n, 1), correct / max(n, 1)


def fit(model, loaders, X_val, y_val, criterion, cfg, device, out_dir):
    t_cfg = cfg["train"]
    epochs = t_cfg["epochs"]
    patience_limit = t_cfg["early_stop_patience"]

    optimizer = Adam(model.parameters(), lr=t_cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, min_lr=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    patience = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        va_loss, va_acc = validate(
            model, X_val, y_val, criterion, device, cfg["eval"]["batch_size"])

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        flag = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            flag = " *best*"
        else:
            patience += 1

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train L:{tr_loss:.4f} A:{tr_acc:.4f} | "
              f"Val L:{va_loss:.4f} A:{va_acc:.4f} | "
              f"LR:{lr_now:.6f}{flag}")

        if patience >= patience_limit:
            print(f"  早停触发（验证 loss 连续 {patience_limit} 轮未提升）")
            break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存训练曲线 CSV
    with open(os.path.join(out_dir, "history.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1, history["train_loss"][i], history["val_loss"][i],
                        history["train_acc"][i], history["val_acc"][i],
                        history["lr"][i]])

    return model, history, elapsed


# ============================================================
# 5. 主流程
# ============================================================

def main():
    args = parse_args()
    cfg = load_config(args)

    run_name = cfg["experiment"]["name"]
    out_dir = os.path.join(cfg["output"]["results_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg["train"]["seed"])
    device = resolve_device(cfg["train"]["device"])

    print("=" * 64)
    print(f"实验: {run_name}")
    print("=" * 64)
    print(f"  设备      : {device}")
    print(f"  模型      : {cfg['model']['type']}")
    print(f"  频带模式  : {cfg['data']['freq']['mode']}")
    print(f"  输出目录  : {out_dir}")

    # ---------- 数据 ----------
    splits, meta = load_npz_splits(cfg["data"]["processed_dir"])
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print(f"\n  数据: Train={len(y_train)} Val={len(y_val)} Test={len(y_test)}"
          f" | Slip%: {meta['train_slip_ratio']:.1%}/{meta['val_slip_ratio']:.1%}/"
          f"{meta['test_slip_ratio']:.1%}")

    # ---------- 频带特征（决定模型真实输入维度）----------
    f_cfg = cfg["data"]["freq"]
    original_input_size = X_train.shape[-1]
    input_size = original_input_size
    band_edges_used = None

    if f_cfg.get("enabled"):
        if f_cfg.get("mode") == "manual":
            edges = f_cfg["edges_hz"]
            print(f"  手工频带: {edges} Hz")
            X_train, band_edges_used = add_freq_features(
                X_train, mode="manual", edges_hz=edges, fs=f_cfg.get("fs", FS))
            X_val, _ = add_freq_features(
                X_val, mode="manual", edges_hz=edges, fs=f_cfg.get("fs", FS))
            X_test, _ = add_freq_features(
                X_test, mode="manual", edges_hz=edges, fs=f_cfg.get("fs", FS))
            input_size = X_train.shape[-1]
            print(f"  特征维度: {input_size} ({original_input_size} 时域 + {input_size - original_input_size} 频带)")

        elif f_cfg.get("mode") == "learnable":
            n_bands = f_cfg.get("n_bands", 4)
            input_size = original_input_size + n_bands * original_input_size
            print(f"  可学习频带: {n_bands} bands, 输入维度将扩展为 {input_size} "
                  f"({original_input_size} 时域 + {n_bands * original_input_size} 频带)")

    # ---------- DataLoader ----------
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(cfg["train"]["seed"])

    loaders = {
        "train": DataLoader(
            TactileWindowDataset(X_train, y_train),
            batch_size=cfg["train"]["batch_size"], shuffle=True,
            drop_last=False, generator=g),
    }

    # ---------- 模型 ----------
    set_seed(cfg["train"]["seed"])
    model = build_model(cfg, input_size, original_input_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  参数量: {n_params:,} (可训练 {n_trainable:,})")

    criterion = build_criterion(cfg, y_train)
    if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)

    # ---------- 训练 ----------
    print(f"\n{'-'*64}\n开始训练\n{'-'*64}")
    model, history, elapsed = fit(
        model, loaders, X_val, y_val, criterion, cfg, device, out_dir)
    print(f"\n训练完成，用时 {elapsed:.1f}s")

    # ---------- 阈值 ----------
    p_val = predict_proba(model, X_val, device, cfg["eval"]["batch_size"])
    p_test = predict_proba(model, X_test, device, cfg["eval"]["batch_size"])

    if cfg["eval"]["tune_threshold"]:
        thr, val_m, test_m = tune_threshold(
            y_val, p_val, y_test, p_test,
            min_f1_ratio=cfg["eval"]["min_f1_ratio"])
        print(f"\n  阈值调优: 选择 threshold={thr:.3f}（验证集 FNR 优先）")
    else:
        thr = 0.5
        y_pred_test = (p_test >= thr).astype(int)
        test_m = compute_metrics(y_test, y_pred_test)
        print(f"\n  使用默认阈值 {thr}")

    # ---------- 测试集评估 ----------
    y_pred = (p_test >= thr).astype(int)
    metrics = compute_metrics(y_test, y_pred)

    print(f"\n{'='*64}\n测试集结果\n{'='*64}")
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
        "run_name": run_name,
        "model_type": cfg["model"]["type"],
        "freq_mode": cfg["data"]["freq"]["mode"],
        "threshold": float(thr),
        "n_params": int(n_params),
        "train_time_sec": round(elapsed, 2),
        "band_edges_used": band_edges_used,
        "data_meta": meta,
    })

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    torch.save({
        "model_state": model.state_dict(),
        "config": cfg,
        "threshold": float(thr),
        "input_size": input_size,
    }, os.path.join(out_dir, "best.pt"))

    if yaml is not None:
        with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # 汇总 CSV（追加，便于横向对比）
    summary_path = os.path.join(cfg["output"]["results_dir"], "summary.csv")
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    row = {
        "run_name": run_name,
        "model": cfg["model"]["type"],
        "freq_mode": cfg["data"]["freq"]["mode"],
        "loss": cfg["train"].get("loss", "ce"),
        "class_weight": cfg["train"].get("class_weight", False),
        "lr": cfg["train"]["lr"],
        "epochs_ran": len(history["train_loss"]),
        "threshold": round(float(thr), 4),
        "acc": round(metrics["accuracy"], 4),
        "f1": round(metrics["f1"], 4),
        "mcc": round(metrics["mcc"], 4),
        "fnr": round(metrics["fnr"], 4),
        "params": int(n_params),
        "time_sec": round(elapsed, 1),
    }
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"\n结果已保存:")
    print(f"  指标    : {os.path.join(out_dir, 'metrics.json')}")
    print(f"  曲线    : {os.path.join(out_dir, 'history.csv')}")
    print(f"  权重    : {os.path.join(out_dir, 'best.pt')}")
    print(f"  汇总    : {summary_path}")


if __name__ == "__main__":
    main()
