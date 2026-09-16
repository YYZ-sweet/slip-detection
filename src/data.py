"""
data.py — 数据加载与封装
========================

这是新拆出来的模块，把"加载 + 归一化 + 打包 Dataset"这件事从 train.py 里剥出来。

为什么拆出来？
  - preprocess.py 负责"一次性预处理"（加载 h5 → 切窗 → 划分 → 存 npz）
  - data.py 负责"训练时取数据"（加载 npz → 归一化 → 转 tensor → 可选拼频带 → 批采样）
  - train.py 负责"训练循环"（forward/loss/optim/eval）

这样 train.py 里就不会有一堆 data 相关的散乱代码。

用法：
  from data import build_loaders

  loaders = build_loaders(config)
  for X_batch, y_batch in loaders["train"]:
      ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TactileWindowDataset(Dataset):
    """
    触觉窗口 PyTorch Dataset。

    输入:
      X: (N, T, C) numpy array，已 Z-score 归一化
      y: (N,) numpy array，标签 0/1
      augment: 是否做时序数据增强（可选）
    返回:
      __getitem__: (X_tensor, y_tensor)
    """

    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            # 简单的时间抖动增强（窗口长度 60±5 随机 crop）
            # 留给你按需实现，先空着
            pass
        return x, y


def load_npz_splits(processed_dir):
    """
    从 processed_dir 加载 train/val/test + 归一化参数。

    返回:
      splits: {"train": (X, y), "val": (X, y), "test": (X, y)}，已归一化
      meta:   dict 含 sample 数量、归一化参数等
    """
    def _load(name):
        d = np.load(os.path.join(processed_dir, f"{name}.npz"))
        return d["X"], d["y_slip"]

    X_train, y_train = _load("train")
    X_val,   y_val   = _load("val")
    X_test,  y_test  = _load("test")

    norm = np.load(os.path.join(processed_dir, "norm_params.npz"))
    mean, std = norm["mean"], norm["std"]

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    meta = {
        "n_train": len(y_train),
        "n_val":   len(y_val),
        "n_test":  len(y_test),
        "train_slip_ratio": float(y_train.mean()),
        "val_slip_ratio":   float(y_val.mean()),
        "test_slip_ratio":  float(y_test.mean()),
        "T": X_train.shape[1],
        "C": X_train.shape[2],
    }

    splits = {
        "train": (X_train.astype(np.float32), y_train),
        "val":   (X_val.astype(np.float32),   y_val),
        "test":  (X_test.astype(np.float32),  y_test),
    }
    return splits, meta


def build_loaders(config):
    """
    构造 DataLoaders。

    config: dict，至少含:
      - data.processed_dir
      - train.batch_size
      - eval.batch_size
      - train.seed

    返回:
      loaders: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
      meta:    同 load_npz_splits 的 meta
    """
    data_cfg = config["data"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]

    splits, meta = load_npz_splits(data_cfg["processed_dir"])

    train_ds = TactileWindowDataset(*splits["train"], augment=False)
    val_ds   = TactileWindowDataset(*splits["val"],   augment=False)
    test_ds  = TactileWindowDataset(*splits["test"],  augment=False)

    g = torch.Generator()
    g.manual_seed(train_cfg["seed"])

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=train_cfg["batch_size"],
            shuffle=True, drop_last=False, generator=g,
        ),
        "val": DataLoader(
            val_ds, batch_size=eval_cfg["batch_size"],
            shuffle=False, drop_last=False,
        ),
        "test": DataLoader(
            test_ds, batch_size=eval_cfg["batch_size"],
            shuffle=False, drop_last=False,
        ),
    }
    return loaders, meta


def fit_band_norm(band, mode="log_zscore", eps=1e-8):
    """
    在训练集频带能量上拟合标准化参数（只能在训练集上调用，防止泄漏）。

    为什么需要：频带能量是 |rFFT|^2，量级可达 1e5~1e6，
    而时域特征 Z-score 后标准差为 1 —— 直接拼接会让 RNN 输入尺度差 1e4 倍，
    频带特征完全主导、时域信息被淹没。

    参数:
      band: (N, T, K*C) 原始频带能量
      mode: "log_zscore"（推荐，能量近似对数正态）/"zscore"/"none"

    返回:
      norm_params: dict 或 None
    """
    if mode == "none":
        return None

    b = np.log1p(band) if mode == "log_zscore" else band.astype(np.float64)
    # 按 (频带, 通道) 分别统计：形状 (K*C,)
    mean = b.mean(axis=(0, 1))
    std = b.std(axis=(0, 1)) + eps
    return {"mode": mode,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32)}


def apply_band_norm(band, norm_params, eps=1e-8):
    """用 fit_band_norm 得到的参数标准化频带能量。"""
    if norm_params is None:
        return band
    mode = norm_params.get("mode", "log_zscore")
    b = np.log1p(band) if mode == "log_zscore" else band.astype(np.float64)
    return ((b - norm_params["mean"]) / (norm_params["std"] + eps)).astype(np.float32)


def add_freq_features(X, mode="manual", norm_params=None, **kwargs):
    """
    给时域窗口拼接频带能量特征（沿时间维广播）。

    输入:
      X: (N, T, C) numpy array
      mode: "manual" 或 "learnable"
      norm_params: fit_band_norm 的返回值；None 表示不标准化（不推荐）
      kwargs:
        - manual:   edges_hz (list), fs (float)
        - learnable: 见 freq_modules.py 的 LearnableBandEnergy

    返回:
      X_aug: (N, T, C + K*C) 拼接后的特征
      band_edges_used: 实际使用的频带边界（Hz），仅 manual 模式有意义
    """
    if mode == "manual":
        from freq_modules import compute_band_energy_batch
        edges_hz = kwargs.get("edges_hz", [0.0, 5.0, 15.0, 40.0, 90.0])
        fs = kwargs.get("fs", 180.0)
        band = compute_band_energy_batch(X, edges=np.array(edges_hz), fs=fs)
        band = apply_band_norm(band, norm_params)
        return np.concatenate([X, band], axis=-1), edges_hz

    elif mode == "learnable":
        # learnable 模式需要在模型里动态算（因为边界随训练变）
        # 这里只返回 X 不变，频带拼接在模型的 forward 里完成
        return X, None

    else:
        raise ValueError(f"Unknown freq mode: {mode}")