"""
freq_modules.py — 频域模块（从你暑假代码搬过来）
================================================

包含:
  1) compute_band_energy_batch : 手工固定频带能量（向量化批量版）
  2) LearnableBandEnergy      : 可学习频带能量层（受 FTFNet 启发）
  3) FocalLoss                : Focal Loss（二分类）
  4) tune_threshold           : 决策阈值调优

这是把「入门小任务 SMOTE处理/freq_modules.py」原封不动搬过来。
作者注释、版权全部保留。
"""

import numpy as np
import torch
import torch.nn as nn


# ========== 全局常量（与 preprocess 保持一致） ==========
FS = 180.0
NYQUIST = FS / 2.0
BAND_EDGES_DEFAULT = [0.0, 5.0, 15.0, 40.0, 90.0]


def compute_band_energy_batch(X, edges=None, fs=FS):
    """
    对一批窗口计算手工频带能量特征（numpy 向量化）。

    输入:
      X: (N, T, C) 时域窗口，T=60, C=54
      edges: 频带边界 (Hz)，默认 [0, 5, 15, 40, 90]
      fs: 采样率 (Hz)
    返回:
      band: (N, T, K*C) 频带能量，每个频带对每个通道算一个能量
    """
    if edges is None:
        edges = BAND_EDGES_DEFAULT
    edges = np.asarray(edges, dtype=np.float32)
    N, T, C = X.shape
    K = len(edges) - 1

    spectrum = np.fft.rfft(X, n=T, axis=1)
    mag2 = np.abs(spectrum) ** 2

    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    band = np.zeros((N, T, K * C), dtype=np.float32)

    for b_idx in range(K):
        mask = (freqs >= edges[b_idx]) & (freqs < edges[b_idx + 1])
        energy = mag2[:, mask, :].sum(axis=1)
        band[:, :, b_idx * C:(b_idx + 1) * C] = energy[:, None, :]

    return band


class LearnableBandEnergy(nn.Module):
    """
    可学习频带能量层（受 FTFNet 启发的简化版）。

    频带边界由参数 beta 决定：边界 = cumsum(softmax(beta)) * Nyquist
    softmax 保证每段长度为正，cumsum 保证边界单调递增。
    """

    def __init__(self, n_bands=4, n_ch=54, fs=FS, tau=1.5):
        super().__init__()
        self.n_bands = n_bands
        self.n_ch = n_ch
        self.fs = fs
        self.tau = tau
        init = torch.full((n_bands - 1,), 0.0)
        self.beta = nn.Parameter(init)

    def band_edges(self):
        w = torch.softmax(self.beta, dim=0)
        cum = torch.cumsum(w, dim=0)
        edges = cum / cum[-1] * (NYQUIST * 0.9)
        edges = torch.cat([torch.zeros(1, device=edges.device), edges])
        edges = torch.cat([edges, torch.full((1,), NYQUIST, device=edges.device)])
        return edges

    def forward(self, x):
        B, T, C = x.shape
        spec = torch.fft.rfft(x, n=T, dim=1)
        mag2 = torch.abs(spec) ** 2
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs, device=x.device)

        edges = self.band_edges()
        feats = []
        for b_idx in range(self.n_bands):
            lo, hi = edges[b_idx], edges[b_idx + 1]
            m = torch.sigmoid((freqs - lo) / self.tau) - torch.sigmoid((freqs - hi) / self.tau)
            e = (mag2 * m[None, :, None]).sum(dim=1)
            feats.append(e)
        out = torch.stack(feats, dim=-1).reshape(B, self.n_bands * C)
        return out[:, None, :].expand(B, T, self.n_bands * C).contiguous()


class FocalLoss(nn.Module):
    """
    Focal Loss（二分类版）。
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.softmax(logits, dim=1)
        pt = p.gather(1, targets[:, None]).squeeze(1)
        logpt = logp.gather(1, targets[:, None]).squeeze(1)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )
        loss = -alpha_t * (1.0 - pt) ** self.gamma * logpt
        return loss.mean()


def tune_threshold(y_val, p_val, y_test=None, p_test=None,
                   min_f1_ratio=0.95, grid=None):
    """
    在验证集上选择决策阈值：F1 不低于最优 F1 × min_f1_ratio 的前提下，FNR 最小。
    """
    if grid is None:
        grid = np.linspace(0.02, 0.98, 97)

    f1s, fnrs = [], []
    for t in grid:
        pred = (p_val >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fnr = 1.0 - rec
        f1s.append(f1)
        fnrs.append(fnr)

    f1s, fnrs = np.array(f1s), np.array(fnrs)
    best_f1 = f1s.max()
    candidates = [t for t, f1, fnr in zip(grid, f1s, fnrs)
                  if f1 >= best_f1 * min_f1_ratio]
    if not candidates:
        best_thr = grid[int(f1s.argmax())]
    else:
        cand_fnrs = [fnrs[np.where(grid == t)[0][0]] for t in candidates]
        best_thr = candidates[int(np.argmin(cand_fnrs))]

    def _metrics(y, p, t):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        return {"acc": acc, "precision": prec, "recall": rec,
                "f1": f1, "fnr": 1.0 - rec, "threshold": t,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    val_metrics = _metrics(y_val, p_val, best_thr)
    test_metrics = _metrics(y_test, p_test, best_thr) if (y_test is not None and p_test is not None) else None
    return best_thr, val_metrics, test_metrics