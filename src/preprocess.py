"""
力学时序信号分类 —— 滑移检测（二分类）+ 可选 SMOTE 过采样
=================================================

这是从你暑假三份代码合并后的统一入口。

变化点（相对暑假原版）：
  1. 支持命令行参数（argparse），不再需要改源码
  2. 路径不再写死 BASE_DIR，可通过 --data_dir 指定
  3. SMOTE 配置从常量改为命令行参数
  4. 增加 --objects 允许挑选部分物体（为后续跨物体泛化实验铺垫）

用法：
  # 默认 baseline（无 SMOTE）
  python src/preprocess.py

  # 启用 SMOTE
  python src/preprocess.py --use_smote

  # 改 SMOTE 比例
  python src/preprocess.py --use_smote --smote_ratio 1.0

  # 只用 brush 一个物体做实验
  python src/preprocess.py --objects brush

数据集: ARQ-CRISP slip_detection_dataset_2021 (uSkin, 18 pins × 3 axes, 180Hz)
任务:   滑移检测 (二分类: NoSlip=0 vs Slip=1)
"""

import os
import argparse
import h5py
import numpy as np
from collections import defaultdict
from imblearn.over_sampling import SMOTE


# ============ 默认配置（命令行参数会覆盖这些）============
DEFAULTS = {
    "data_dir":      "./data",
    "output_dir":    "./processed_data",
    "window_size":   60,        # ~333ms @ 180Hz
    "stride":        20,
    "test_ratio":    0.15,
    "val_ratio":     0.15,
    "seed":          42,
    "use_smote":     False,
    "smote_k":       3,
    "smote_ratio":   0.5,
    "objects":       ["brush", "screwDriver", "SpoolSolder"],
}


# 3 个 HDF5 文件 → 3 种物体（注意 screwDriver 文件名带 (1)）
OBJECT_FILES = {
    "brush":       "slipDataset_brush_tactile.h5",
    "screwDriver": "slipDataset_screwDriver_tactile(1).h5",
    "SpoolSolder": "slipDataset_SpoolSolder_tactile.h5",
}

# 滑移标签含义（来自数据集官方文档）：
# 0: 未抓取/抓取静止，无滑移
# 1: 抓取静止，有滑移
# 2: 物体被释放
# 3: 物体被抓取
# 4: 抓取移动中，无滑移
# 5: 抓取移动中，有滑移
# 6: 其他触觉事件
SLIP_LABELS = {1, 5}
NOSLIP_LABELS = {0, 4}


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="滑移检测数据预处理（窗口切分 + 划分 + 可选 SMOTE）"
    )
    p.add_argument("--data_dir", default=DEFAULTS["data_dir"],
                   help=f"原始 h5 文件目录（默认 {DEFAULTS['data_dir']}）")
    p.add_argument("--output_dir", default=DEFAULTS["output_dir"],
                   help=f"预处理 npz 输出目录（默认 {DEFAULTS['output_dir']}）")
    p.add_argument("--window_size", type=int, default=DEFAULTS["window_size"],
                   help=f"滑动窗口大小（默认 {DEFAULTS['window_size']}）")
    p.add_argument("--stride", type=int, default=DEFAULTS["stride"],
                   help=f"滑动窗口步长（默认 {DEFAULTS['stride']}）")
    p.add_argument("--test_ratio", type=float, default=DEFAULTS["test_ratio"],
                   help=f"测试集比例（默认 {DEFAULTS['test_ratio']}）")
    p.add_argument("--val_ratio", type=float, default=DEFAULTS["val_ratio"],
                   help=f"验证集比例（默认 {DEFAULTS['val_ratio']}）")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                   help=f"随机种子（默认 {DEFAULTS['seed']}）")

    # SMOTE 配置
    p.add_argument("--use_smote", action="store_true",
                   help="启用训练集 SMOTE 过采样")
    p.add_argument("--smote_k", type=int, default=DEFAULTS["smote_k"],
                   help=f"SMOTE 近邻数（默认 {DEFAULTS['smote_k']}）")
    p.add_argument("--smote_ratio", type=float, default=DEFAULTS["smote_ratio"],
                   help=f"SMOTE 过采样目标比例（默认 {DEFAULTS['smote_ratio']}）")

    # 物体选择（为跨物体泛化实验铺垫）
    p.add_argument("--objects", nargs="+",
                   default=DEFAULTS["objects"],
                   choices=list(OBJECT_FILES.keys()),
                   help=f"使用哪些物体（默认 {' '.join(DEFAULTS['objects'])}）")

    return p.parse_args()


def load_all_h5_files(args):
    """
    加载 args.objects 指定的所有 HDF5 文件，提取滑动窗口。

    返回:
      X: (N, window_size, 54) 时序特征
      y_slip: (N,) 滑移标签 {0=NoSlip, 1=Slip}
    """
    X_list, y_slip_list = [], []
    stats = defaultdict(int)

    for obj_name in args.objects:
        fname = OBJECT_FILES[obj_name]
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [跳过] 文件不存在: {fpath}")
            continue

        f = h5py.File(fpath, "r")

        # 数据结构: GGCNN2/{obj_name}/{Pose1-3}/{Exp1-10}/datasets
        for root_key in f.keys():  # "GGCNN2"
            root_group = f[root_key]
            for obj_key in root_group.keys():
                if obj_key != obj_name:
                    continue
                obj_group = root_group[obj_key]
                for pose_key in obj_group.keys():
                    pose_group = obj_group[pose_key]
                    for exp_key in pose_group.keys():
                        exp_group = pose_group[exp_key]

                        raw = exp_group["tactile_data_raw"][:]
                        slips = exp_group["tactile_slips_label"][:]

                        # raw shape: (18, 3, T) → (T, 18, 3) → (T, 54)
                        T = raw.shape[2]
                        data = raw.transpose(2, 0, 1).reshape(T, -1).astype(np.float32)

                        stats["total_experiments"] += 1
                        stats[f"{obj_name}_experiments"] += 1

                        # 滑动窗口切分
                        for start in range(0, T - args.window_size + 1, args.stride):
                            window = data[start:start + args.window_size]
                            window_labels = slips[start:start + args.window_size]

                            # 窗口级标签：多数投票
                            noslip_count = sum(1 for l in window_labels if l in NOSLIP_LABELS)
                            slip_count = sum(1 for l in window_labels if l in SLIP_LABELS)

                            if noslip_count > slip_count and noslip_count > len(window_labels) * 0.7:
                                slip_label = 0
                            elif slip_count > noslip_count and slip_count > len(window_labels) * 0.7:
                                slip_label = 1
                            else:
                                continue  # 跳过混合窗口

                            X_list.append(window)
                            y_slip_list.append(slip_label)

        f.close()

    X = np.stack(X_list, axis=0)
    y_slip = np.array(y_slip_list, dtype=np.int64)

    print(f"\n{'='*60}")
    print(f"数据集统计:")
    print(f"{'='*60}")
    print(f"  总实验次数: {stats['total_experiments']}")
    for obj in args.objects:
        print(f"  {obj}: {stats[f'{obj}_experiments']}次实验")
    print(f"  滑动窗口总数: {X.shape[0]}")
    print(f"  数据形状: {X.shape}")
    print(f"\n  --- 滑移检测标签分布 ---")
    print(f"  NoSlip(0): {np.sum(y_slip==0)} ({np.sum(y_slip==0)/len(y_slip)*100:.1f}%)")
    print(f"  Slip(1):   {np.sum(y_slip==1)} ({np.sum(y_slip==1)/len(y_slip)*100:.1f}%)")

    return X, y_slip


def _apply_smote(X_train, y_slip_train, args):
    """仅对滑移检测任务的训练集做 SMOTE（窗口级，2D flatten）。"""
    print(f"\n  --- SMOTE 处理（仅训练集 / 仅滑移检测任务）---")
    print(f"  处理前: NoSlip={int((y_slip_train==0).sum())}, "
          f"Slip={int((y_slip_train==1).sum())}, "
          f"Slip占比={y_slip_train.mean():.2%}")

    N, T, F = X_train.shape
    X_flat = X_train.reshape(N, T * F)

    n_minority = int((y_slip_train == 1).sum())
    k = min(args.smote_k, n_minority - 1) if n_minority > 1 else 1

    smote = SMOTE(
        random_state=args.seed,
        k_neighbors=k,
        sampling_strategy=args.smote_ratio,
    )

    X_sm_flat, y_slip_sm = smote.fit_resample(X_flat, y_slip_train)
    X_train = X_sm_flat.reshape(-1, T, F)

    print(f"  处理后: NoSlip={int((y_slip_sm==0).sum())}, "
          f"Slip={int((y_slip_sm==1).sum())}, "
          f"Slip占比={y_slip_sm.mean():.2%}")

    return X_train, y_slip_sm


def split_and_save(X, y_slip, args):
    """划分训练/验证/测试集，仅对训练集做 SMOTE，然后保存。"""
    n = len(y_slip)
    indices = np.random.permutation(n)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    X_train = X[train_idx]
    y_slip_train = y_slip[train_idx]

    if args.use_smote:
        X_train, y_slip_train = _apply_smote(X_train, y_slip_train, args)

    os.makedirs(args.output_dir, exist_ok=True)

    splits = {
        "train": (X_train, y_slip_train),
        "val":   (X[val_idx],   y_slip[val_idx]),
        "test":  (X[test_idx],  y_slip[test_idx]),
    }

    for name, (dx, dy_slip) in splits.items():
        np.savez(
            os.path.join(args.output_dir, f"{name}.npz"),
            X=dx, y_slip=dy_slip
        )
        print(f"  {name}: {dx.shape[0]} samples, Slip%={dy_slip.mean()*100:.1f}%")

    # Z-score 归一化（只用【原始】训练集统计量，未用 SMOTE 合成样本）
    mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
    std = X[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
    np.savez(os.path.join(args.output_dir, "norm_params.npz"), mean=mean, std=std)
    print(f"\n  归一化参数已保存 (mean shape={mean.shape})")

    # 把本次预处理参数也存一份，方便后续追溯
    config = vars(args).copy()
    # 把 list 转成 str，避免 yaml dump 报错
    config["objects"] = " ".join(config["objects"])
    np.savez(
        os.path.join(args.output_dir, "preprocess_config.npz"),
        **config
    )


def main_entry():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("Step 1: 加载 ARQ-CRISP slip_detection_dataset_2021")
    print("=" * 60)
    print(f"  data_dir = {args.data_dir}")
    print(f"  objects  = {args.objects}")
    print(f"  use_smote = {args.use_smote}")
    X, y_slip = load_all_h5_files(args)

    print(f"\n{'='*60}")
    print("Step 2: 划分训练/验证/测试集")
    print(f"{'='*60}")
    split_and_save(X, y_slip, args)

    print(f"\n预处理完成! 数据保存至: {args.output_dir}")


if __name__ == "__main__":
    main_entry()