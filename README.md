# Slip Detection — 触觉滑移检测

> 首都师范大学信工 / 研一 / 导师刘铁 / 方向：力控灵巧手 / 触觉感知
> 起点：暑假期间 GitHub 数据集 + LSTM/GRU 滑移检测 → 加入 SMOTE 过采样 → 加入频带特征
> 目标：研一发表一篇小论文（中文核心 / EI 会议起步，研二冲 RA-L）

---

## 1. 一键复现

```bash
# 装环境
conda create -n tactile python=3.10
conda activate tactile
pip install -r requirements.txt

# 1. 把数据放到 data/ 下（h5 文件，不入 git）
#    data/slipDataset_brush_tactile.h5
#    data/slipDataset_screwDriver_tactile.h5
#    data/slipDataset_SpoolSolder_tactile.h5

# 2. 预处理（默认 baseline：不启用 SMOTE）
python src/preprocess.py
# 或：启用 SMOTE
python src/preprocess.py --use_smote

# 3. 训练 baseline
python src/train.py --config configs/baseline.yaml

# 4. 训练 + 频带
python src/train.py --config configs/freq_manual.yaml
```

---

## 2. 项目结构

```
slip-detection/
├── README.md             ← 你正在看
├── requirements.txt
├── .gitignore
├── configs/              ← 每个 yaml 对应一组实验
│   ├── baseline.yaml
│   ├── smote.yaml
│   ├── freq_manual.yaml
│   └── freq_learnable.yaml
├── src/
│   ├── preprocess.py     ← 加载 h5 + 切窗 + 划分 + 可选 SMOTE
│   ├── data.py           ← 加载 npz + 归一化 + PyTorch Dataset
│   ├── train.py          ← 训练入口（支持 yaml 配置）
│   ├── eval.py           ← 评估指标 + 阈值调优
│   └── freq_modules.py   ← 手工/可学习频带 + Focal Loss
├── data/                 ← 原始 h5（.gitignore）
├── processed_data/       ← 预处理 npz（.gitignore）
├── outputs/              ← 模型 ckpt + 曲线图（.gitignore）
├── experiments/          ← 实验结果 CSV / JSON（不忽略，跟着代码走）
├── docs/                 ← 设计文档
└── notes/                ← 论文五问笔记
```

---

## 3. 实验矩阵

| 配置文件 | 含义 | 对应原版 |
|---|---|---|
| `baseline.yaml` | 纯时域 GRU，无 SMOTE 无频带 | 入门小任务 |
| `smote.yaml` | 纯时域 GRU + 训练集 SMOTE | 入门小任务 SMOTE处理 |
| `freq_manual.yaml` | 纯时域 GRU + 手工 0-5/5-15/15-40/40-90 Hz 频带能量 | 入门小任务加频带 |
| `freq_learnable.yaml` | 纯时域 GRU + 可学习频带边界（FTFNet 简化版） | 入门小任务加频带/YuanBao 进阶版 |

---

## 4. 工作进度（自查清单）

### 第一阶段：工程化地基（W1-W2）

- [ ] GitHub 仓库建好，第一次推送
- [ ] `preprocess.py` 支持命令行参数（已改造）
- [ ] `train.py` 支持 yaml 配置
- [ ] WSL2 Ubuntu 下能跑通
- [ ] `requirements.txt` 固化
- [ ] 写好总 README（就是本文件）

### 第二阶段：精读四剑客（W3-W4）

- [ ] Reactive Slip Control (ICRA 2026) 五问笔记
- [ ] Ye et al. (Science Robotics 2026) 五问笔记
- [ ] FORTE (RA-L 2026) 五问笔记
- [ ] Within arm's reach (Science Robotics 2026) 五问笔记
- [ ] 在周会上用 10 分钟讲清"我和 Reactive Slip 那篇的差异"

### 第三阶段：方向对齐（W5-W6）

- [ ] 开题预答辩幻灯片
- [ ] 和导师面对面/视频汇报一次
- [ ] 开题报告文献综述初稿

### 第四阶段：补齐实验（W7-W10）

- [ ] 在 BioTacSP-DoS 上跑通 baseline
- [ ] 复现 Sparsh TacBench 滑移任务
- [ ] 测三维度数据：延迟 / 跨物体泛化 / 轻量化

### 第五阶段：创新点 + 论文（W11-W14）

- [ ] 确定创新点
- [ ] 写 6-8 页论文初稿

### 第六阶段：打磨 + 投稿（W15-W16）

- [ ] 投稿
- [ ] 寒假计划

---

## 5. 注意事项

- **原始数据（h5）不入 git**，在 `.gitignore` 已配
- **预处理数据（npz）也不入 git**，跑 `preprocess.py` 重新生成即可
- **实验结果（experiments/）跟代码一起提交**，每次实验一个文件夹，commit message 写清做了什么
- **commit 频率**：哪怕只改一行也 commit，养成习惯
- **遇到报错先搜 30 分钟**，再决定问谁

---

## 6. 关键论文清单（速查）

完整版见 `../论文清单.md`。本工程要特别关注这 4 篇：

1. **Reactive Slip Control in Multiingered Grasping** — ICRA 2026 (arXiv 2602.16127) — 必精读
2. **Ye et al., Visual-Tactile Pretraining** — Science Robotics 2026.1 — 必精读
3. **FORTE: Tactile Force and Slip Sensing on Compliant Fingers** — RA-L 2026 — 必精读
4. **Within arm's reach** — Science Robotics 2026.1 — 必精读