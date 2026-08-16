# POMO for CVRP · 中文注释学习版

> 基于 [HKUDS/POMO](https://github.com/HKUDS/POMO) 的带容量约束车辆路径问题（CVRP）强化学习求解实现。
> 本仓库为学习版本：**代码逻辑与原始实现保持一致**，在此之上加入全量中文注释，并针对本机环境做了兼容性调整，方便阅读、调试与二次开发。

![Python](https://img.shields.io/badge/Python-3.13.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2Bcu132-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 项目简介

本项目使用 **POMO**（*Policy Optimization with Multiple Optima*，NeurIPS 2020）算法求解 **CVRP**（带容量约束的车辆路径问题）：

- 用注意力编码器-解码器网络生成路线；
- 每个实例同时从多个不同起点并行解码（`pomo_size` 条候选路线），从而逼近多个局部最优；
- 使用 REINFORCE 算法训练，并以“所有起点奖励的平均值”作为共享基线，降低方差；
- 测试时通过 8 倍坐标增广进一步提升解的稳定性。

> 简单说：**训练**时模型学会“怎样选下一个客户更优”，**测试**时用它生成一条条完整路线并统计平均成本。

---

## 相对原版的改动

| 改动 | 说明 |
| --- | --- |
| 🧠 全量中文注释 | 环境、模型、训练器、测试器及工具函数均补充了中文注释，便于学习 |
| ⏱️ 修复时间显示问题 | `TimeEstimator` 改用 `time.time()` 获取真实时间戳，训练/测试进度中的剩余时间估算恢复正常 |
| 🔧 修复 `_ops.py` 兼容问题 | `copy_all_src()` 复制源码快照时跳过 `torch.ops` 这类“伪模块”的不存在路径，避免 `FileNotFoundError` |
| 🚀 调低训练规模 | `train_n100.py` 默认 `epochs=10`，方便快速验证流程；正式训练时改回大 epoch 即可 |

---

## 环境要求

本项目在以下环境中验证通过：

| 依赖 | 版本 |
| --- | --- |
| Python | 3.13.14 |
| PyTorch | 2.13.0+cu132（GPU 版） |
| CUDA | 13.2 |
| GPU | NVIDIA RTX 4060 Laptop 8GB |

---

## 快速开始

### 1. 训练

```bash
python train_n100.py
```

训练入口为 `train_n100.py`。训练时会在 `result/` 下生成带时间戳的结果目录，并每 `model_save_interval` 个 epoch 保存一次模型。

### 2. 测试

```bash
python test_n100.py
```

测试入口为 `test_n100.py`。测试前请先调整 `tester_params['model_load']`，把 `path` 指向包含训练好的 checkpoint 的目录，例如：

```python
'model_load': {
    'path': './result/saved_CVRP100_model',  # 存放 checkpoint 的目录
    'epoch': 30500,                          # 加载第几个 epoch 的模型
}
```

测试数据默认使用项目自带的 `vrp100_test_seed1234.pt`（固定种子，便于对比）。

---

## 项目结构

```text
pomo-study/
├── train_n100.py        # 训练入口（N=100）
├── test_n100.py         # 测试入口（N=100）
├── CVRPTrainer.py       # 训练器：REINFORCE + POMO 多起点训练流程
├── CVRPTester.py        # 测试器：加载 checkpoint，批量评估 + 8 倍增广
├── CVRPModel.py         # 注意力编码器 / 解码器网络
├── CVRPEnv.py           # CVRP 环境：状态、容量约束、掩码、奖励
├── CVRProblemDef.py     # 问题定义：随机生成实例 + 8 倍坐标增广
├── utils.py             # 日志、结果保存、时间估算、绘图等工具
├── log_image_style/     # 绘图样式 JSON 配置
├── result/              # 训练/测试结果（含 pomo_model 模型目录）
├── vrp100_test_seed1234.pt  # 固定种子的测试数据
└── README.md
```

---

## 核心参数说明

### 环境参数（`train_n100.py`）

| 参数 | 含义 | 当前值 |
| --- | --- | --- |
| `problem_size` | 客户节点数 N（不含仓库 0） | 100 |
| `pomo_size` | 每个实例并行的起点数（候选路线数） | 100 |

### 模型参数

| 参数 | 含义 | 当前值 |
| --- | --- | --- |
| `embedding_dim` | 节点嵌入向量维度 | 128 |
| `encoder_layer_num` | 编码器堆叠层数 | 6 |
| `head_num` | 多头注意力头数 | 8 |
| `qkv_dim` | 每个注意力头的维度（8×16=128） | 16 |
| `logit_clipping` | logit 裁剪系数 c | 10 |
| `ff_hidden_dim` | 前馈网络隐藏层维度 | 512 |
| `eval_type` | 解码方式：`argmax` 贪心 / `softmax` 采样 | argmax |

### 训练参数

| 参数 | 含义 | 当前值 |
| --- | --- | --- |
| `epochs` | 训练总 epoch 数 | 10（快速验证，正式可调回 8100） |
| `train_episodes` | 每个 epoch 处理的实例数 | 10,000 |
| `train_batch_size` | 每批样本数 | 64 |
| `lr` / `weight_decay` | Adam 学习率 / 权重衰减 | 1e-4 / 1e-6 |
| `milestones` / `gamma` | 学习率衰减节点 / 衰减系数 | [8001, 8051] / 0.1 |
| `model_save_interval` | 每多少 epoch 保存一次 checkpoint | 500 |

### 测试参数（`test_n100.py`）

| 参数 | 含义 | 当前值 |
| --- | --- | --- |
| `test_episodes` | 测试实例总数 | 10,000 |
| `test_batch_size` | 测试 batch 大小（增广时取 `aug_batch_size`） | 400 |
| `augmentation_enable` / `aug_factor` | 是否启用坐标增广 / 增广倍数 | True / 8 |
| `test_data_load` | 是否加载固定测试数据 | True（`vrp100_test_seed1234.pt`） |

---

## 训练产物

每次运行会在 `result/` 下生成类似 `20260816_195854_train_cvrp_n100_with_instNorm` 的目录：

```text
result/<时间戳>_train_cvrp_n100_with_instNorm/
├── run_log                 # 运行日志
├── src/                    # 本次运行的源码快照（方便复现）
├── fig_train_score.jpg     # 训练得分曲线（每 img_save_interval 保存）
├── fig_train_loss.jpg      # 训练损失曲线
└── pomo_model/             # 训练好的模型 checkpoint
    └── checkpoint-{epoch}.pt
```

> 模型文件保存在 `pomo_model/` 目录下，恢复训练或测试时通过 `model_load` 指定 `path` 与 `epoch`。

---

## 常见问题（FAQ）

**Q1：`pomo的原代码出现 AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'`**

说明用错了环境：原因是cuda与torch不适配，建议下载适配的troch。

**Q2：`FileNotFoundError: [Errno 2] No such file or directory: '...\_ops.py'`**

旧版python与新版不兼容的问题，具体修改请参考对应代码部分。

**Q3：`TypeError: unsupported operand type(s) for -: 'datetime.time' and 'datetime.time'`**

不要使用旧版from datetime import time，直接调用time这个库。

---

## 参考与致谢

- 论文：Yeong-Dae Kwon et al., *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning*, NeurIPS 2020
- 原始代码：[HKUDS/POMO](https://github.com/HKUDS/POMO)
- 许可证：MIT
