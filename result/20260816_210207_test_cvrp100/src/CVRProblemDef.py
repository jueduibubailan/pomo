# CVRProblemDef（中文注释版）
# 原代码：POMO 项目（MIT License, Copyright (c) 2021 Yeong-Dae Kwon）
# 用途：定义 CVRP 问题数据的生成方式，以及评测时用的 8 倍数据增广。
#       这份文件是 CVRPEnv 的“上游”，环境里的问题都从这里来。

import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    """
    随机生成一批 CVRP 问题实例。

    每个问题包含：
        - 1 个仓库（depot）：坐标在 [0,1] 正方形内随机
        - problem_size 个客户节点：坐标同样在 [0,1] 内随机
        - 每个客户有一个需求量（归一化到 0~1 之间的小数）

    参数：
        batch_size   : 一次生成多少个问题
        problem_size : 每个问题有多少个客户节点（20 / 50 / 100）

    返回值：
        depot_xy     : 仓库坐标      (batch, 1, problem_size)
        node_xy      : 客户坐标      (batch, problem, 2)
        node_demand  : 客户需求量    (batch, problem)
    """

    # 仓库坐标：在 [0,1] 正方形内均匀随机
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    # 根据问题规模选择“需求量缩放系数”。
    # 车辆容量被归一化为 1.0，所以需求量也必须缩放到 0~1 之间。
    # 规模越大，可服务的客户越多，需求量的缩放系数也越大
    # （否则总需求量会轻易超过容量，问题会太难）。
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    # 每个客户的需求量：先随机取整数 1~9，再除以缩放系数，
    # 得到 0.03~0.30（20 节点）、0.025~0.225（50 节点）、0.02~0.18（100 节点）左右
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def augment_xy_data_by_8_fold(xy_data):
    """
    对坐标数据做 8 倍增广（只用于评测，不用于训练）。

    原理：所有坐标都在 [0,1] 单位正方形内，
    对它做 8 种“对称变换”（镜像 + 旋转 90 度的所有组合），
    会得到 8 个几何上完全等价的问题。
    同一批问题用 8 个视角各算一遍答案，取最好的结果，
    相当于变相提高了求解质量，且不改变问题的本质。

    参数：
        xy_data : 坐标数据，形状 (batch, N, 2)
                  比如 (batch, problem, 2) 或 (batch, 1, 2)

    返回：
        8 个变体拼接后的数据，形状 (8*batch, N, 2)
    """
    # xy_data.shape: (batch, N, 2)

    # 把 x 坐标和 y 坐标拆开
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    # 8 种变换：
    #   1~4：原样 / 左右镜像（x -> 1-x）/ 上下镜像（y -> 1-y）/ 两者都镜像
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    #   5~8：把 x、y 互换（等价于沿对角线镜像，可看作旋转 90 度的组合）再配合镜像
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    # 把 8 个变体按 batch 维拼接，batch 变成原来的 8 倍
    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data