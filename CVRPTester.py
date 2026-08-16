# -*- coding: utf-8 -*-
"""
CVRPTester.py —— 中文注释版（学习用）

文件作用：
    模型的“测试/评估”模块。与 CVRPTrainer 不同，这里不做任何梯度更新，
    只负责加载训练好的 checkpoint，在测试数据上运行 POMO 采样，
    统计普通分数（no-aug score）和 8 倍数据增广后的分数（aug score）。

整体流程：
    1. __init__：读参数、准备设备（GPU/CPU）、创建环境与模型、恢复 checkpoint
    2. run()：按批次循环测试，用 TimeEstimator 估算剩余时间并打印日志
    3. _test_one_batch()：对一批问题做增广 -> POMO rollout -> 计算平均分数

本文件不改动任何原始逻辑，只增加注释，方便学习阅读。
"""

import torch

from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils import *   # 引入 get_result_folder、TimeEstimator、AverageMeter 等工具


class CVRPTester:
    """
    测试器：加载已训练好的模型，在测试问题上评估性能。
    """
    def __init__(self,
                 env_params,      # 环境参数（问题规模、车辆容量等）
                 model_params,    # 模型参数（编码器/解码器维度等）
                 tester_params):  # 测试参数（batch 大小、测试轮数、是否增广等）

        # ---------- 保存参数 ----------
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # ---------- 结果文件夹与日志器 ----------
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()   # 生成带时间戳的结果目录

        # ---------- 设备（GPU / CPU）----------
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)                 # 指定使用哪块 GPU
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')  # 默认张量类型设为 CUDA
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ---------- 创建环境（CVRP 环境）和模型（注意力模型）----------
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # ---------- 加载训练好的模型参数（checkpoint）----------
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)  # 加载到指定设备
        self.model.load_state_dict(checkpoint['model_state_dict'])          # 恢复模型权重

        # ---------- 时间估算器（用于打印已用/剩余时间）----------
        self.time_estimator = TimeEstimator()

    def run(self):
        """
        主测试流程：按批次测试全部问题，并打印统计结果。
        """
        self.time_estimator.reset()   # 重新计时

        score_AM = AverageMeter()      # 记录普通分数（无增广）的平均值
        aug_score_AM = AverageMeter()  # 记录增广后分数的平均值

        # 如果配置了保存好的测试数据，就加载它（否则使用随机生成的问题）
        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']  # 总共要测试多少个问题
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            # 每次最多测 test_batch_size 个，最后一批可能不足，取两者较小值
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # 测一个 batch，返回（无增广分数, 增广分数）
            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)      # 累加进平均值统计
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            # ---------- 日志：进度 + 已用/剩余时间 + 本批分数 ----------
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            # 全部测完：打印最终平均分数
            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size):
        """
        对一批问题执行测试：
        1) 可选 8 倍数据增广（把坐标做 8 种对称变换，提升结果稳定性）
        2) 模型前向编码所有节点
        3) POMO rollout：多个起点并行生成解
        4) 计算本批平均分数
        """

        # ---------- 增广倍数 ----------
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']   # 例如 8
        else:
            aug_factor = 1                                  # 不增广

        # ---------- 准备：模型切到 eval 模式，关闭梯度 ----------
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)  # 生成/加载 batch*aug_factor 份问题
            reset_state, _, _ = self.env.reset()            # 环境回到初始状态
            self.model.pre_forward(reset_state)             # 一次性编码所有节点（编码器只跑一次）

        # ---------- POMO Rollout（并行多起点解码）----------
        state, reward, done = self.env.pre_step()           # 第一步：每个 POMO 起点各自选第一个客户
        while not done:
            selected, _ = self.model(state)                 # 根据当前状态选下一个客户
            # selected shape: (batch, pomo) —— 每个问题、每个起点各选一个动作
            state, reward, done = self.env.step(selected)   # 环境更新状态、累计奖励、判断是否结束

        # ---------- 计算分数 ----------
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (增广倍数, batch, pomo) —— 把奖励按“增广”维度拆开

        max_pomo_reward, _ = aug_reward.max(dim=2)          # 对每个问题，取 POMO 多个起点里最好的结果
        # shape: (增广倍数, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # 只用第 0 份（原始数据，未增广）的结果；取负使成本为正

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)   # 再在“增广”维度上取最好的
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()       # 增广后的分数（通常 <= 普通分数）

        return no_aug_score.item(), aug_score.item()          # 转成 Python 数值返回
