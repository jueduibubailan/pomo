# CVRPTrainer（中文注释版）
# 原代码：POMO 项目（MIT License, Copyright (c) 2021 Yeong-Dae Kwon）
# 用途：负责 CVRP（带容量约束的车辆路径问题）强化学习模型的训练流程，
#       包括数据加载、模型训练、日志记录、曲线绘制和断点保存/恢复。
#
# 训练算法：REINFORCE（策略梯度）+ POMO 多起点技巧
#   - POMO 会从一个问题实例生成 pomo_size 条不同的解（每个起点一条）
#   - 用同批次所有解的奖励平均值作为基线（baseline），计算优势（advantage）
#   - 损失 = -优势 x 对数概率，梯度上升让高奖励解的概率变大

from logging import getLogger   # 获取日志对象（和 utils 里的 logger 配合使用）
from utils import *
import torch
from CVRPModel import CVRPModel as Model # 注意力网络模型：根据状态输出选哪个城市的概率
from CVRPEnv import CVRPEnv as Env       # CVRP 环境：负责问题生成、状态转换、奖励计算
from torch.optim import Adam as Optimizer          # Adam 优化器
from torch.optim.lr_scheduler import MultiStepLR as Scheduler  # 多段学习率衰减

class CVRPTrainer:
    def __init__(self,
                 env_params,          # 环境参数（问题规模、容量等）
                 model_params,        # 模型参数（嵌入维度、注意力层数等）
                 optimizer_params,    # 优化器与学习率调度参数
                 trainer_params): # 训练参数（epoch 数、批大小、CUDA、保存间隔等）
        # ---- 保存所有参数，方便类内部各处使用 ----
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # ---- 日志与结果保存 ----
        self.logger = getLogger(name='trainer')     # 本类的专用日志对象
        self.result_folder = get_result_folder()    # 结果文件夹路径
        self.result_log = LogData()                 # 记录训练曲线数据（如 train_score）

        # ---- 设备选择（GPU / CPU） ----
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            # 指定用哪块 GPU，并把默认张量类型设为 GPU 上的 FloatTensor
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            # 没有 GPU 就用 CPU
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # 模型：根据当前状态输出选择下一个城市的概率分布
        self.model = Model(**self.model_params)
        # 环境：随机生成 CVRP 问题、执行一步动作并返回新状态和奖励
        self.env = Env(**self.env_params)
        # 优化器：更新模型参数（Adam）
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        # 调度器：按设定好的 epoch 节点降低学习率（如 80/100/120 个 epoch 后衰减）
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # ---- 断点恢复：加载之前训练好的模型继续训练 ----
        self.start_epoch = 1                        # 默认从第 1 个 epoch 开始
        model_load = trainer_params['model_load']   # 读取配置里是否要加载旧模型
        if model_load['enable']:
            # 拼出检查点文件路径，例如 ./result/xxx/checkpoint-100.pt
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            # 加载检查点（map_location 保证 GPU/CPU 可以互相迁移）
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            # 恢复模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # 从上次结束的 epoch 的下一个开始继续训练
            self.start_epoch = 1 + model_load['epoch']
            # 恢复之前记录的曲线数据（让画图能接着画）
            self.result_log.set_raw_data(checkpoint['result_log'])
            # 恢复优化器状态（动量等历史信息）
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 让学习率调度器接着上次的位置走（last_epoch 设为上次的 epoch-1）
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded !!')
        # ---- 时间估算器：打印“已用时间/预计剩余时间” ----
        self.time_estimator = TimeEstimator()

    def run(self):
        """
        主训练循环：从 start_epoch 一直训练到总 epoch 数。
        每个 epoch 做的事情：
            1. 学习率衰减一步
            2. 训练一个 epoch（若干批数据）
            3. 记录并打印训练得分/损失
            4. 画最新曲线、按间隔保存模型和曲线图
        """
        # 让时间估算器从 start_epoch 开始计时
        self.time_estimator.reset(self.start_epoch)

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            # 学习率衰减：调度器每步会根据已走过的 epoch 调整学习率
            self.scheduler.step()

            # 训练一个 epoch，返回平均得分和平均损失
            train_score, train_loss = self._train_one_epoch(epoch)
            # 记录曲线数据：x = epoch，y = 得分 / 损失
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # 日志记录与检查点保存
            ############################
            # 打印本 epoch 的预计已用/剩余时间
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            # 是否全部训练完成（最后一个 epoch）
            all_done = (epoch == self.trainer_params['epochs'])
            # 每隔多少个 epoch 保存一次模型 / 图片（从配置里读取）
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # 每个 epoch 都保存一张“最新曲线图”（从第 2 个 epoch 开始，因为第 1 个只有 1 个点）
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                # 用配置好的样式分别画 train_score 和 train_loss 两条曲线
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            # 保存模型检查点：训练结束，或到了设定的保存间隔
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                # 把当前所有状态打包成一个字典：
                checkpoint_dict = {
                    'epoch': epoch,                                   # 当前 epoch
                    'model_state_dict': self.model.state_dict(),      # 模型权重
                    'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态
                    'scheduler_state_dict': self.scheduler.state_dict(),  # 调度器状态
                    'result_log': self.result_log.get_raw_data()      # 已记录的曲线数据
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # 保存带 epoch 编号的曲线图（训练结束，或到了图片保存间隔）
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            # 训练结束：打印完成信息，并把所有曲线数据打印出来
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
    def _train_one_epoch(self, epoch):
        """
        训练一个完整的 epoch：
        把训练数据按 batch 分批喂给模型，累加每个 batch 的得分与损失，
        最后返回本 epoch 的平均得分和平均损失。
        """
        # 两个“平均值计数器”：分别累计本 epoch 的得分和损失
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        # 本 epoch 要训练的总回合数（episode = 一个问题实例）
        train_num_episode = self.trainer_params['train_episodes']
        episode = 0        # 已处理的回合数
        loop_cnt = 0       # 用于只在前 10 个 batch 打印详细日志
        while episode < train_num_episode:

            # 剩余不足一个完整 batch 时，只取剩余数量（保证不超）
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # 训练这一个 batch，返回该 batch 的平均得分和平均损失
            avg_score, avg_loss = self._train_one_batch(batch_size)
            # 累加进平均值计数器（每个值代表 batch_size 个样本）
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # 只在“从断点恢复后的第一个 epoch”打印前 10 个 batch 的进度，
            # 方便确认训练是否正常起步
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # 每个 epoch 结束时打印一次整体平均结果
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        # 返回本 epoch 的平均得分和平均损失
        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        """
        训练一个 batch 的核心过程，也是强化学习的关键：
            1. 重置模型与环境，加载 batch 个随机问题
            2. POMO 采样：对每个问题生成 pomo_size 条解（从不同起点出发）
            3. 用 REINFORCE 计算损失（奖励 - 基线 作为优势）
            4. 反向传播并更新模型参数
        """

        # ---- 准备阶段 ----
        ###############################################
        self.model.train()                    # 切换到训练模式（启用 dropout 等）
        self.env.load_problems(batch_size)    # 给环境加载 batch 个随机 CVRP 问题
        reset_state, _, _ = self.env.reset()  # 重置环境，得到初始状态
        self.model.pre_forward(reset_state)   # 模型预计算：把节点特征编码成嵌入向量（只算一次）

        # 存放每一步选城市的概率（用于计算整条路径的联合概率）
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # 形状：(batch, pomo, 0~problem)
        #   第一维：batch 里的每个问题
        #   第二维：pomo（同一个问题的不同起点，POMO 的核心）
        #   第三维：路径长度（每一步追加一个概率，这里从 0 开始）

        # ---- POMO 路径采样（Rollout）----
        ###############################################
        # 先走“第一步之前”的预处理步骤，得到初始状态（第一个起点等）
        state, reward, done = self.env.pre_step()

        # 循环直到当前问题全部完成（done=True）
        while not done:
            # 模型根据当前状态选择下一个城市，并给出对应概率
            selected, prob = self.model(state)
            # 形状：(batch, pomo) —— 每个问题、每个起点各选了一个城市
            # 环境执行这些动作：状态转移、计算容量约束、更新奖励
            state, reward, done = self.env.step(selected)
            # 把这一步的概率存进 prob_list（第三维拼接，记录整条路径）
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # ---- 计算损失（REINFORCE 算法）----
        ###############################################
        # 优势 = 当前奖励 - 同批次所有解的奖励平均值（作为基线）
        # POMO 的思想：奖励比“平均解”好的解，概率要增大；比平均差的，概率要减小
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # 形状：(batch, pomo)

        # 整条路径的对数概率 = 每一步对数概率之和
        log_prob = prob_list.log().sum(dim=2)
        # 形状：(batch, pomo)

        # 损失 = -优势 x 对数概率
        # 前面加负号是因为我们要“最大化奖励”，而 PyTorch 做的是梯度下降最小化损失
        loss = -advantage * log_prob
        # 形状：(batch, pomo)
        # 所有样本取平均，得到一个标量损失
        loss_mean = loss.mean()

        # ---- 计算得分（用于记录和画图）----
        ###############################################
        # 同一个问题从 pomo 个起点得到 pomo 条解，取其中最好的（奖励最大）作为该问题成绩
        max_pomo_reward, _ = reward.max(dim=1)  # 取每个问题中 pomo 解里的最好奖励
        # 奖励是负的（路径越长越负），取负号变成正的路径长度/成本，方便直观理解
        score_mean = -max_pomo_reward.float().mean()  # 平均到 batch 上

        # ---- 反向传播与参数更新 ----
        ###############################################
        self.model.zero_grad()    # 清空上一次的梯度
        loss_mean.backward()      # 反向传播计算梯度
        self.optimizer.step()     # 优化器更新模型参数

        # 返回标量形式的得分和损失（转成 Python 数字，用于日志与画图）
        return score_mean.item(), loss_mean.item()
