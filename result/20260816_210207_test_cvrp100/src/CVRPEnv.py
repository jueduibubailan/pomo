# CVRPEnv（中文注释版）
# 原代码：POMO 项目（MIT License, Copyright (c) 2021 Yeong-Dae Kwon）
# 用途：CVRP（带容量约束的车辆路径问题）的“环境”。
#       环境负责：
#           1. 生成/加载随机问题（城市坐标、需求量）
#           2. 管理每一步的状态（当前城市、剩余容量、可去城市掩码）
#           3. 路径结束后计算奖励（路线长度取负号）
#
# 关键设定：
#   - 仓库编号为 0，客户节点编号为 1 ~ problem_size
#   - 剩余容量 load 是归一化的：初始为 1.0（= 100% 容量），回仓库自动补满
#   - POMO 每个问题同时维护 pomo_size 条解，所以很多张量形状是 (batch, pomo, ...)

from dataclasses import dataclass   # 数据类：用来定义简单的“装数据的容器”
import torch                        # PyTorch 深度学习框架

# 从问题定义文件导入：随机生成问题、把数据做 8 倍旋转/镜像增广
from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold

@dataclass
class Reset_State:
    """
    重置（reset）时返回的环境初始信息：
    只是把问题数据打包成一个对象，方便模型读取。
    """
    depot_xy: torch.Tensor = None
    # 形状: (batch, 1, 2) —— 每个问题的仓库坐标
    node_xy: torch.Tensor = None
    # 形状: (batch, problem, 2) —— 每个客户节点的坐标
    node_demand: torch.Tensor = None
    # 形状: (batch, problem) —— 每个客户节点的需求量

@dataclass
class Step_State:
    """
    每走一步时返回给模型的当前状态：
    模型只需要看这个对象，就能决定下一步去哪。
    """
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # 形状: (batch, pomo) —— 用来索引“哪个问题、哪条解”
    selected_count: int = None
    # 已走的步数（第几步）
    load: torch.Tensor = None
    # 形状: (batch, pomo) —— 每条解当前剩余容量（0~1 之间，1 表示满载）
    current_node: torch.Tensor = None
    # 形状: (batch, pomo) —— 每条解当前所在城市编号
    ninf_mask: torch.Tensor = None
    # 形状: (batch, pomo, problem+1) —— 非法城市掩码（0 可去，负无穷不可去）
    finished: torch.Tensor = None
    # 形状: (batch, pomo) —— 每条解是否已经访问完全部客户节点

class CVRPEnv:
    """
    CVRP 环境主体。
    和 Gym 类似，提供 reset() / step() 接口：
        reset() -> 返回初始问题数据
        step(selected) -> 让所有解同时走一步，返回新状态
    """

    def __init__(self, **env_params):

        # ---- 初始化时固定不变的常量 ----
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']   # 客户节点数量
        self.pomo_size = env_params['pomo_size']         # 每个问题的并行解数量

        # 是否使用“预先保存好的问题”做测试（flag 默认关闭）
        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None   # 读取保存问题时用到的“读到第几个”指针

        # ---- 加载问题后确定的大小 ----
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo) —— 预先生成好的索引，后面取数用
        self.depot_node_xy = None
        # 形状: (batch, problem+1, 2) —— 仓库 + 所有客户的坐标
        self.depot_node_demand = None
        # 形状: (batch, problem+1) —— 仓库 + 所有客户的需求量（仓库为 0）

        # ---- 动态状态 1（路径层面的信息）----
        ####################################
        self.selected_count = None       # 已经走了几步
        self.current_node = None
        # 形状: (batch, pomo) —— 当前所在城市
        self.selected_node_list = None
        # 形状: (batch, pomo, 0~) —— 完整记录每条解走过的城市序列（路径）

        # ---- 动态状态 2（约束层面的信息）----
        ####################################
        self.at_the_depot = None
        # 形状: (batch, pomo) —— 当前是否在仓库
        self.load = None
        # 形状: (batch, pomo) —— 剩余容量（1 = 满载，回仓库补满到 1）
        self.visited_ninf_flag = None
        # 形状: (batch, pomo, problem+1) —— 记录哪些城市已经访问过（访问过 = 负无穷）
        self.ninf_mask = None
        # 形状: (batch, pomo, problem+1) —— 最终的“不可去”掩码
        #       = 已访问掩码 + 需求量超过剩余容量的掩码
        self.finished = None
        # 形状: (batch, pomo) —— 是否已访问完全部客户（只剩回仓库）

        # ---- 返回给模型的状态对象 ----
        ####################################
        self.reset_state = Reset_State()   # reset 时返回
        self.step_state = Step_State()     # step 时返回

    def use_saved_problems(self, filename, device):
        """
        使用预先保存的问题集（通常是固定测试集），保证评测结果可复现。
        """
        self.FLAG__use_saved_problems = True

        # 从文件加载问题数据
        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0   # 从第 0 个问题开始读

    def load_problems(self, batch_size, aug_factor=1):
        """
        加载 batch 个问题，供这一轮训练/测试使用。

        aug_factor：数据增广倍数。
            - 1  ：不增广
            - 8  ：把每个问题做 8 种旋转/镜像变换，得到 8 个等价变体，
                   用“同一答案验证多个视角”的方式提高评测稳定性
        """
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            # 正常训练：随机生成一批问题
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            # 评测：从保存的问题里按顺序取一批
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size]
            self.saved_index += batch_size   # 指针后移

        # 数据增广：8 倍时 batch 也扩大 8 倍
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)     # 坐标做 8 种变换
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)             # 需求量只重复，不改变
            else:
                raise NotImplementedError

        # 把仓库和客户拼在一起，方便按编号统一取数
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # 形状: (batch, problem+1, 2) —— 索引 0 是仓库，1~problem 是客户
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # 形状: (batch, 1) —— 仓库需求量是 0
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # 形状: (batch, problem+1)

        # 预生成索引矩阵，后面做 gather 取数时用
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        # 把问题数据放进 reset_state，供模型 pre_forward 读取
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        # 索引放进 step_state，供模型 forward 读取
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        """
        重置环境，开始新的一条路径。
        初始化所有动态状态，返回初始问题数据。
        """
        self.selected_count = 0
        self.current_node = None
        # 形状: (batch, pomo)

        # 已选节点列表初始为空（第三维长度 0，之后每步追加）
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # 形状: (batch, pomo, 0~)

        # 一开始“在仓库”：True（路径起点固定在仓库）
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # 形状: (batch, pomo)

        # 剩余容量 = 1（满载出发）
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # 形状: (batch, pomo)

        # 已访问掩码：全 0（什么都没访问过）
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # 形状: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # 形状: (batch, pomo, problem+1)

        # 完成标志：全 False
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # 形状: (batch, pomo)

        reward = None   # 路径还没走完，没有奖励
        done = False    # 还没结束
        return self.reset_state, reward, done

    def pre_step(self):
        """
        “第 0 步”的过渡：在模型第一次决策前，把初始状态打包返回。
        （路径的第一步是强制回仓库，由模型 forward 里的规则处理）
        """
        # 把当前内部状态同步到 step_state 里
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        """
        所有解同时走一步。

        参数 selected：模型选好的下一站城市编号，形状 (batch, pomo)。
        本函数会更新：路径记录、剩余容量、访问掩码、完成标志，
        全部完成时计算奖励（= 负的路线总长度）。
        """
        # selected.shape: (batch, pomo)

        # ---- 动态状态 1：更新路径 ----
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # 形状: (batch, pomo)
        # 把这一步选的城市追加到路径记录里
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # 形状: (batch, pomo, 0~) —— 第三维逐步变长

        # ---- 动态状态 2：更新容量与掩码 ----
        ####################################
        # 判断每条解是否“正在仓库”（选了城市 0）
        self.at_the_depot = (selected == 0)

        # 取出被选中的城市的“需求量”：
        # 先把所有城市的需求量复制到每条解上，再用编号 gather 取对应值
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # 形状: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # 形状: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # 形状: (batch, pomo) —— 本次去到的城市的需求量

        # 扣除容量；如果去了仓库，则容量补满为 1
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # 回到仓库重新装满

        # 把这次访问的城市标记为“已访问”（负无穷）
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # 形状: (batch, pomo, problem+1)

        # 仓库特殊处理：只要不在仓库，就把仓库恢复成“可去”状态（0），
        # 因为仓库可以多次回访（补货）。而“正在仓库”时保持负无穷，
        # 防止模型原地不动（选了仓库又选仓库）。
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        # 最终掩码 = 已访问掩码 + 容量不足掩码
        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001   # 一点点容差，避免浮点误差导致误判
        # 需求量 > 剩余容量 + 容差 的城市不可去
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # 形状: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # 形状: (batch, pomo, problem+1)

        # 判断是否“全部客户都访问过了”：此时已访问掩码全是负无穷
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # 形状: (batch, pomo)
        # 完成标志累加（布尔相加等价于“或”），一旦完成就一直保持
        self.finished = self.finished + newly_finished
        # 形状: (batch, pomo)

        # 对已完成的解，把仓库掩码恢复为 0：
        # 因为所有客户都去过了，最后一步只能回仓库
        self.ninf_mask[:, :, 0][self.finished] = 0

        # 同步状态到 step_state，返回给模型
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # ---- 结束判定与奖励 ----
        done = self.finished.all()   # 所有解都完成了才算结束
        if done:
            # 奖励 = 负的路线总长度
            # 注意负号：强化学习习惯“奖励越大越好”，
            # 而路径越短越好，所以把距离取负号变成奖励
            reward = -self._get_travel_distance()
        else:
            reward = None   # 没结束就没有奖励（稀疏奖励，只在最后给）

        return self.step_state, reward, done

    def _get_travel_distance(self):
        """
        根据记录的城市序列，计算每条解的总行驶距离。
        原理：把相邻两个城市连起来算欧氏距离，再求和。
        因为路径是闭合的（最后会回仓库），还要算“最后一个城市 -> 仓库”这一段，
        实现上用一个巧妙的技巧：把序列整体向左平移一位（roll），
        原序列和移位序列逐点相减，正好得到每段首尾相连的距离。
        """
        # 把路径中每个城市编号，映射成它的坐标
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # 形状: (batch, pomo, 路径长度, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # 形状: (batch, pomo, problem+1, 2)

        # 按路径顺序取出坐标序列
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # 形状: (batch, pomo, 路径长度, 2)

        # 向左平移一位：第 i 个位置变成“下一个城市”的坐标，
        # 这样原序列和移位序列逐点相减 = 相邻两城之间的距离
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        # 计算每段距离：差值的平方和开根号（欧氏距离）
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # 形状: (batch, pomo, 路径长度)

        # 把所有路段加起来 = 总路程
        travel_distances = segment_lengths.sum(2)
        # 形状: (batch, pomo)
        return travel_distances
