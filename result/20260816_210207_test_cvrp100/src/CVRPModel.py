# CVRPModel（中文注释版）
# 原代码：POMO 项目（MIT License, Copyright (c) 2021 Yeong-Dae Kwon）
# 用途：CVRP（带容量约束的车辆路径问题）的深度神经网络模型，
#       基于“注意力机制”（Attention），结构类似 Transformer：
#           - 编码器（Encoder）：一次性把所有城市（含仓库）编码成向量
#           - 解码器（Decoder）：每走一步，根据“当前所在城市 + 剩余容量”
#                               算出下一个去哪个城市的概率
#
# 训练时（POMO）：解码器每一步都要为 batch 个问题 x pomo 个不同起点
# 同时生成决策，所以很多张量的形状是 (batch, pomo, ...)。

import torch                # PyTorch 深度学习框架
import torch.nn as nn       # 神经网络模块库（线性层、激活函数等）
import torch.nn.functional as F  # 神经网络函数库（relu、softmax 等）

class CVRPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params       # 模型参数（维度、层数、注意力头数等）

        self.encoder = CVRP_Encoder(**model_params)   # 编码器
        self.decoder = CVRP_Decoder(**model_params)   # 解码器
        self.encoded_nodes = None              # 保存编码结果，供解码器每一步复用
        # 形状: (batch, problem+1, EMBEDDING_DIM)
        # 注意 problem+1：多出来的是“仓库”（depot，索引 0），其余是客户节点
    def pre_forward(self, reset_state):
        """
        预编码：在开始走路径之前，把所有节点信息一次性编码。
        因为所有节点信息在整条路径中是不变的，所以只算一次，后面直接复用。
        """
        depot_xy = reset_state.depot_xy
        # 形状: (batch, 1, 2) —— 仓库的 x、y 坐标
        node_xy = reset_state.node_xy
        # 形状: (batch, problem, 2) —— 每个客户节点的 x、y 坐标
        node_demand = reset_state.node_demand
        # 形状: (batch, problem) —— 每个客户节点的需求量

        # 把需求量和坐标拼在一起：每个节点用 [x, y, 需求量] 三个特征表示
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # 形状: (batch, problem, 3)

        # 送入编码器，得到所有节点的嵌入向量（仓库也一起编码）
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        # 形状: (batch, problem+1, embedding)

        # 把编码结果准备好（转成注意力需要的 K、V），解码器每一步直接用
        self.decoder.set_kv(self.encoded_nodes)
    def forward(self, state):
        """
        每走一步调用一次，返回 (selected, prob)：
            selected : 本步选中的城市编号，形状 (batch, pomo)
            prob     : 选中城市的概率，形状 (batch, pomo)（非训练时可为 None）

        分三种情况处理前两步的特殊规则：
            - 第 1 步：强制回仓库（路径起点是仓库）
            - 第 2 步：POMO 技巧——让 pomo 条解分别从不同客户节点出发
            - 第 3 步起：正常用解码器按概率选城市
        """
        batch_size = state.BATCH_IDX.size(0)   # batch 大小
        pomo_size = state.BATCH_IDX.size(1)    # 每个问题的并行解数量（POMO）

        if state.selected_count == 0:  # 第 1 步：必须先选仓库（索引 0）
            # 所有解都选仓库，所以 selected 全为 0，概率为 1
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

            # 下面是被注释掉的“原始 Attention Model”写法：
            # 它把“所有节点平均”作为上下文（q1），把“第一个访问的城市”作为另一个上下文（q2）。
            # POMO 简化后不再使用，所以 set_q1 / set_q2 相关代码都被注释保留着。

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # 第 2 步：POMO 多起点
            # 让第 i 条解从“城市 i+1”出发（城市编号从 1 开始，0 是仓库）
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))  # 这一步是规定好的，概率为 1

        else:  # 第 3 步及以后：正常决策
            # 取出“当前所在城市”的嵌入向量（每个解当前所在的城市可能不同）
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # 形状: (batch, pomo, embedding)

            # 解码器计算去往每个城市（含仓库）的概率
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
            # 形状: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                # 训练 / 软采样模式：按概率分布随机抽样（探索）
                while True:  # 防止抽样抽到概率为 0 的城市（pytorch 旧版 bug 的规避写法）
                    with torch.no_grad():
                        # 把 (batch*pomo, 候选城市数) 的分布按行抽样，选一个城市
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # 形状: (batch, pomo)

                    # 取出被选中的那个城市对应的概率
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # 形状: (batch, pomo)

                    # 只要所有被选中的概率都不为 0，就接受这次抽样（否则重抽）
                    if (prob != 0).all():
                        break
            else:
                # 贪婪模式（evaluation）：直接选概率最大的城市，不需要概率值
                selected = probs.argmax(dim=2)
                # 形状: (batch, pomo)
                prob = None  # 概率值不需要，随便给个 None

        return selected, prob

########################################
# 编码器（ENCODER）
########################################

class CVRP_Encoder(nn.Module):
    """
    编码器：把原始输入变成嵌入向量，并经过若干层自注意力。
    Transformer 架构中的 Encoder 部分。
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']        # 嵌入维度
        encoder_layer_num = self.model_params['encoder_layer_num']  # 编码器层数

        # 仓库只有 2 个输入特征（x, y），客户节点有 3 个（x, y, 需求量）
        self.embedding_depot = nn.Linear(2, embedding_dim)   # 仓库特征 -> 嵌入
        self.embedding_node = nn.Linear(3, embedding_dim)    # 节点特征 -> 嵌入
        # 堆叠 encoder_layer_num 层自注意力层
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand):
        """
        输入仓库坐标和节点特征，输出所有节点（含仓库）的编码向量。
        """
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # 形状: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        # 形状: (batch, problem, embedding)

        # 仓库和客户节点拼在一起，仓库排最前面（索引 0）
        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # 形状: (batch, problem+1, embedding)

        # 逐层经过自注意力层，每层输出还是同样形状
        for layer in self.layers:
            out = layer(out)

        return out
        # 形状: (batch, problem+1, embedding)
class EncoderLayer(nn.Module):
    """
    单层编码器（标准 Transformer Encoder Block）：
        多头自注意力 -> 残差连接 + 归一化 -> 前馈网络 -> 残差连接 + 归一化
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']  # 嵌入维度
        head_num = self.model_params['head_num']            # 注意力头数
        qkv_dim = self.model_params['qkv_dim']              # 每个头的 Q/K/V 维度

        # 自注意力：每个节点同时扮演 Query/Key/Value
        # 输入 embedding 维，输出 head_num * qkv_dim（每个头各一份）
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # 把多头结果拼回去：head_num*qkv_dim -> embedding
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # 残差 + 归一化（用 InstanceNorm，对每个样本单独归一化）
        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        # 前馈网络（两层 MLP）
        self.feed_forward = FeedForward(**model_params)
        # 第二个残差 + 归一化
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        """
        输入：所有节点的嵌入 (batch, problem+1, embedding)
        输出：经过自注意力后的新嵌入（形状不变）
        """
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        # 每个节点都同时生成 Q、K、V，并按注意力头拆开
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q/k/v 形状: (batch, head_num, problem, qkv_dim)

        # 多头自注意力：每个节点去“看”所有其他节点，加权汇总信息
        out_concat = multi_head_attention(q, k, v)
        # 形状: (batch, problem, head_num*qkv_dim)

        # 线性投影，把多头结果合并回 embedding 维度
        multi_head_out = self.multi_head_combine(out_concat)
        # 形状: (batch, problem, embedding)

        # 残差连接 + 归一化（把输入和注意力输出相加）
        out1 = self.add_n_normalization_1(input1, multi_head_out)
        # 前馈网络（增加非线性表达能力）
        out2 = self.feed_forward(out1)
        # 再次残差连接 + 归一化
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # 形状: (batch, problem, embedding)

class FeedForward(nn.Module):
    """
    前馈网络（FFN）：两层全连接 + ReLU 激活。
    作用：给注意力输出增加非线性变换能力（Transformer 标准组件）。
    """

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']  # 中间隐藏层维度（通常比 embedding 大）

        # 第一层：embedding -> ff_hidden（升维）
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        # 第二层：ff_hidden -> embedding（降回原维度）
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        # 线性变换 -> ReLU 激活 -> 线性变换
        return self.W2(F.relu(self.W1(input1)))

class AddAndInstanceNormalization(nn.Module):
    """
    残差连接 + 实例归一化：
        输出 = InstanceNorm(输入1 + 输入2)
    归一化是按“每个样本每个特征通道”独立做的（InstanceNorm），
    对每个问题实例独立归一化，适合 batch 里实例差异大的情况。
    """

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        # InstanceNorm1d：对每个样本的每个通道单独归一化
        # affine=True 表示归一化后还带可学习的缩放/偏移参数
        # track_running_stats=False 表示只按当前 batch 统计，不记录全局统计量
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        """
        输入两个形状相同的张量，输出它们的“残差归一化”结果。
        """
        # input.shape: (batch, problem, embedding)

        added = input1 + input2   # 残差连接（相加）
        # 形状: (batch, problem, embedding)

        # InstanceNorm1d 期望 (batch, channels, length)，所以把 embedding 换到第 2 维
        transposed = added.transpose(1, 2)
        # 形状: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # 形状: (batch, embedding, problem)

        # 换回原来的维度顺序
        back_trans = normalized.transpose(1, 2)
        # 形状: (batch, problem, embedding)

        return back_trans

def _get_encoding(encoded_nodes, node_index_to_pick):
    """
    从所有节点的编码中，按城市编号“取出”对应城市的向量。
    相当于给每个 (batch, pomo) 的位置，收集它当前所在城市的嵌入。
    """
    # encoded_nodes.shape: (batch, problem+1, embedding)
    # node_index_to_pick.shape: (batch, pomo) —— 每个解当前所在的城市编号

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    # 把城市编号复制成和 embedding 同维度，方便用 gather 取向量
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # 形状: (batch, pomo, embedding)

    # gather：沿“城市”这一维，按编号取对应的嵌入向量
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # 形状: (batch, pomo, embedding)

    return picked_nodes

########################################
# 解码器（DECODER）
########################################

class CVRP_Decoder(nn.Module):
    """
    解码器：每走一步，根据“当前城市 + 剩余容量”算出下一步的概率分布。
    包含两部分：
        1. 多头注意力：当前城市作为 Query，去关注所有城市（含仓库）
        2. 单头注意力：把注意力结果和所有城市比较，算出一个匹配分数，再转成概率
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # 下面两个被注释的 Wq_1 / Wq_2 属于旧版 Attention Model 的上下文查询，
        # POMO 只用“当前节点 + 剩余容量”，所以只需要 Wq_last
        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # Query 输入 = 当前城市嵌入 + 剩余容量（所以输入维度是 embedding+1）
        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        # Key/Value 来自编码器输出的所有节点
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # 多头结果合并回 embedding 维度
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # 缓存的 Key（所有城市的编码，编码器算好后存这里）
        self.v = None  # 缓存的 Value（同上）
        self.single_head_key = None  # 缓存的单头 Key（用于最后算概率分数）
        # self.q1 = None  # 旧版 Attention Model 缓存的上下文查询（已弃用）
        # self.q2 = None  # 旧版 Attention Model 缓存的查询（已弃用）

    def set_kv(self, encoded_nodes):
        """
        编码器算完后调用一次：把城市编码转成 K、V 缓存起来，
        解码时每一步都能直接使用，不用重复计算。
        """
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        # 多头注意力用的 K 和 V
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # 形状: (batch, head_num, problem+1, qkv_dim)

        # 单头注意力用的 Key（直接转置，保持原始嵌入，不再降维）
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # 形状: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        """
        旧版 Attention Model 的接口（已弃用，仅保留注释代码）。
        原本用于设置“整图平均”作为上下文 Query。
        """
        # encoded_q.shape: (batch, n, embedding)  # n 可以是 1 或 pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # 形状: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        """
        旧版 Attention Model 的接口（已弃用，仅保留注释代码）。
        原本用于设置“第一个访问的城市”作为上下文 Query。
        """
        # encoded_q.shape: (batch, n, embedding)  # n 可以是 1 或 pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # 形状: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, ninf_mask):
        """
        解码一步：
            输入：
                encoded_last_node : 当前所在城市的嵌入 (batch, pomo, embedding)
                load              : 当前剩余容量 (batch, pomo)
                ninf_mask         : 非法城市掩码 (batch, pomo, problem)
                                    合法位置是 0，非法位置是负无穷
            输出：
                probs : 下一步去每个城市的概率 (batch, pomo, problem)
        """
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)  # 注意这里只有 problem 个客户，
        #                 因为仓库（索引 0）始终可选，不需要掩码

        head_num = self.model_params['head_num']

        # ---- 第 1 部分：多头注意力 ----
        #######################################################
        # 把剩余容量拼到当前城市嵌入上，作为 Query 的输入特征
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # 形状: (batch, pomo, EMBEDDING_DIM+1)

        # 生成 Query（只有“当前节点”一个查询位置）
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # 形状: (batch, head_num, pomo, qkv_dim)

        # 旧版写法是三个 Query 相加（图平均 + 首节点 + 当前节点），
        # POMO 简化为只用当前节点这一个 Query
        # q = self.q1 + self.q2 + q_last
        q = q_last
        # 形状: (batch, head_num, pomo, qkv_dim)

        # 多头注意力：当前节点去关注所有城市，得到上下文向量
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # 形状: (batch, pomo, head_num*qkv_dim)

        # 合并多头结果，回到 embedding 维度
        mh_atten_out = self.multi_head_combine(out_concat)
        # 形状: (batch, pomo, embedding)

        # ---- 第 2 部分：单头注意力（算概率） ----
        #######################################################
        # 用上下文向量和每个城市的嵌入做点积，得到“匹配分数”
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # 形状: (batch, pomo, problem+1) —— 注意这里其实包含仓库（problem+1 列）

        # 缩放：除以嵌入维度的平方根，防止分数过大导致 softmax 梯度消失
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        # 截断（logit clipping）：用 tanh 把分数限制在 [-C, C]，让概率更平滑
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # 形状: (batch, pomo, problem+1)

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        # 加上掩码：非法城市（超出容量等）加上负无穷，softmax 后概率变成 0
        score_masked = score_clipped + ninf_mask

        # softmax 转成概率分布
        probs = F.softmax(score_masked, dim=2)
        # 形状: (batch, pomo, problem+1)

        return probs


########################################
# 神经网络子模块 / 工具函数
########################################

def reshape_by_heads(qkv, head_num):
    """
    把 (batch, n, head_num*key_dim) 重排成多头格式 (batch, head_num, n, key_dim)。
    这样每个头可以独立做注意力。
    """
    # q.shape: (batch, n, head_num*key_dim)  # n 可以是 1 或 PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    # 最后一维拆成 (head_num, key_dim)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # 形状: (batch, n, head_num, key_dim)

    # 把 head_num 挪到第 2 维
    q_transposed = q_reshaped.transpose(1, 2)
    # 形状: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """
    标准多头注意力实现：
        1. 分数 = Q 和 K 的点积
        2. 除以 sqrt(key_dim) 缩放
        3. 加上可选掩码（把非法位置变成负无穷）
        4. softmax 得到权重
        5. 权重和 V 加权求和，得到输出

    支持两种掩码：
        rank2_ninf_mask : (batch, problem)         —— 所有查询位置共用同一个掩码
        rank3_ninf_mask : (batch, group, problem)  —— 每个查询位置各自的掩码
    """
    # q 形状: (batch, head_num, n, key_dim)  # n 可以是 1 或 PROBLEM_SIZE
    # k, v 形状: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)   # 被注意的城市数量

    # Q 与 K 的点积：衡量“匹配程度”
    score = torch.matmul(q, k.transpose(2, 3))
    # 形状: (batch, head_num, n, problem)

    # 缩放，防止点积过大
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    # 根据掩码类型把掩码广播到所有头上
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    # softmax 得到每个头、每个位置的注意力权重
    weights = nn.Softmax(dim=3)(score_scaled)
    # 形状: (batch, head_num, n, problem)

    # 加权求和 V，得到注意力输出
    out = torch.matmul(weights, v)
    # 形状: (batch, head_num, n, key_dim)

    # 把多头重新合并回原来的形状
    out_transposed = out.transpose(1, 2)
    # 形状: (batch, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # 形状: (batch, n, head_num*key_dim)

    return out_concat

