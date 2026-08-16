# 工具函数库（中文注释版）
# 原代码：MIT License, Copyright (c) 2021 Yeong-Dae Kwon（POMO 项目）
# 用途：给训练脚本提供日志、结果保存、统计、计时、绘图等通用功能
#
# 本文件主要包含：
#   1. 结果文件夹管理（get/set_result_folder）
#   2. 日志系统初始化（create_logger）
#   3. 平均值统计（AverageMeter）
#   4. 训练曲线数据记录（LogData）
#   5. 训练时间估算（TimeEstimator）
#   6. 日志曲线绘图与保存（util_save_log_image_with_label 等）
#   7. 复制源码到结果目录（copy_all_src，保证实验可复现）
import json
import logging
import os
import shutil
from datetime import datetime
import pytz
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# 记录进程启动时间（按首尔时区），之后用来给结果文件夹命名
process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
# 结果文件夹路径模板：日期_时间 再加一个可选的描述后缀 {desc}
# 例如：./result/20260816_120000{desc}
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'

def get_result_folder():
    """返回当前结果文件夹路径。"""
    return result_folder
def set_result_folder(folder):
    """把全局的结果文件夹路径改成新值。"""
    global result_folder    # 声明修改的是全局变量
    result_folder = folder

def create_logger(log_file=None):
    """
    初始化日志系统。
    参数 log_file 是一个字典，可包含：
        - 'filepath': 结果文件夹路径（没有就用默认的 result_folder）
        - 'desc'    : 描述文字，会拼到文件夹名后面
        - 'filename': 日志文件名（默认 log.txt）
    日志会同时输出到文件和控制台。
    """
    # 如果没给路径，就用默认结果文件夹
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    # 有描述文字时，把路径模板里的 {desc} 替换成 '_描述'
    if 'desc' in log_file:
        log_file['filepath']= log_file['filepath'].format(desc='_'+log_file['desc'])
    else:
        #没有描述时，{desc} 替换成空字符串
        log_file['filepath'] = log_file['filepath'].format(desc='')

    # 同步更新全局结果文件夹路径
    set_result_folder(log_file['filepath'])
    # 决定日志文件名（默认 log.txt）
    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    # 如果文件夹不存在，就创建它
    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])
    # 日志文件已存在就追加写入（'a'），否则新建（'w'）
    file_mode = 'a' if os.path.isfile(filename) else 'w'

    # 获取全局根 logger，并设置日志级别为 INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)

    # 日志格式：时间、来源文件名、行号、消息内容
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    # 先清掉已有的 handler，避免重复打印日志
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # 添加“写文件”的 handler
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # 添加“输出到控制台”的 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

def copy_all_src(dst_root):
    """
    把当前项目用到的所有源码文件复制到 dst_root/src 目录下。
    目的：实验结束后结果文件夹里保留一份当时的源码，方便复现。
    """
    # 确定“执行目录”：
    # 在 Jupyter notebook 中运行时，用当前工作目录；否则用脚本所在目录
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])
    # 通过 sys.path 前两个路径推断“项目根目录”（home_dir）
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    # 取两个候选路径中较短的（更靠近项目根的那个）作为 home_dir
    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # 目标目录：结果文件夹下的 src
    dst_path = os.path.join(dst_root, 'src')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

        # 遍历所有已经导入的模块
        for item in sys.modules.items():
            key, value = item

            # 只处理有源文件（__file__）的模块
            if hasattr(value, '__file__') and value.__file__:
                src_abspath = os.path.abspath(value.__file__)
                # 关键修复：torch.ops 这类“伪模块”的 __file__ 只是占位符，
                # 文件根本不存在，直接跳过，避免 shutil.copy 报 FileNotFoundError
                if not os.path.exists(src_abspath):
                    continue
                # 如果模块文件位于项目根目录下，就复制它
                if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                    dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                    # 若目标已有同名文件，改成 "name(0).py"、"name(1).py"... 避免覆盖
                    if os.path.exists(dst_filepath):
                        split = list(os.path.splitext(dst_filepath))
                        split.insert(1, '({})')  # 变成 ['.../name', '({})', '.py']
                        filepath = ''.join(split)  # '.../name({}).py'
                        post_index = 0

                        while os.path.exists(filepath.format(post_index)):
                            post_index += 1

                        dst_filepath = filepath.format(post_index)

                    # 真正执行复制
                    shutil.copy(src_abspath, dst_filepath)

class LogData:
    """
    训练曲线数据存储器。
    它把“每个指标（key）”的数据存成若干个 [x, y] 点对，例如：
        data['train_loss'] = [[0, 10.5], [1, 9.8], [2, 9.2], ...]
    其中 x 通常是迭代/回合编号，y 是对应的指标值。
    这样后续可以直接画图，或者把数据保存下来。
    """
    def __init__(self):
        self.keys = set() # 所有指标名（用集合保证不重复）
        self.data = {} # 指标名 -> 点的列表

    def get_raw_data(self):
        """返回原始数据（键集合和全部数据），用于保存/断点续训。"""
        return self.keys, self.data

    def set_raw_data(self, r_data):
        """恢复之前保存的原始数据。"""
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        """
        一次性追加一整段数据。
        - append_all(key, y_list)：只给 y 值，x 自动取 0,1,2,...
        - append_all(key, x_list, y_list)：同时给 x 和 y
        """
        if len(args) == 1:
            # 只给了一组 y 值，x 自动生成 [0, 1, 2, ..., n-1]
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            # 同时给了 x 和 y
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            # 已有这个指标，就把新数据接在后面
            self.data[key].extend(value)
        else:
            # 第一次出现：把 [x列表, y列表] 拼成 [[x0,y0],[x1,y1],...] 的形式
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        """
        追加单个数据点。
        - append(key, y)            ：只给 y，x 自动取当前点个数
        - append(key, (x, y))       ：元组形式 [x, y]
        - append(key, [x, y])       ：列表形式
        - append(key, x, y)         ：两个参数形式
        """
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                # 传入的是单个数值：x 用当前已有点的个数（已有数据则用长度，否则用 0）
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                # 元组 (x, y)
                value = list(args)
            elif type(args) == list:
                # 列表 [x, y]
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            # 两个参数：append(key, x, y)
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            # 已有该指标：直接追加一个点
            self.data[key].append(value)
        else:
            # 第一次出现：初始化成只含一个点的列表
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        """返回某个指标最近一个数据点 [x, y]；不存在返回 None。"""
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        """判断某个指标是否存在。"""
        return key in self.keys

    def get(self, key):
        """
        返回某个指标的 y 值列表（只取数值部分，不含 x）。
        实现：把 [[x0,y0],[x1,y1],...] 按列拆成 x 列和 y 列，取 y 列。
        """
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        """
        返回某个指标的 (x列表, y列表)。
        start_idx：可指定从某个 x 值开始取（找第一个等于 start_idx 的 x）。
        """
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        # 只有一个点时，squeeze 后不是列表，直接返回
        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            # 默认从头开始
            return xs, ys
        elif start_idx in xs:
            # 找到 start_idx 第一次出现的位置，从这里开始截取
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        """返回所有指标名。"""
        return self.keys

class TimeEstimator:
    """
    训练时间估算器。
    根据已经花掉的时间和当前进度，估算剩余时间：
        剩余时间 = 已用时间 / 已完成比例 x 剩余比例
    """

    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')  # 专用日志对象
        self.start_time = time.time()                     # 开始计时
        self.count_zero = 0                               # 用于校正起始计数

    def reset(self, count=1):
        """
        重新开始计时。count 表示本轮从哪个编号开始（如第 1 个 epoch）。
        count_zero 设为 count-1，这样计算时 (count - count_zero) 正好等于已完成数量。
        """
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        """
        计算已用时间和剩余时间（单位：小时）。
        count：当前进度；total：总进度。
        """
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time   # 已经过的时间（秒）
        remain = total - count                       # 还剩多少
        # 按已完成部分推算剩余时间
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0                       # 秒 -> 小时
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        """把时间格式化成易读的字符串：超过 1 小时显示小时，否则显示分钟。"""
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time * 60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time * 60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        """打印当前进度以及预计已用/剩余时间。"""
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))

def util_save_log_image_with_label(result_file_prefix,
                                   img_params,
                                   result_log: LogData,
                                   labels=None):
    """
    把 LogData 画成曲线图并保存为 jpg 图片。

    参数：
        result_file_prefix : 图片保存路径前缀，如 './result/xxx/fig'
        img_params         : 字典，需包含 'json_foldername' 和 'filename'，
                             指向一个 JSON 绘图配置文件（图大小、坐标范围、网格等）
        result_log         : LogData 数据
        labels             : 要画的指标名列表（默认画全部）
    """
    # 确保保存目录存在
    dirname = os.path.dirname(result_file_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # 根据配置和数据画图
    _build_log_image_plt(img_params, result_log, labels)

    # 没有指定标签时，默认用所有指标名做文件名
    if labels is None:
        labels = result_log.get_keys()
    file_name = '_'.join(labels)

    # 保存当前画布为 jpg 并关闭
    fig = plt.gcf()
    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)

def _build_log_image_plt(img_params,
                         result_log: LogData,
                         labels=None):
    """
    内部函数：真正负责画图。
    先从 JSON 配置文件读取图的大小、坐标轴范围、网格等设置，
    再把每个指标画成一条曲线。
    """
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    # 读取 JSON 配置文件（路径相对于本文件所在目录）
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)

    with open(log_image_config_file, 'r') as f:
        config = json.load(f)

    # 按配置创建指定大小的画布
    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)

    # 画出每个指标对应的曲线
    if labels is None:
        labels = result_log.get_keys()
    for label in labels:
        plt.plot(*result_log.getXY(label), label=label)

    # 设置 y 轴范围；配置里是 None 时自动用数据范围
    ylim_min = config['ylim']['min']
    ylim_max = config['ylim']['max']
    if ylim_min is None:
        ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None:
        ylim_max = plt.gca().dataLim.ymax
    plt.ylim(ylim_min, ylim_max)

    # 设置 x 轴范围；配置里是 None 时自动用数据范围
    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None:
        xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None:
        xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)

    # 图例字号设为 18，并添加图例和网格
    plt.rc('legend', **{'fontsize': 18})
    plt.legend()
    plt.grid(config["grid"])

def util_print_log_array(logger, result_log: LogData):
    """
    把 LogData 里所有指标的值打印出来，方便快速查看。
    """
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    # 遍历每个指标名，用 logger 打印它的 y 值列表
    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key + '_list', result_log.get(key)))

class AverageMeter:
    """
    简单的“滑动平均”计数器，用来统计平均指标（如平均损失、平均奖励）。
    原理：维护总和 sum 和样本数 count，平均值 = sum / count。
    """

    def __init__(self):
        self.reset()          # 初始化时把统计归零

    def reset(self):
        """清零所有统计值。"""
        self.sum = 0          # 数值总和
        self.count = 0        # 样本数量

    def update(self, val, n=1):
        """加入一个新值 val，n 表示这个值代表 n 个样本。"""
        self.sum += (val * n) # 总和累加
        self.count += n       # 样本数累加

    @property
    def avg(self):
        """返回当前平均值；样本数为 0 时返回 0，避免除零错误。"""
        return self.sum / self.count if self.count else 0
