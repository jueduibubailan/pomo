#机器环境配置
DEBUG_MODE = False #调试开关（在第87行使用），False会使用gpu并进行完整训练，True 时 _set_debug_mode() 会把 epochs、train_episodes、train_batch_size 临时改为 2、4、2
USE_CUDA = not DEBUG_MODE #是否使用CUDA，这个值的意思其实就是True
CUDA_DEVICE_NUM = 0 #使用GPU的编号

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))#把工作目录切到 train_n100.py 所在文件夹，保证后续相对路径
sys.path.insert(0, '..')#把上一级目录（CVRP/）加入模块搜索路径，使 `from CVRProblemDef import ...` 能找到 CVRProblemDef.py
sys.path.insert(0, '../..')#把上上级目录（NEW_py_ver/）加入搜索路径，使 `from utils.utils import ...` 能找到工具模块。

import logging
from utils import create_logger,copy_all_src
from CVRPTrainer import CVRPTrainer as Trainer

#环境配置
env_params = {
    'problem_size': 100,#客户节点数N，但不包括仓库节点0
    'pomo_size': 100,#pomo并行起点数：每个样本同时维护 100 条候选路线，以不同客户开头
}

#模型结构配置
model_params = {
    'embedding_dim': 128,#节点嵌入向量维度
    'sqrt_embedding_dim': 128**(1/2),#解码打分缩放因子 sqrt(d)
    'encoder_layer_num': 6,#编码器堆叠的层数
    'qkv_dim': 16,#多头注意力中每个头的维度
    'head_num': 8,#注意力头数（8 x 16 = 128 = embedding_dim）
    'logit_clipping': 10,#logit 裁剪系数 c，打分公式 score = c * tanh(score / sqrt(d))
    'ff_hidden_dim': 512,#前馈网络隐藏层维度
    'eval_type': 'argmax',#非训练模式下的解码方式：argmax 贪心选择；改为 'softmax' 则按概率采样
}

#优化器与调度器
optimizer_params = {
    'optimizer': {
        'lr': 1e-4,#Adam 初始学习率
        'weight_decay': 1e-6#Adam 权重衰减
    },
    'scheduler': {
        'milestones': [8001, 8051],#学习率衰减的 epoch 点
        'gamma': 0.1#到达 milestone 时学习率乘以 0.1
    }
}

#训练控制
trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,#是否使用 GPU 及 GPU 编号
    # 'epochs': 8100,#训练总 epoch 数
    'epochs': 10, #测试训练epoch数量
    'train_episodes': 10 * 1000,#每个 epoch 处理的 episode 总数
    'train_batch_size': 64,#每批样本数（最后一个 batch 自动取剩余量）
    'prev_model_path': None,#预留字段，本配置未使用
    'logging': {
        'model_save_interval': 500,#每 500 个 epoch 保存一次模型 checkpoint
        'img_save_interval': 500,#每 500 个 epoch 保存一次得分/损失曲线图

        #绘制 train_score / train_loss 曲线的样式配置
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    #是否从断点续训（True 时需配置 path 与 epoch）
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_CVRP20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 2000,  # epoch version of pre-trained model to laod.

    }
}

#日志配置
logger_params = {
    'log_file': {
        'desc': 'train_cvrp_n100_with_instNorm',#结果目录后缀，最终目录形如 result/时间戳_train_cvrp_n100_with_instNorm
        'filename': 'run_log'#日志文件名（脚本原始写法，不带扩展名）
    }
}

def main():
    if DEBUG_MODE:#开头设置为false，就是在这里判断
        _set_debug_mode()
    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)
    copy_all_src(trainer.result_folder)

    trainer.run()

def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2

#遍历 globals() 中所有以 params 结尾的字典
# （env_params / model_params / optimizer_params / trainer_params / logger_params），
# 逐项写入日志，便于回溯本次运行配置。
def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

if __name__ == '__main__':
    main()