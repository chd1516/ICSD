"""
SGDP Optimizer Implementation copied from https://github.com/clovaai/AdamP/blob/master/adamp/sgdp.py

Paper: `Slowing Down the Weight Norm Increase in Momentum-based Optimizers` - https://arxiv.org/abs/2006.08217
Code: https://github.com/clovaai/AdamP

Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import math

class SGDP(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eps=1e-8, delta=0.1, wd_ratio=0.1):
        """
        初始化SGDP优化器，扩展自SGD，增加了处理平庸鞍点的能力。

        参数:
        - params: 一个迭代器或可迭代对象，包含模型的参数。
        - lr: 学习率，控制模型参数更新的速度。
        - momentum: 动量值，用于加速学习过程。
        - dampening: 动量的阻尼因子，减少动量的影响。
        - weight_decay: 权重衰减，相当于在损失函数中添加一个L2正则化项。
        - nesterov: 是否使用Nesterov动量，一种更先进的动量方法。
        - eps: 用于数值稳定性的小量。
        - delta: 控制动量项更新幅度的参数。
        - wd_ratio: 权重衰减与学习率的比例。
        """
        # 定义优化器的默认参数字典，用于统一配置
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, delta=delta, wd_ratio=wd_ratio)
        # 调用父类构造方法，初始化优化器
        super(SGDP, self).__init__(params, defaults)


    def _channel_view(self, x):
        """
        重塑输入张量的形状，以适应后续的神经网络层。

        该方法主要用于调整输入数据的维度，使其能够适应后续的全连接层或其他层类型。
        它会保持输入数据中样本数量的不变，同时将每个样本的特征一维化。

        参数:
        x: 输入的张量，通常是一个多维张量（如图像数据）。

        返回值:
        一个经过重塑的张量，其形状变为(batch_size, -1)，其中batch_size为输入数据的样本数量，
        -1表示张量中除了batch_size之外的所有维度会被乘到一起，形成一个新的维度。
        """
        # 重塑输入张量的形状，以适应后续的神经网络层。
        return x.view(x.size(0), -1)


    def _layer_view(self, x):
        """
        调整输入数据的维度。

        此方法主要用于神经网络中，用来调整输入数据的形状，使其能够更好地与后续层匹配。
        在此情况下，输入数据被视图为单个批量，每个批量包含多个元素。

        参数:
        x : Tensor
            输入的数据，通常是一个从上一层神经网络传递过来的多维度张量。

        返回:
        Tensor
            调整后的数据，形状为(1, -1)，表示每个批量包含多个展平的元素。
        """
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        """
        计算两个向量或向量集合之间的余弦相似度。

        参数:
        - x: 第一个向量或向量集合。
        - y: 第二个向量或向量集合。
        - eps: 一个较小的值，用于避免除以零的情况。
        - view_func: 用于调整向量视图的函数，以确保x和y具有兼容的形状进行计算。

        返回:
        - 一个表示相似度的标量或向量，值的范围在-1到1之间，其中1表示完全相似，-1表示完全不相似。
        """
        # 使用view_func调整x和y的形状，以确保它们具有兼容的形状进行后续计算
        x = view_func(x)
        y = view_func(y)

        # 计算x和y的范数，并添加一个较小的eps值以避免除以零的情况
        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)

        # 计算x和y之间的点积
        dot = (x * y).sum(dim=1)

        # 计算余弦相似度并返回
        return dot.abs() / x_norm / y_norm

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        """
        执行投影操作以确保梯度和参数之间的方向一致性，并调整扰动以维持指定的方向性约束。

        参数:
        - p: 模型参数张量。
        - grad: 梯度张量。
        - perturb: 待应用到参数的扰动张量。
        - delta: 方向性约束的阈值。
        - wd_ratio: 权重衰减比率，用于调整扰动的幅度。
        - eps: 小的正值，用于数值稳定性。

        返回:
        - perturb: 调整后的扰动张量。
        - wd: 实际使用的权重衰减，可能根据条件改变。
        """

        # 初始化权重衰减为1，后续可能根据条件修改。
        wd = 1

        # 构建expand_size以匹配参数p的形状，但仅扩展通道维度。
        expand_size = [-1] + [1] * (len(p.shape) - 1)

        # 遍历两种视图函数，用于不同维度上的方向性检查。
        for view_func in [self._channel_view, self._layer_view]:

            # 计算梯度和参数之间的余弦相似度。
            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            # 检查最大余弦相似度是否低于阈值，如果是，则表示方向性约束被违反。
            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                # 将参数投影到单位超球面上。
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                # 调整扰动以保持方向性约束。
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                # 设置权重衰减为指定的比率。
                wd = wd_ratio

                # 方向性调整完成后，返回调整后的扰动和权重衰减。
                return perturb, wd

        # 如果没有进行调整，则返回原始的扰动和权重衰减。
        return perturb, wd


    def step(self, closure=None):
        """
        执行优化器的单步更新。

        参数:
            closure (callable, 可选): 一个闭合函数，用于计算并返回损失。默认为 None。

        返回:
            loss (float, optional): 如果提供了 closure 函数，返回损失值。
        """
        loss = None
        # 如果提供了闭合函数，调用它以计算损失
        if closure is not None:
            loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            # 获取当前参数组的超参数
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # 遍历参数组中的每个参数
            for p in group['params']:
                # 如果参数没有梯度，则跳过该参数
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 如果状态为空，则初始化动量
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                # SGD 更新
                buf = state['momentum']
                buf.mul_(momentum).add_(1 - dampening, grad)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # 投影操作（特定于该优化器的自定义操作）
                wd_ratio = 1
                if len(p.shape) > 1:
                    d_p, wd_ratio = self._projection(p, grad, d_p, group['delta'], group['wd_ratio'], group['eps'])

                # 权重衰减
                if weight_decay != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio / (1-momentum))

                # 参数更新
                p.data.add_(-group['lr'], d_p)

        return loss
