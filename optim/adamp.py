"""
AdamP Optimizer Implementation copied from https://github.com/clovaai/AdamP/blob/master/adamp/adamp.py

Paper: `Slowing Down the Weight Norm Increase in Momentum-based Optimizers` - https://arxiv.org/abs/2006.08217
Code: https://github.com/clovaai/AdamP

Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import math

class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        """
        AdamP优化器的初始化函数。

        参数:
        - params: 一个迭代器，通常为模型的参数，用于优化。
        - lr (float): 学习率，控制参数更新的速度，默认为1e-3。
        - betas (Tuple[float, float]): Adam算法的一阶和二阶矩估计的衰减率，默认为(0.9, 0.999)。
        - eps (float): 用于数值稳定性的一个非常小的值，默认为1e-8。
        - weight_decay (float): 权重衰减，用于正则化，默认为0。
        - delta (float): AdamP算法的局部动量阈值，默认为0.1。
        - wd_ratio (float): AdamP算法的权重衰减与学习率的比例，默认为0.1。
        - nesterov (bool): 是否启用Nesterov动量，默认为False。

        返回:
        无
        """
        # 将所有参数及其默认值存储在一个字典中，便于后续使用和读取。
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        # 调用父类（优化器）的初始化方法，设置参数和默认值。
        super(AdamP, self).__init__(params, defaults)


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
                # 调整扰动以保持与参数的方向一致性。
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                # 设置权重衰减为指定的比率。
                wd = wd_ratio

                # 方向性调整完成后，返回调整后的扰动和权重衰减。
                return perturb, wd

        # 如果没有进行方向性调整，返回原始扰动和权重衰减。
        return perturb, wd


    def step(self, closure=None):
        """
        执行优化器的单步更新。

        参数:
            closure (callable, 可选): 一个重新计算损失的闭包函数，用于获取新的损失值。

        返回:
            float 或 None: 如果提供了闭包函数，返回新的损失值，否则为 None。
        """
        loss = None
        # 如果提供了闭包函数，调用它以重新计算损失
        if closure is not None:
            loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            # 遍历组中的每个参数
            for p in group['params']:
                # 如果参数没有梯度，则跳过该参数
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # 如果状态为空，则进行状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam 算法的实施
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # 根据是否使用 Nesterov 加速来计算参数更新量
                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # 执行投影操作，以保持参数在所需的范围内
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # 执行权重衰减
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # 更新参数
                p.data.add_(-step_size, perturb)

        return loss
