"""NovoGrad Optimizer.
Original impl by Masashi Kimura (Convergence Lab): https://github.com/convergence-lab/novograd
Paper: `Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks`
    - https://arxiv.org/abs/1905.11286
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class NovoGrad(Optimizer):
    def __init__(self, params, grad_averaging=False, lr=0.1, betas=(0.95, 0.98), eps=1e-8, weight_decay=0):
        """
        NovoGrad优化器的构造函数

        参数:
        params (iterable): 一个包含需要优化的模型参数的iterable。
        grad_averaging (bool, optional): 是否启用梯度平均。默认为False。
        lr (float, optional): 学习率。默认为0.1。
        betas (tuple, optional): 用于计算运行平均值的系数。默认为(0.95, 0.98)。
        eps (float, optional): 用于数值稳定性的一个小常数。默认为1e-8。
        weight_decay (float, optional): 权重衰减（L2正则化）。默认为0。

        返回:
        None
        """
        # 将优化器的配置参数保存到defaults字典中，便于后续使用和调整
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # 调用父类构造器，传入参数和默认配置
        super(NovoGrad, self).__init__(params, defaults)

        # 保存学习率、beta系数、epsilon和权重衰减作为实例变量，以便在优化过程中使用
        self._lr = lr
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._eps = eps
        self._wd = weight_decay

        # 是否启用梯度平均的标志，影响优化过程中的梯度更新方式
        self._grad_averaging = grad_averaging

        # 用于内部机制，标记动量是否已经初始化
        self._momentum_initialized = False


    def step(self, closure=None):
        """
        执行优化器的单步更新。

        参数:
            closure (callable, 可选): 一个重新计算损失的闭包函数，用于获取梯度并执行更新。

        返回:
            float: 如果提供了闭包函数，返回损失值；否则返回None。
        """
        # 初始化损失值为None
        loss = None

        # 如果提供了闭包函数，调用它以获取损失，并更新参数
        if closure is not None:
            loss = closure()

        # 如果动量尚未初始化，则初始化动量相关的状态
        if not self._momentum_initialized:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data

                    # 如果梯度是稀疏的，抛出错误，因为NovoGrad不支持稀疏梯度
                    if grad.is_sparse:
                        raise RuntimeError('NovoGrad does not support sparse gradients')

                    # 计算梯度的范数的平方
                    v = torch.norm(grad)**2
                    # 计算动量
                    m = grad/(torch.sqrt(v) + self._eps) + self._wd * p.data
                    # 初始化状态
                    state['step'] = 0
                    state['v'] = v
                    state['m'] = m
                    state['grad_ema'] = None
            # 将动量标记为已初始化
            self._momentum_initialized = True

        # 遍历参数组，更新每个参数
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['step'] += 1

                # 从状态中获取必要的变量
                step, v, m = state['step'], state['v'], state['m']
                grad_ema = state['grad_ema']

                # 获取梯度
                grad = p.grad.data
                # 更新梯度的指数移动平均
                g2 = torch.norm(grad)**2
                grad_ema = g2 if grad_ema is None else grad_ema * self._beta2 + g2 * (1. - self._beta2)
                grad *= 1.0 / (torch.sqrt(grad_ema) + self._eps)

                # 如果使用梯度平均，则按比例调整梯度
                if self._grad_averaging:
                    grad *= (1. - self._beta1)

                # 更新梯度的范数的平方
                g2 = torch.norm(grad)**2
                v = self._beta2*v + (1. - self._beta2)*g2
                # 更新动量
                m = self._beta1*m + (grad / (torch.sqrt(v) + self._eps) + self._wd * p.data)
                # 计算偏差修正因子
                bias_correction1 = 1 - self._beta1 ** step
                bias_correction2 = 1 - self._beta2 ** step
                # 计算步长
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # 更新状态
                state['v'], state['m'] = v, m
                state['grad_ema'] = grad_ema
                # 更新参数
                p.data.add_(-step_size, m)

        # 返回损失值
        return loss
