""" RMSProp modified to behave like Tensorflow impl

Originally cut & paste from PyTorch RMSProp
https://github.com/pytorch/pytorch/blob/063946d2b3f3f1e953a2a3b54e0b34f1393de295/torch/optim/rmsprop.py
Licensed under BSD-Clause 3 (ish), https://github.com/pytorch/pytorch/blob/master/LICENSE

Modifications Copyright 2020 Ross Wightman
"""

import torch
from torch.optim import Optimizer


class RMSpropTF(Optimizer):
    """Implements RMSprop algorithm (TensorFlow style epsilon)

    NOTE: This is a direct cut-and-paste of PyTorch RMSprop with eps applied before sqrt
    and a few other modifications to closer match Tensorflow for matching hyper-params.

    Noteworthy changes include:
    1. Epsilon applied inside square-root
    2. square_avg initialized to ones
    3. LR scaling of update accumulated in momentum buffer

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing (decay) constant (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_decay (bool, optional): decoupled weight decay as per https://arxiv.org/abs/1711.05101
        lr_in_momentum (bool, optional): learning rate scaling is included in the momentum buffer
            update as per defaults in Tensorflow

    """

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0., centered=False,
                 decoupled_decay=False, lr_in_momentum=True):
        """
        RMSpropTF类的初始化方法。

        参数:
            params (iterable): 包含所有需要进行优化的模型参数的迭代器。
            lr (float, optional): 学习率。默认为1e-2。
            alpha (float, optional): RMSprop的平滑系数。默认为0.9。
            eps (float, optional): 添加到分母的常数项，用于数值稳定性。默认为1e-10。
            weight_decay (float, optional): 权重衰减（L2正则化强度）。默认为0。
            momentum (float, optional): 动量项。默认为0。
            centered (bool, optional): 是否计算梯度的均值。默认为False。
            decoupled_decay (bool, optional): 是否使用Decoupled权重衰减。默认为False。
            lr_in_momentum (bool, optional): 是否在动量中包含学习率。默认为True。
        """
        # 验证输入参数的有效性
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        # 将所有默认参数值存储在字典中
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay,
                        decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)

        # 调用父类的初始化方法，传入参数和默认值
        super(RMSpropTF, self).__init__(params, defaults)


    def __setstate__(self, state):
        # 调用父类的__setstate__方法来恢复优化器的状态
        super(RMSpropTF, self).__setstate__(state)
        # 为每个参数组设置默认的动量和集中参数
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)


    def step(self, closure=None):
        """
        执行单个优化步骤。

        参数：
            closure (callable, optional): 重新评估模型并返回损失的闭包。
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop 不支持稀疏梯度')
                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(p.data)  # PyTorch 初始化为零
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if 'decoupled_decay' in group and group['decoupled_decay']:
                        p.data.add_(-group['weight_decay'], p.data)
                    else:
                        grad = grad.add(group['weight_decay'], p.data)

                # Tensorflow 操作顺序更新平方平均值
                square_avg.add_(one_minus_alpha, grad.pow(2) - square_avg)
                # square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)  # PyTorch 原始操作

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(one_minus_alpha, grad - grad_avg)
                    # grad_avg.mul_(alpha).add_(1 - alpha, grad)  # PyTorch 原始操作
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(group['eps']).sqrt_()  # eps 移到 sqrt 内部
                else:
                    avg = square_avg.add(group['eps']).sqrt_()  # eps 移到 sqrt 内部

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    # Tensorflow 在动量缓冲区中累积 LR 缩放
                    if 'lr_in_momentum' in group and group['lr_in_momentum']:
                        buf.mul_(group['momentum']).addcdiv_(group['lr'], grad, avg)
                        p.data.add_(-buf)
                    else:
                        # PyTorch 通过 LR 缩放参数更新
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss
