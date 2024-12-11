""" Nvidia NovoGrad Optimizer.
Original impl by Nvidia from Jasper example:
    - https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper
Paper: `Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks`
    - https://arxiv.org/abs/1905.11286
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class NvNovoGrad(Optimizer):
    """
    Implements Novograd algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0.98))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=0, grad_averaging=False, amsgrad=False):
        """
        NvNovoGrad优化器的构造函数。

        参数:
        params -- 一个迭代器或序列，包含所有需要进行优化的模型参数或者包含这些参数的字典。
        lr (可选) -- 学习率，控制参数更新的步长，默认为1e-3。
        betas (可选) -- 用于计算运行平均值的系数元组，分别对应于第一和第二时刻，默认为(0.95, 0.98)。
        eps (可选) -- 用于数值稳定性的一个小常数，默认为1e-8。
        weight_decay (可选) -- 权重衰减（L2正则化强度），默认为0。
        grad_averaging (可选) -- 是否使用梯度平均， 默认为False。
        amsgrad (可选) -- 是否使用amsgrad算法来保持更精确的最大二阶矩估计，默认为False。

        返回:
        None
        """
        # 验证学习率的有效性
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        # 验证epsilon的有效性
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        # 验证beta参数的有效性
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        # 将所有参数封装成一个字典
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        amsgrad=amsgrad)

        # 调用父类构造器，初始化优化器
        super(NvNovoGrad, self).__init__(params, defaults)


    def __setstate__(self, state):
        """
        当对象从序列化状态恢复时，此方法被调用以恢复对象的状态。

        参数:
        state: 一个包含对象状态的字典。

        返回:
        无
        """
        # 调用父类的__setstate__方法来恢复基础状态
        super(NvNovoGrad, self).__setstate__(state)

        # 遍历参数组，为没有明确指定'amsgrad'的组设置默认值为False
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """执行单个优化步骤。

        参数：
            closure (可调用对象, 可选)：重新评估模型并返回损失的闭包。
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
                    raise RuntimeError('不支持稀疏梯度。')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 梯度值的指数移动平均
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 平方梯度值的指数移动平均
                    state['exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)
                    if amsgrad:
                        # 保持所有平方梯度值的指数移动平均的最大值
                        state['max_exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # 保持至今所有二阶矩运行平均的最大值
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # 使用最大值规范化梯度的运行平均
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                grad.div_(denom)
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                if group['grad_averaging']:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(-group['lr'], exp_avg)

        return loss

