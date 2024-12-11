""" AdamW Optimizer
Impl copied from PyTorch master
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        """
        AdamW优化器的构造函数
        与传统的Adam优化器相比，AdamW在权重衰减的处理上有所不同，能够更有效地改进模型的泛化能力
        参数:
            params (iterable): 包含所有需要进行优化的模型参数的迭代器
            lr (float, optional): 学习率, 控制参数更新的幅度. 默认为1e-3
            betas (Tuple[float, float], optional): Adam算法中使用的两个指数衰减率. 默认为(0.9, 0.999)
            eps (float, optional): 用于数值稳定性的一个非常小的值. 默认为1e-8
            weight_decay (float, optional): 权重衰减系数, 相当于L2正则化的强度. 默认为1e-2
            amsgrad (bool, optional): 是否使用AMSGrad算法，该算法对Adam算法进行了改进. 默认为False
        """
        # 验证学习率是否为非负数，否则抛出异常
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        # 验证epsilon值是否为非负数，否则抛出异常
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        # 验证beta参数是否在合理范围内，否则抛出异常
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        # 将所有参数配置保存到defaults字典中
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        # 调用父类构造器，初始化优化器
        super(AdamW, self).__init__(params, defaults)


    def __setstate__(self, state):
        # 当对象从序列化状态恢复时，调用这个方法来恢复对象的状态
        # 通过超类方法恢复基础状态
        super(AdamW, self).__setstate__(state)
        # 遍历参数组，为每个优化器的参数组设置默认的'amsgrad'值
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    def step(self, closure=None):
        """
        执行单个优化步骤。

        参数：
            closure (可调用对象，可选)：重新评估模型并返回损失的闭包。
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 执行步骤权重衰减
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # 执行优化步骤
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam不支持稀疏梯度，请考虑使用SparseAdam')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 梯度值的指数移动平均
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 平方梯度值的指数移动平均
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # 维护所有平方梯度值的指数移动平均的最大值
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 衰减第一和第二矩的运行平均系数
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # 维护至今所有第二矩运行平均的最大值
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # 使用最大值对梯度的运行平均进行标准化
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

