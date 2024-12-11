import torch
from torch.optim import Optimizer


class Nadam(Optimizer):
    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).

    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        schedule_decay (float, optional): momentum schedule decay (default: 4e-3)

    __ http://cs229.stanford.edu/proj2015/054_report.pdf
    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf

        Originally taken from: https://github.com/pytorch/pytorch/pull/1408
        NOTE: Has potential issues but does work well on some problems.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=4e-3):
        """
        Nadam优化器的构造函数
        Nadam是Nesterov-adaptive Adam的缩写，结合了Nesterov动量和Adam优化算法的特点
        该优化器在训练神经网络时可以动态调整学习率

        参数:
        - params: 一个迭代器，包含模型的参数，用于优化
        - lr: 学习率，控制参数更新的步长，默认值为2e-3
        - betas: 一个二元元组，包含用于计算梯度的指数衰减率，默认值为(0.9, 0.999)
        - eps: 用于数值稳定性的一个小常数，默认值为1e-8
        - weight_decay: 权重衰减，用于正则化，默认值为0
        - schedule_decay: 学习率衰减率，默认值为4e-3
        """
        # 将参数默认值封装成字典，便于后续使用和管理
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, schedule_decay=schedule_decay)
        # 调用父类构造器，初始化优化器
        super(Nadam, self).__init__(params, defaults)


    def step(self, closure=None):
        """
        执行单个优化步骤。

        参数：
            closure (可调用对象，可选)：重新评估模型并返回损失的闭包。
        """
        loss = None
        # 如果提供了闭包，重新计算损失
        if closure is not None:
            loss = closure()

        # 遍历每个参数组和参数，进行优化更新
        for group in self.param_groups:
            for p in group['params']:
                # 如果参数没有梯度，则跳过该参数
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 如果状态为空，进行状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['m_schedule'] = 1.
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                # 计算动量调度
                m_schedule = state['m_schedule']
                schedule_decay = group['schedule_decay']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                state['step'] += 1
                t = state['step']

                # 如果weight_decay不为0，梯度中加入权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # 计算当前和下一个时间步的动量缓存
                momentum_cache_t = beta1 * (1. - 0.5 * (0.96 ** (t * schedule_decay)))
                momentum_cache_t_1 = beta1 * (1. - 0.5 * (0.96 ** ((t + 1) * schedule_decay)))
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                state['m_schedule'] = m_schedule_new

                # 更新指数移动平均值和平方值
                exp_avg.mul_(beta1).add_(1. - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1. - beta2, grad, grad)
                exp_avg_sq_prime = exp_avg_sq / (1. - beta2 ** t)
                denom = exp_avg_sq_prime.sqrt_().add_(eps)

                # 更新参数
                p.data.addcdiv_(-group['lr'] * (1. - momentum_cache_t) / (1. - m_schedule_new), grad, denom)
                p.data.addcdiv_(-group['lr'] * momentum_cache_t_1 / (1. - m_schedule_next), exp_avg, denom)

        # 返回计算的损失
        return loss
