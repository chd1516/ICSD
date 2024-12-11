"""RAdam Optimizer.
Implementation lifted from: https://github.com/LiyuanLucasLiu/RAdam
Paper: `On the Variance of the Adaptive Learning Rate and Beyond` - https://arxiv.org/abs/1908.03265
"""
import math
import torch
from torch.optim.optimizer import Optimizer, required


class RAdam(Optimizer):
    """
    RAdam优化器类，继承自Optimizer基类。

    该类实现了Rectified Adam（RAdam）优化算法，这是一种自适应学习率的优化方法，
    用于在训练深度学习模型时更新权重参数。RAdam相比传统的Adam优化器，增加了对
    适应性权重调整的改进，特别是在训练初期的性能表现上。

    参数:
    - params: 可迭代对象或字典，定义了要优化的模型参数或参数组。
    - lr (float, 可选): 学习率，控制参数更新的步长，默认为1e-3。
    - betas (Tuple[float, float], 可选): Adam算法中使用的两个指数衰减率，
      用于计算梯度的一阶矩估计（均值）和二阶矩估计（方差），默认为(0.9, 0.999)。
    - eps (float, 可选): 添加到分母上的数值稳定项，防止除以零错误，默认为1e-8。
    - weight_decay (float, 可选): 权重衰减系数，用于L2正则化，防止模型过拟合，默认为0。

    返回:
    - None
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # 将优化器的默认参数构建成字典，便于后续使用
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # 初始化优化器时，为参数更新过程中所需的缓冲区分配空间
        # 这里为简化示例，仅预分配了10个缓冲区，实际使用时应根据需要动态调整
        self.buffer = [[None, None, None] for ind in range(10)]

        # 调用父类Optimizer的构造方法，初始化参数和默认配置
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        # 在RAdam类中恢复状态
        super(RAdam, self).__setstate__(state)


    def step(self, closure=None):
        """
        执行优化器的单步更新。

        参数:
            closure (callable, optional): 一个重新计算损失的闭包函数，用于在进行参数更新后重新计算梯度。默认为None。

        返回:
            float: 如果提供了闭包函数，返回损失值，否则返回None。
        """
        # 初始化损失值为None
        loss = None
        # 如果提供了闭包函数，调用它以计算损失
        if closure is not None:
            loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            # 遍历参数组中的所有参数
            for p in group['params']:
                # 如果参数没有梯度，则跳过该参数
                if p.grad is None:
                    continue
                # 将参数的梯度转换为float类型
                grad = p.grad.data.float()

                # 检查梯度是否为稀疏，如果是，抛出异常
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                # 将参数数据转换为float类型
                p_data_fp32 = p.data.float()

                # 获取参数的状态
                state = self.state[p]

                # 如果状态为空，初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    # 确保状态中的平均值和平方平均值与参数数据类型匹配
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                # 从状态中获取指数移动平均值和平方平均值
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # 获取参数组的beta1和beta2值
                beta1, beta2 = group['betas']

                # 更新指数移动平方平均值
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # 更新指数移动平均值
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # 更新步骤计数
                state['step'] += 1
                # 获取或计算缓冲区中的N_sma和步长
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # 根据N_sma值计算步长
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # 应用权重衰减
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # 根据N_sma值选择更新参数的方法
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                # 将更新后的参数数据复制回参数
                p.data.copy_(p_data_fp32)

        # 返回计算得到的损失值
        return loss


class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        初始化PlainRAdam优化器。

        参数:
            params (iterable): 包含模型参数的迭代器。
            lr (float, 可选): 学习率。默认为1e-3。
            betas (Tuple[float, float], 可选): Adam算法的beta1和beta2参数，用于指数衰减的平均值和方差的计算。默认为(0.9, 0.999)。
            eps (float, 可选): 用于数值稳定性的一个非常小的值。默认为1e-8。
            weight_decay (float, 可选): 权重衰减（L2正则化）。默认为0。

        返回:
            None
        """
        # 将参数和它们的默认值组合成字典，以便于传递给父类的初始化方法
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # 调用父类（优化器）的初始化方法，传入参数和它们的默认值
        super(PlainRAdam, self).__init__(params, defaults)


    def __setstate__(self, state):
        # 在实例从pickle过程中，恢复实例的状态
        # 此方法对于实现自定义的实例状态加载逻辑非常重要
        super(PlainRAdam, self).__setstate__(state)


    def step(self, closure=None):
        """
        执行优化器的单步更新。

        参数:
            closure (callable, optional): 一个重新计算损失函数的闭包，用于在进行更新前重新计算梯度。

        返回:
            float: 如果提供了闭包，返回损失值；否则返回None。
        """

        # 调用闭包以计算损失，如果闭包被提供的话
        loss = None
        if closure is not None:
            loss = closure()

        # 遍历每个参数组，对每个组内的参数执行更新
        for group in self.param_groups:
            for p in group['params']:
                # 如果参数没有梯度，则跳过该参数
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                # RAdam不支持稀疏梯度
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                # 获取参数的状态，用于存储优化器的具体信息
                state = self.state[p]

                # 如果状态为空，则初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    # 确保状态变量与参数数据类型匹配
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # 更新指数移动平均值和指数移动平方平均值
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # 更新步骤计数器
                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # 应用权重衰减，如果设置了的话
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # 根据N_sma的值选择更新步长的方式
                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                            N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)

                # 将更新后的参数数据复制回参数
                p.data.copy_(p_data_fp32)

        # 返回计算的损失，如果有的话
        return loss
