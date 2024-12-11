""" Adafactor Optimizer

Lifted from https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

Original header/copyright below.

"""
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import math


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.
    This implementation is based on: `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate depending on the
    *scale_parameter*, *relative_step* and *warmup_init* options.

    To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(self, params, lr=None, eps=1e-30, eps_scale=1e-3, clip_threshold=1.0,
                 decay_rate=-0.8, betas=None, weight_decay=0.0, scale_parameter=True, warmup_init=False):
        """
        Adafactor优化器的构造函数。

        参数:
            params (iterable): 要优化的参数或参数组。
            lr (float, optional): 学习率。如果为None，则使用相对学习率。默认为None。
            eps (float): Adam风格的数值稳定性项。默认为1e-30。
            eps_scale (float): Adafactor特有的额外稳定性项。默认为1e-3。
            clip_threshold (float): 梯度的阈值，用于控制更新大小。默认为1.0。
            decay_rate (float): 梯度的指数衰减率。默认为-0.8。
            betas (Tuple[float, float], optional): Adam风格的衰减率。默认为None。
            weight_decay (float): 权重衰减（L2正则化）。默认为0.0。
            scale_parameter (bool): 是否对学习率进行缩放。默认为True。
            warmup_init (bool): 是否使用warmup初始化。默认为False。

        提示:
            如果warmup_init为True且lr不为None，则会引发ValueError。
        """
        # 判断是否使用相对学习率
        relative_step = lr is None
        # 如果使用warmup初始化且不使用相对学习率，则引发错误
        if warmup_init and not relative_step:
            raise ValueError('warmup_init requires relative_step=True')

        # 如果betas为None，则beta1为None，否则取betas的第一个元素
        beta1 = None if betas is None else betas[0]
        # 构造默认参数字典
        defaults = dict(lr=lr, eps=eps, eps_scale=eps_scale, clip_threshold=clip_threshold, decay_rate=decay_rate,
                        beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        # 调用父类构造方法，初始化优化器
        super(Adafactor, self).__init__(params, defaults)


    @staticmethod
    def _get_lr(param_group, param_state):
        """
        计算并返回指定参数组的学习率。

        该方法首先判断参数组是否采用相对步长。如果采用，将根据当前步数和参数状态计算学习率。
        学习率的计算考虑了预热初始化和平滑参数的影响，以确保在训练初期学习率逐渐增加，
        避免直接使用过大的学习率导致的训练不稳定。

        参数:
            - param_group: 参数组，包含训练中的一组参数及其相关配置。
            - param_state: 参数状态，包含与参数组相关的统计信息，如步数和RMS等。

        返回:
            - 当前步骤的学习率。
        """
        # 判断是否使用相对步长
        if param_group['relative_step']:
            # 根据是否启用预热初始化，计算最小步长
            min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            # 计算当前步骤的学习率，确保不会超过1.0，并且随着步数增加而减小
            lr_t = min(min_step, 1.0 / math.sqrt(param_state['step']))
            # 初始化参数缩放因子
            param_scale = 1.0
            # 判断是否启用参数缩放
            if param_group['scale_parameter']:
                # 确保参数缩放因子不会小于设定的epsilon缩放因子，且不会超过当前参数的RMS值
                param_scale = max(param_group['eps_scale'], param_state['RMS'])
            # 根据计算结果更新参数组的学习率
            param_group['lr'] = lr_t * param_scale
        # 返回当前学习率
        return param_group['lr']

    @staticmethod
    def _get_options(param_group, param_shape):
        # 判断参数是否为矩阵形式，即维度数量是否大于等于2
        factored = len(param_shape) >= 2
        # 判断是否使用一阶矩，即beta1参数是否为None
        use_first_moment = param_group['beta1'] is not None
        # 返回参数是否为矩阵形式和是否使用一阶矩的判断结果
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        """
        计算给定张量的均方根（Root Mean Square）

        参数:
        tensor: 输入的张量，可以是一维、二维或多维

        返回:
        float: 输入张量的均方根值

        该方法的计算公式为：
        RMS = sqrt(sum(tensor^2) / element_count)

        其中：
        - sum(tensor^2) 表示张量中所有元素的平方和
        - element_count 表示张量中元素的数量

        这个方法主要用于神经网络中规范化操作，比如计算梯度的均方根，用于自适应学习率等。
        """
        return tensor.norm(2) / (tensor.numel() ** 0.5)


    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        """
        估算平方梯度的近似值。

        该方法通过计算行和列的指数移动平均平方值的比率，然后对这些比率进行逆平方根运算，
        来获得梯度的近似平方根。这种估算用于优化算法中，以帮助调整学习率。

        参数:
        exp_avg_sq_row (torch.Tensor): 沿着行维度的指数移动平均平方值。
        exp_avg_sq_col (torch.Tensor): 沿着列维度的指数移动平均平方值。

        返回:
        torch.Tensor: 估算的平方梯度的行和列的乘积。
        """
        # 计算行的平方根倒数因子
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        # 计算列的平方根倒数因子
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        # 返回行和列因子的乘积作为平方梯度的近似值
        return torch.mul(r_factor, c_factor)


    def step(self, closure=None):
        """
        执行单个优化步骤。

        参数:
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
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError('Adafactor不支持稀疏梯度。')

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # 梯度值的指数移动平均
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).to(grad)
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].to(grad)
                    if factored:
                        state['exp_avg_sq_row'] = state['exp_avg_sq_row'].to(grad)
                        state['exp_avg_sq_col'] = state['exp_avg_sq_col'].to(grad)
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                lr_t = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad ** 2 + group['eps']
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-1))
                    exp_avg_sq_col.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-2))
                    # 使用行和列的指数移动平均来近似梯度平方的指数移动平均
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(1.0 - beta2t, update)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group['clip_threshold']).clamp_(min=1.0))
                update.mul_(lr_t)

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group["beta1"]).add_(1 - group["beta1"], update)
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * lr_t, p_data_fp32)

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss
