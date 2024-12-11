""" TanH Scheduler

TanH schedule with warmup, cycle/restarts, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import math
import numpy as np
import torch

from .scheduler import Scheduler


_logger = logging.getLogger(__name__)


class TanhLRScheduler(Scheduler):
    """
    Hyberbolic-Tangent decay with restarts.
    This is described in the paper https://arxiv.org/abs/1806.01593
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lb: float = -6.,
                 ub: float = 4.,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        """
        初始化优化器的学习率调度器。

        参数:
        optimizer (torch.optim.Optimizer): 被包装的优化器。
        t_initial (int): 初始周期长度。
        lb (float, 可选): 学习率的下界对数尺度。默认为 -6.0。
        ub (float, 可选): 学习率的上界对数尺度。默认为 4.0。
        t_mul (float, 可选): 每完成一个周期后，下一个周期长度是上一个周期长度的倍数。默认为 1.0。
        lr_min (float, 可选): 周期中学习率可以达到的最小值。默认为 0.0。
        decay_rate (float, 可选): 每个周期学习率的衰减率。默认为 1.0。
        warmup_t (int, 可选): 预热周期的数量。默认为 0。
        warmup_lr_init (float, 可选): 预热初始学习率。默认为 0。
        warmup_prefix (bool, 可选): 如果为 True，则在预热阶段使用线性增加的学习率，否则使用 cosin 学习率。默认为 False。
        cycle_limit (int, 可选): 完成的周期数上限。默认为 0，表示无限制。
        t_in_epochs (bool, 可选): 如果为 True，则 t 值以 epoch 为单位，否则以迭代次数为单位。默认为 True。
        noise_range_t (Tuple[int, int], 可选): 噪声的起始和结束时间。默认为 None。
        noise_pct (float, 可选): 在引入噪声的时间范围内，噪声的影响比例。默认为 0.67。
        noise_std (float, 可选): 噪声的标准差。默认为 1.0。
        noise_seed (int, 可选): 随机数种子，用于生成噪声。默认为 42。
        initialize (bool, 可选): 是否初始化学习率。默认为 True。
        """
        # 调用父类的构造方法进行初始化
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        # 参数校验
        assert t_initial > 0
        assert lr_min >= 0
        assert lb < ub
        assert cycle_limit >= 0
        assert warmup_t >= 0
        assert warmup_lr_init >= 0

        # 保存参数到实例变量
        self.lb = lb
        self.ub = ub
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs

        # 预热阶段的处理
        if self.warmup_t:
            # 计算预热阶段的学习率增量
            t_v = self.base_values if self.warmup_prefix else self._get_lr(self.warmup_t)
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in t_v]
            # 初始化预热阶段的学习率
            super().update_groups(self.warmup_lr_init)
        else:
            # 非预热阶段，初始化学习率为 base_values
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        """
        根据给定的时间步t计算学习率。

        该函数支持带有warmup阶段和循环学习率调度的学习率调整策略。首先判断时间步t是否处于warmup阶段，
        如果是，则线性插值从初始warmup学习率到目标学习率。如果超过了warmup阶段，将根据配置的调度策略计算学习率。
        调度策略可以是基于周期的，其中学习率在每个周期内变化，且变化范围随周期数增加而减小。

        参数:
            t: 当前的时间步数。

        返回:
            一个列表，包含每个优化器组的学习率。
        """

        # 判断当前时间步是否在warmup阶段内
        if t < self.warmup_t:
            # 如果在warmup阶段内，线性增加学习率从warmup初始学习率到目标学习率
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # 如果超过了warmup阶段，进行调度策略相关的计算
            if self.warmup_prefix:
                # 如果配置了warmup_prefix，从时间步中减去warmup的时间长度
                t = t - self.warmup_t

            # 计算当前周期内的实际时间步
            if self.t_mul != 1:
                # 计算当前周期数
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                # 计算当前周期的时间长度
                t_i = self.t_mul ** i * self.t_initial
                # 计算当前周期内已经过去的时间步
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                # 如果t_mul为1，表示每个周期的时间长度固定
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            # 判断当前周期数是否小于设置的周期限制
            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                # 根据周期数计算学习率衰减因子
                gamma = self.decay_rate ** i
                # 计算最小学习率和最大学习率
                lr_min = self.lr_min * gamma
                lr_max_values = [v * gamma for v in self.base_values]

                # 计算当前周期内的相对时间步
                tr = t_curr / t_i
                # 根据当前周期内的相对时间步计算学习率
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 - math.tanh(self.lb * (1. - tr) + self.ub * tr))
                    for lr_max in lr_max_values
                ]
            else:
                # 如果超过了周期限制，保持最小学习率不变
                lrs = [self.lr_min * (self.decay_rate ** self.cycle_limit) for _ in self.base_values]
        return lrs


    def get_epoch_values(self, epoch: int):
        """
        根据epoch值获取学习率。

        此函数用于根据当前epoch值获取相应学习率。如果对象的计时模式基于epochs，
        则计算并返回学习率；如果计时模式不基于epochs，则返回None。

        参数:
        epoch (int): 当前的epoch值。

        返回:
        如果对象的计时模式基于epochs，则返回对应的学习率值，否则返回None。
        """
        # 当对象的计时模式基于epochs时，计算并返回指定epoch的学习率。
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None


    def get_update_values(self, num_updates: int):
        """
        根据更新次数获取学习率更新值。

        此方法用于根据模型的更新次数来计算或检索学习率的更新值。它根据
        `t_in_epochs` 参数的布尔值来决定是否调用内部方法来获取学习率。

        参数:
            num_updates (int): 已完成的更新次数。

        返回:
            如果 `t_in_epochs` 为 False，则返回学习率更新值；
            否则返回 None。
        """
        # 当 t_in_epochs 为 False，表示不按 epoch 来更新学习率，直接返回计算的学习率更新值
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            # 当 t_in_epochs 为 True，表示按 epoch 来更新学习率，此时不支持获取单独的更新值，返回 None
            return None


    def get_cycle_length(self, cycles=0):
        """
        计算并返回当前任务的周期长度。

        如果指定了参数cycles，则计算该参数值下的周期长度；
        如果未指定cycles或其值为0，则使用任务的默认周期限制（self.cycle_limit）进行计算。
        确保计算的周期长度至少为1。

        当任务的乘数（t_mul）为1.0时，意味着周期长度将随着周期数线性增长，
        否则，周期长度的增长遵循一个指数衰减的规律。

        参数:
        - cycles: int, 指定计算周期长度时使用的周期数，默认为0，表示使用任务的默认周期限制。

        返回:
        - int: 计算得到的周期长度。
        """
        # 当传入的cycles参数为0时，使用对象的默认周期限制
        if not cycles:
            cycles = self.cycle_limit
        # 确保最小的周期数为1
        cycles = max(1, cycles)
        # 当乘数为1.0时，周期长度随周期数线性增长
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            # 当乘数不为1.0时，周期长度的增长遵循指数衰减规律，通过公式计算得到周期长度
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))
