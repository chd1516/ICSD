""" Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import math
import numpy as np
import torch

from .scheduler import Scheduler

from pdb import set_trace as breakpoint

_logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=True,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        """
        初始化CosineAnnealingLR调度器。

        该调度器通过在多个周期内调整学习率，以模拟余弦退火过程，旨在提高模型性能。

        参数:
        optimizer: torch.optim.Optimizer - 被包装的优化器。
        t_initial: int - 初始周期长度，必须大于0。
        t_mul: float - 每个周期之后，下一个周期长度为前一个周期长度乘以t_mul，默認为1。
        lr_min: float - 周期内的最小学习率，默認为0。
        decay_rate: float - 学习率的衰减率，每个周期结束时应用于当前学习率，默認为1（无衰减）。
        warmup_t: int - 预热阶段的周期长度，默認为0（无预热）。
        warmup_lr_init: float - 预热阶段初始学习率，默認为0。
        warmup_prefix: bool - 如果为True，预热阶段将在周期的开头进行，默認为True。
        cycle_limit: int - 完成固定数量的周期后，调度器将停止更新学习率，默認为0（无限制）。
        t_in_epochs: bool - 如果为True，周期长度将按epoch数计算；否则按迭代次数计算，默認为True。
        noise_range_t: Tuple[int, int] - 学习率噪声的范围，以周期内的步数为单位，默認为None（无噪声）。
        noise_pct: float - 噪声振幅占学习率范围的比例，默認为0.67。
        noise_std: float - 噪声的标准差，用于控制噪声的强度，默認为1.0。
        noise_seed: int - 随机数种子，用于生成学习率噪声，默認为42。
        initialize: bool - 如果为True，初始化时会设置学习率，默認为True。
        """
        # 调用父类（Scheduler）的初始化方法，传递相关参数。
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        # 确保t_initial大于0，lr_min非负。
        assert t_initial > 0
        assert lr_min >= 0

        # 如果t_initial、t_mul和decay_rate都为1，那么学习率将不会改变。
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")

        # 保存用户指定的参数。
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs

        # 如果有预热阶段，计算预热阶段每步的学习率增量，并初始化预热学习率。
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            # 如果没有预热阶段，初始化warmup_steps为全1列表。
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        """
        根据给定的时间步t计算学习率。

        该函数支持带有warmup阶段和余弦退火策略的学习率调度。如果指定了warmup阶段，则在warmup阶段内，
        学习率从warmup_lr_init线性增加到目标的学习率。超过warmup阶段后，学习率根据余弦退火策略和可能的decay率进行调整，
        直到达到最小学习率lr_min。cycle_limit参数用于限制调整周期的数量。

        参数:
            t: 当前的时间步。

        返回:
            lrs: 一个包含各个优化器组的学习率列表。
        """
        # 在warmup阶段内，线性增加学习率
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # 如果设置了warmup_prefix，从时间步t中减去warmup阶段的时间
            if self.warmup_prefix:
                t = t - self.warmup_t

            # 计算当前周期的时间长度和当前时间步在当前周期中的位置
            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            # 根据周期数计算学习率的decay因子
            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            # 根据cosine annealing策略计算学习率
            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                # 达到cycle_limit后，保持最小学习率
                lrs = [self.lr_min for _ in self.base_values]

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
        # 当 t_in_epochs 为 False，表示不按 epoch 来更新学习率时，计算并返回学习率更新值。
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
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
