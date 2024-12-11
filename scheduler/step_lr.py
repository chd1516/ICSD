""" Step Scheduler

Basic step LR schedule with warmup, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch

from .scheduler import Scheduler


class StepLRScheduler(Scheduler):
    """
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_t: float,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        """
        初始化学习率调度器。

        参数:
            optimizer (torch.optim.Optimizer): 被包装的优化器。
            decay_t (float): 学习率开始衰减的周期数或批次数。
            decay_rate (float): 每个周期的学习率衰减率，默认为1（不衰减）。
            warmup_t (int): 预热阶段的周期数或批次数，默认为0（无预热）。
            warmup_lr_init (float): 预热阶段初始学习率，默认为0。
            t_in_epochs (bool): 周期数还是批次数，默认为True（周期数）。
            noise_range_t (tuple): 随机噪声的范围，默认为None（无噪声）。
            noise_pct (float): 噪声的影响比例，默认为0.67。
            noise_std (float): 噪声的标准差，默认为1.0。
            noise_seed (int): 随机种子，默认为42。
            initialize (bool): 是否初始化学习率，默认为True。
        返回:
            None
        """
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        # 衰减周期数或批次数
        self.decay_t = decay_t
        # 学习率衰减率
        self.decay_rate = decay_rate
        # 预热阶段的周期数或批次数
        self.warmup_t = warmup_t
        # 预热阶段初始学习率
        self.warmup_lr_init = warmup_lr_init
        # 使用周期数还是批次数
        self.t_in_epochs = t_in_epochs
        # 如果有预热阶段
        if self.warmup_t:
            # 计算每个参数组的预热步骤
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            # 更新参数组的初始学习率
            super().update_groups(self.warmup_lr_init)
        else:
            # 如果没有预热阶段，预热步骤设为1
            self.warmup_steps = [1 for _ in self.base_values]


    def _get_lr(self, t):
        """
        根据当前训练周期t获取学习率。

        如果当前周期t小于预热期（warmup）周期数，则使用线性增长的方式计算学习率；
        否则，使用指数衰减的方式计算学习率。

        参数:
        t -- 当前的训练周期数

        返回:
        lrs -- 一个列表，包含每个学习率分量的值
        """
        # 在预热期（warmup）内，学习率线性增长
        if t < self.warmup_t:
            # 从初始学习率开始，根据每个分量的步长s线性增长
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        # 预热期结束后，学习率按指数衰减
        else:
            # 根据基础学习率值v和衰减率进行衰减，t // self.decay_t表示周期性的重置
            lrs = [v * (self.decay_rate ** (t // self.decay_t)) for v in self.base_values]
        return lrs


    def get_epoch_values(self, epoch: int):
        """
        根据epoch值获取学习率。

        此函数的目的是为了在训练的不同阶段动态调整学习率。它首先检查时间单位是否为epoch，
        如果是，就计算并返回对应epoch的学习率。否则，返回None。

        参数:
        epoch (int): 当前的epoch数。

        返回:
        如果时间单位是epoch，则返回对应epoch的学习率；否则返回None。
        """
        # 检查时间单位是否为epoch
        if self.t_in_epochs:
            # 如果是，计算并返回对应epoch的学习率
            return self._get_lr(epoch)
        else:
            # 如果不是，返回None
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

