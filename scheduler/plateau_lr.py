""" Plateau Scheduler

Adapts PyTorch plateau scheduler and allows application of noise, warmup.

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

from .scheduler import Scheduler


class PlateauLRScheduler(Scheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self,
                 optimizer,  # 优化器对象
                 decay_rate=0.1,  # 学习率衰减比例
                 patience_t=10,  # 当指标停止提升时，等待的epoch数
                 verbose=True,  # 是否在调整学习率时打印信息
                 threshold=1e-4,  # 指标提升的最小变化量，以过滤出较小的波动
                 cooldown_t=0,  # 学习率调整后，不再调整的epoch数
                 warmup_t=0,  # 预热阶段的epoch数
                 warmup_lr_init=0,  # 预热阶段初始学习率
                 lr_min=0,  # 学习率的最小值
                 mode='max',  # 优化目标的模式，'max'或'min'
                 noise_range_t=None,  # 噪声范围，用于在预热阶段增加学习率的随机抖动
                 noise_type='normal',  # 抖动噪声的类型，'normal'或'uniform'
                 noise_pct=0.67,  # 噪声百分比，控制噪声的幅度
                 noise_std=1.0,  # 正态分布噪声的标准差
                 noise_seed=None,  # 噪声随机种子，用于结果复现
                 initialize=True,  # 是否初始化优化器的学习率
                 ):
        # 调用父类的构造方法，初始化优化器和学习率
        super().__init__(optimizer, 'lr', initialize=initialize)

        # 创建ReduceLROnPlateau学习率调度器，用于根据评估指标动态调整学习率
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=patience_t,
            factor=decay_rate,
            verbose=verbose,
            threshold=threshold,
            cooldown=cooldown_t,
            mode=mode,
            min_lr=lr_min
        )

        # 设置噪声相关参数，用于预热阶段学习率的随机抖动
        self.noise_range = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        # 如果未指定噪声随机种子，使用默认值42
        self.noise_seed = noise_seed if noise_seed is not None else 42

        # 设置预热阶段的参数
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        # 如果有预热阶段，计算每个优化器组的学习率增量，并初始化学习率为warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            # 如果没有预热阶段，直接设置学习率为初始值
            self.warmup_steps = [1 for _ in self.base_values]
        self.restore_lr = None  # 用于存储和恢复原始学习率的变量

    def state_dict(self):
        """
        获取状态字典。

        该方法用于返回一个包含学习率调度器的 'best' 和 'last_epoch' 的状态字典。
        'best' 表示调度器迄今为止记录的最佳指标，
        'last_epoch' 表示上一个训练周期的最后记录的epoch数。

        Returns:
            dict: 包含 'best' 和 'last_epoch' 的状态字典。
        """
        return {
            'best': self.lr_scheduler.best,
            'last_epoch': self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """
        加载状态字典，主要用于恢复学习率调度器的状态。

        :param state_dict: 包含模型状态和调度器状态的字典。
        """
        # 从状态字典中恢复最佳模型状态，用于学习率调度器。
        self.lr_scheduler.best = state_dict['best']

        # 如果状态字典中包含上一个epoch的信息，则恢复该信息。
        # 这对于继续训练过程非常重要，确保学习率调度器知道从哪个epoch开始。
        if 'last_epoch' in state_dict:
            self.lr_scheduler.last_epoch = state_dict['last_epoch']

    # override the base class step fn completely
    def step(self, epoch, metric=None):
        """
        根据当前epoch调整学习率。

        在warmup阶段内，线性增加学习率。超出warmup阶段后，根据metric和epoch
        步进基础学习率调度器，并根据设置的噪声范围可能地应用噪声。

        参数:
            epoch (int): 当前的训练轮数。
            metric (float, optional): 用于调度的度量标准，可能影响学习率的调整。
        """
        # 在预热阶段内，线性增加学习率
        if epoch <= self.warmup_t:
            lrs = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
            super().update_groups(lrs)
        else:
            # 预热阶段后，恢复实际学习率，然后步进基础调度器
            if self.restore_lr is not None:
                # 恢复上一次噪声扰动前的实际学习率
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.restore_lr[i]
                self.restore_lr = None

            self.lr_scheduler.step(metric, epoch)  # 步进基础调度器

            # 根据噪声范围，可能地应用噪声
            if self.noise_range is not None:
                # 确定是否需要应用噪声
                if isinstance(self.noise_range, (list, tuple)):
                    apply_noise = self.noise_range[0] <= epoch < self.noise_range[1]
                else:
                    apply_noise = epoch >= self.noise_range
                if apply_noise:
                    self._apply_noise(epoch)

    def _apply_noise(self, epoch):
        """
        为优化器的学习率应用噪声。

        此函数根据预定义的噪声类型和种子生成噪声，然后将该噪声应用于优化器每个参数组的学习率。
        它通过修改学习率来为模型引入随机性，这对于某些类型的训练可能有益，例如强化学习或某些
        类型的进化算法。

        参数:
        epoch (int): 当前的训练 epoch。用于噪声生成的种子中，以引入随时间变化的噪声。

        返回:
        无返回值。直接修改优化器的参数组学习率，并缓存原始学习率以便将来恢复。
        """
        # 初始化随机数生成器，以便生成可复现的噪声
        g = torch.Generator()
        # 手动设置随机数生成器的种子，确保每个 epoch 的噪声不同
        g.manual_seed(self.noise_seed + epoch)

        # 根据配置的噪声类型生成噪声
        if self.noise_type == 'normal':
            # 对于正态分布噪声，循环生成直到噪声在指定范围内
            while True:
                # 生成单个正态分布噪声值，并检查是否在允许的百分比范围内
                noise = torch.randn(1, generator=g).item()
                if abs(noise) < self.noise_pct:
                    break
        else:
            # 对于其他类型的噪声，如均匀分布，按指定范围生成噪声
            noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct

        # 应用噪声到学习率之前，缓存原始学习率以便后续恢复
        restore_lr = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            # 获取并缓存当前参数组的原始学习率
            old_lr = float(param_group['lr'])
            restore_lr.append(old_lr)
            # 计算应用噪声后的新学习率，并应用到参数组
            new_lr = old_lr + old_lr * noise
            param_group['lr'] = new_lr
        # 缓存所有参数组的原始学习率，以便将来恢复
        self.restore_lr = restore_lr
