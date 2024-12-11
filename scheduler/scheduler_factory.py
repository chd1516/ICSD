""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler
from .plateau_lr import PlateauLRScheduler


def create_scheduler(args, optimizer):
    """
    根据参数创建一个学习率调度器。

    支持四种类型的学习率调度器：'cosine'、'tanh'、'step' 和 'plateau'。
    学习率噪声是可选的，如果提供，则根据噪声范围对学习率进行随机调整。

    参数:
        args: 命令行参数，包含配置信息如调度器类型、最小学习率、衰减率等。
        optimizer: 优化器实例，用于模型参数的优化。

    返回:
        lr_scheduler: 创建的学习率调度器实例。
        num_epochs: 总的训练周期数，考虑调度器的周期长度和冷却周期。
    """

    # 获取训练的总周期数
    num_epochs = args.epochs

    # 处理学习率噪声
    if getattr(args, 'lr_noise', None) is not None:
        # 如果lr_noise是列表或元组，则将其转换为与训练周期数对应的噪声范围
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            # 如果噪声范围只有一个值，直接使用该值
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            # 如果lr_noise不是列表或元组，直接乘以训练周期数
            noise_range = lr_noise * num_epochs
    else:
        # 如果没有提供lr_noise，噪声范围为None
        noise_range = None

    # 根据不同的调度器类型创建相应的学习率调度器
    if args.sched == 'cosine':
        # 创建余弦退火调度器
        '''
        optimizer：基础优化器，用于更新模型参数。
        t_initial：初始周期长度，默认为总的训练周期数 num_epochs。
        t_mul：周期长度的乘法因子，默认为 1。这意味着每个周期长度不变。
        lr_min：学习率的最小值，用于限制学习率的下限。
        decay_rate：衰减率，用于控制学习率的衰减速率。
        warmup_lr_init：预热初始学习率，用于预热阶段的起始学习率。
        warmup_t：预热周期长度，用于预热阶段的持续时间。
        cycle_limit：周期的最大限制，默认为 1，表示只有一个周期。
        t_in_epochs：周期长度是否以 epoch 为单位，默认为 True。
        noise_range_t：噪声范围，用于引入随机噪声。
        noise_pct：噪声百分比，默认为 0.67，用于控制噪声的比例。
        noise_std：噪声标准差，默认为 1，用于控制噪声的强度。
        noise_seed：噪声种子，默认为 42，用于保证噪声的一致性。
        '''
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=args.min_lr,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
        # 更新总的训练周期数，考虑周期长度和冷却周期
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'tanh':
        # 创建双曲正切退火调度器
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
        # 更新总的训练周期数，考虑周期长度和冷却周期
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'step':
        # 创建步进式衰减调度器
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
    elif args.sched == 'plateau':
        # 创建根据验证指标变化的调度器
        mode = 'min' if 'loss' in getattr(args, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_epochs,
            lr_min=args.min_lr,
            mode=mode,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cooldown_t=0,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )

    # 返回学习率调度器和总的训练周期数
    return lr_scheduler, num_epochs

