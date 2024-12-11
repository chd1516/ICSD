""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim

from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    为模型参数添加权重衰减。权重衰减是正则化技术的一种，用于防止过拟合。

    参数:
    - model: PyTorch模型，用于遍历其参数并应用权重衰减。
    - weight_decay: 权重衰减系数，应用于具有2个以上维度的权重参数。
    - skip_list: 需要跳过的层列表，这些层不应用权重衰减。

    返回:
    - 列表，包含两个字典，分别用于模型参数的优化配置，一个用于无权重衰减的参数，一个用于有权重衰减的参数。
    """
    # 初始化两个空列表，分别用于存储不使用和使用权重衰减的参数
    decay = []
    no_decay = []

    # 遍历模型的所有参数，named_parameters()方法返回模型参数及其名称
    for name, param in model.named_parameters():
        # 如果参数不需要梯度更新（例如，被冻结的权重），则跳过该参数
        if not param.requires_grad:
            continue
        # 如果参数形状为1维或者参数名称以“.bias”结尾，或者参数名称在skip_list中，则不对其应用权重衰减
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            # 将不需要权重衰减的参数添加到no_decay列表中
            no_decay.append(param)
        else:
            # 将需要权重衰减的参数添加到decay列表中
            decay.append(param)

    # 返回参数配置列表，每个配置包含参数列表和对应的权重衰减值
    return [
        {'params': no_decay, 'weight_decay': 0.},  # 对no_decay列表中的参数不应用权重衰减
        {'params': decay, 'weight_decay': weight_decay}  # 对decay列表中的参数应用weight_decay系数作为权重衰减
    ]


# 定义一个函数，用于根据配置创建优化器
def create_optimizer(args, model, filter_bias_and_bn=True):
    """
    根据给定的参数和模型创建一个优化器。

    参数:
    - args: 包含配置项的对象，如优化器类型、学习率、权重衰减等。
    - model: 需要优化的模型。
    - filter_bias_and_bn: 是否从权重衰减中排除偏置和批patch归一化参数，默认为True。

    返回:
    - optimizer: 根据配置创建的优化器实例。
    """

    # 将优化器类型的字符串表示转换为小写
    opt_lower = args.opt.lower()
    # 获取权重衰减参数 这是用来计算梯度的权重衰减，即对权重参数的衰减程度
    weight_decay = args.weight_decay

    # 如果启用了权重衰减并且需要过滤偏置和批归一化参数

    if weight_decay and filter_bias_and_bn:
        # 定义要跳过的参数集合
        skip = {}
        # 如果模型定义了no_weight_decay方法，用于指定哪些参数不应受到权重衰减影响
        if hasattr(model, 'no_weight_decay'): # 该项目没有启用该功能 model = icsd(args, config) unire里面没有这个方法
            skip = model.no_weight_decay()
        # 使用自定义的add_weight_decay函数添加权重衰减，对特定参数进行跳过
        parameters = add_weight_decay(model, weight_decay, skip)
        # 将权重衰减设为0，因为在参数处理时已经考虑了权重衰减
        weight_decay = 0.
    else:
        # 如果不进行权重衰减或不过滤偏置和批归一化参数，直接使用模型的所有参数
        parameters = model.parameters()

    # 对于融合优化器，需要确认安装了APEX且CUDA可用
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX和CUDA是融合优化器的必要条件'

    # 初始化优化器参数字典
    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    # 根据配置更新优化器参数，如eps、betas等
    #eps 参数（epsilon 参数）在很多优化器中都有使用，尤其是在基于梯度下降的方法中。它的主要作用是防止数值不稳定，特别是在除法操作中避免除以零的情况。
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    # 处理优化器类型字符串，分离出主优化器类型
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    # 根据优化器类型创建相应的优化器实例
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None) #移除 eps 参数是因为 SGD 优化器本身不需要 eps 参数。
        # 创建 SGD 优化器实例。
        # parameters 是需要优化的模型参数。
        # momentum 是动量参数，从 args 中获取。
        # nesterov=True 表示使用 Nesterov 加速梯度。
        # **opt_args 将剩余的优化器参数传递给 SGD 优化器。
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    # 以下为根据不同优化器类型创建优化器的代码，省略了每种优化器的注释，因为它们与上面的SGD部分类似
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        # 如果优化器类型无效，抛出异常
        assert False and "Invalid optimizer"
        raise ValueError

    # 如果优化器类型包含lookahead，则包装优化器为Lookahead
    #Lookahead 是一种优化器包装器，可以在基础优化器之上添加额外的优化策略，提高模型的训练效果。
    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            # 使用 Lookahead 包装器来增强基础优化器。这里的 optimizer 是之前创建的基础优化器实例
            optimizer = Lookahead(optimizer)

    # 返回创建的优化器
    return optimizer
