""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        """
        初始化Lookahead优化器包装器。

        参数:
        base_optimizer: 基础优化器实例，Lookahead将基于此优化器工作。
        alpha: 浮动数字，控制慢速更新的速率，应在0到1之间。
        k: 整数，表示Lookahead的步长，即每进行k次基础优化器更新后，进行一次慢速更新。

        异常:
        如果alpha或k的值不在合法范围内，将抛出ValueError异常。
        """
        # 验证alpha参数的合法性
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        # 验证k参数的合法性
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        # 定义Lookahead优化器的默认参数字典
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        # 更新基础优化器的defaults字典以包含Lookahead的默认参数
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # 手动将我们的默认参数添加到参数组中
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        """
        使用Lookahead策略更新慢速参数。

        Lookahead是一种优化算法，通过主优化器进行快速更新，并定期将参数移动到慢速参数，以提高训练稳定性和性能。
        本函数遍历给定参数组中的每个快速参数（fast_p），并根据'lookahead_alpha'值将其更新至慢速参数。

        参数:
        - group: 一个参数组，包含'params'(需要更新的快速参数)和'lookahead_alpha'(控制慢速参数向快速参数移动的速度)。
        """
        # 遍历参数组中的每个快速参数
        for fast_p in group["params"]:
            # 如果参数的梯度为None，则跳过该参数
            if fast_p.grad is None:
                continue
            # 获取对应参数的状态字典
            param_state = self.state[fast_p]
            # 如果状态字典中没有'slow_buffer'，则创建一个并初始化为与参数数据形状相同的张量
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            # 获取慢速参数的缓冲区
            slow = param_state['slow_buffer']
            # 根据Lookahead算法更新慢速参数：slow = slow + alpha * (fast - slow)
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            # 将快速参数的数据复制为慢速参数的新值
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        """
        同步lookahead操作。

        遍历参数组并更新慢速参数，以保持模型参数的同步。
        """
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        """
        执行一次优化步骤。

        此函数是优化器的核心，它基于当前模型参数和闭包（如果提供）来更新参数。
        它还会根据预先设定的规则，决定是否进行一次lookahead策略的更新。

        参数:
            closure (可选): 一个闭包函数，用于在每次优化步骤后重新计算损失。默认为None。

        返回:
            损失值，如果闭包被提供则返回，否则返回None。
        """
        # 断言以确保self.param_groups和基础优化器的param_groups是同一个对象。
        # 这是为了保证参数组的一致性和接下来操作的有效性。
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)

        # 调用基础优化器的step方法来执行一次标准的优化步骤。
        loss = self.base_optimizer.step(closure)

        # 遍历所有的参数组，为每个组更新lookahead_step计数器，并根据lookahead_k的值决定是否执行slow更新。
        for group in self.param_groups:
            # 每次调用step方法时，参数组的lookahead_step计数器增加1。
            group['lookahead_step'] += 1

            # 如果当前lookahead_step是lookahead_k的倍数，则执行一次update_slow方法调用，
            # 这是lookahead优化策略的一部分，用于周期性地更新"slow"参数。
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)

        # 返回损失值，如果提供了闭包函数的话。
        return loss


    def state_dict(self):
        """
        获取优化器的状态字典，包括快速状态和慢速状态

        此函数用于生成包含优化器状态的字典，其中包含快速状态和慢速状态，
        以及参数组的配置。快速状态包含基础优化器的状态，慢速状态则是
        优化器实例自身的状态。

        Returns:
            dict: 包含快速状态、慢速状态和参数组的字典
        """
        # 获取基础优化器的状态字典
        fast_state_dict = self.base_optimizer.state_dict()

        # 记录优化器实例自身的慢速状态
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }

        # 从基础优化器的状态字典中提取快速状态
        fast_state = fast_state_dict['state']

        # 从基础优化器的状态字典中提取参数组配置
        param_groups = fast_state_dict['param_groups']

        # 返回包含快速状态、慢速状态和参数组的字典
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        """
        从给定的状态字典中加载优化器的状态。

        此函数负责加载Lookahead优化器的快速状态和慢速状态。快速状态直接加载到基础优化器中，
        而慢速状态则通过super()加载。如果状态字典中不包含慢速状态，表明该状态字典可能来自
        没有应用Lookahead的优化器，此时会在加载后重新应用Lookahead的默认设置，以确保不丢失任何特定于Lookahead的配置。

        参数:
        state_dict (dict): 包含优化器状态的字典。
        """
        # 创建一个只包含快速状态和参数组的新字典，用于快速加载
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        # 将快速状态字典加载到基础优化器中
        self.base_optimizer.load_state_dict(fast_state_dict)

        # 初始化一个标志，用于指示是否需要为慢速状态重新应用默认设置
        slow_state_new = False
        # 如果状态字典中不包含慢速状态，则打印提示信息并初始化慢速状态
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True

        # 创建一个包含慢速状态和参数组的新字典，用于慢速加载
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # 尽管这里似乎是多余的，但可以节省代码
        }
        # 通过super()加载慢速状态字典
        super(Lookahead, self).load_state_dict(slow_state_dict)

        # 确保self.param_groups和基础优化器的param_groups引用相同的容器
        self.param_groups = self.base_optimizer.param_groups

        # 如果慢速状态是新初始化的，则重新应用默认设置，以补充可能缺失的Lookahead特定配置
        if slow_state_new:
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)
