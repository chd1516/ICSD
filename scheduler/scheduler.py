from typing import Dict, Any

import torch


class Scheduler:
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.

    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,  # 优化器实例，必须是torch.optim下的优化器或其子类
                 param_group_field: str,  # 用于识别参数组中哪个字段是我们需要操作的对象
                 noise_range_t=None,  # 噪声的范围，可选参数
                 noise_type='normal',  # 噪声的类型，可以是'normal'或'uniform'
                 noise_pct=0.67,  # 应用噪声的参数组字段比例
                 noise_std=1.0,  # 噪声的标准差，仅当noise_type为'normal'时有效
                 noise_seed=None,  # 随机种子，用于生成噪声
                 initialize: bool = True) -> None:  # 标志位，确定是否在初始化时设置初始值
        self.optimizer = optimizer  # 保存优化器实例
        self.param_group_field = param_group_field  # 保存需要操作的参数组字段名
        self._initial_param_group_field = f"initial_{param_group_field}"  # 定义一个字段名，用于保存初始的参数组字段值
        # 如果需要初始化，那么检查每个参数组是否包含param_group_field，并保存其初始值
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        # 如果不初始化，那么检查每个参数组是否包含_initial_param_group_field
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        # 从参数组中提取初始值
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # 是否对所有实例都有必要设置一个度量标准？
        self.noise_range_t = noise_range_t  # 保存噪声范围
        self.noise_pct = noise_pct  # 保存噪声应用比例
        self.noise_type = noise_type  # 保存噪声类型
        self.noise_std = noise_std  # 保存噪声标准差
        self.noise_seed = noise_seed if noise_seed is not None else 42  # 设置随机种子，默认为42
        self.update_groups(self.base_values)  # 调用方法更新参数组


    def state_dict(self) -> Dict[str, Any]:
        """
        获取对象的状态字典表示。

        此方法用于返回对象的属性及其值的字典表示，
        但不包括(optimizer)属性，通常用于保存或复制对象的状态，
        排除优化器相关属性可以避免不必要的序列化错误或大的内存占用。

        返回:
            包含对象属性及其对应值的字典，不包含(optimizer)属性。
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载状态字典到当前对象。

        该方法的作用是将给定的状态字典（state_dict）合并到对象的现有状态中，
        实现对象状态的加载和恢复。这对于在保存和加载模型、配置等方面非常有用。

        参数:
        state_dict: Dict[str, Any] - 一个字典，包含了对象的状态信息。键通常为字符串类型，
        而值可以是任意类型。

        返回值:
        None - 方法没有返回值，它直接修改对象的状态。

        注意:
        此方法假定state_dict中的键对应于对象的可设置属性，且不进行类型检查或错误处理。
        如果state_dict中包含无效键或值，将引发异常。
        """
        self.__dict__.update(state_dict)  # 将状态字典更新到对象的__dict__中，从而覆盖或添加属性


    def get_epoch_values(self, epoch: int):
        """
        根据指定的训练轮次（epoch）获取相关值。

        通常在模型训练过程中，此方法用于检索特定轮次的训练状态，例如损失、准确率等。
        该方法可能返回None，表示在当前上下文中没有该轮次的数据或者数据不可用。

        参数:
        epoch (int): 训练轮次的编号，从1开始。

        返回:
        该方法可能返回None，具体返回值取决于具体的实现和上下文。
        """
        # 返回特定epoch的数据，这里返回None可能是因为数据未找到或不适用
        return None


    def get_update_values(self, num_updates: int):
        """
        获取更新的值

        此函数的目的是为未来可能实现的更新机制预留空间。当前设计为不返回任何值。

        参数:
        num_updates (int): 已进行的更新次数，预留参数，目前未使用。

        返回:
        None: 当前不返回任何值，预留用于未来实现。
        """
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        """
        根据当前epoch调整学习率相关参数。

        本方法首先接收epoch信息和一个可选的metric指标，然后从数据库中获取当前epoch对应的参数值。
        获取到的参数值会添加上一些噪声，以适应一些算法对参数随机性的需求，并最终更新参数组。

        :param epoch: int, 当前的训练轮数。
        :param metric: float, 可选参数，用于指定某个评价指标。
        :return: None
        """
        # 存储metric值，可能用于后续操作
        self.metric = metric
        # 获取当前epoch对应的参数值
        values = self.get_epoch_values(epoch)
        # 如果获取到的参数值不为空
        if values is not None:
            # 为参数值添加噪声，以增加随机性
            values = self._add_noise(values, epoch)
            # 更新参数组，应用新的参数值
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        """
        分步骤更新粒子群优化器的状态。

        参数:
        - num_updates: int, 指定更新的步骤数。
        - metric: float, 可选参数，表示性能指标。

        该方法首先根据更新步骤数获取更新值，如果获取到的更新值不为空，
        则会向这些值添加噪声（以增加解的多样性），最后更新粒子群的状态。
        """
        # 更新性能指标
        self.metric = metric

        # 获取更新步骤的值
        values = self.get_update_values(num_updates)

        # 检查获取的更新值是否有效
        if values is not None:
            # 向更新值添加噪声，并更新粒子群的状态
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        """
        更新参数组中的参数。

        该方法用于将给定的值赋值给优化器参数组中的每个参数组。如果传入的值不是列表或元组，
        则会将其复制以匹配参数组的数量。这样可以确保每个参数组都被赋予一个值。

        参数:
        values (list, tuple, any): 要更新到参数组中的值。可以是列表、元组，或其他任意类型。

        返回:
        None
        """
        # 检查values是否为列表或元组，如果不是则将其复制以匹配参数组的数量
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)

        # 遍历优化器的参数组和values，将每个参数组的[self.param_group_field]字段更新为对应的值
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        """
        根据当前训练周期t向学习率lrs中添加噪声。

        当噪声范围noise_range_t适用于当前周期t时，根据噪声类型noise_type生成相应的噪声，并将其添加到学习率中。
        支持的噪声类型有高斯噪声（'normal'）和均匀噪声。对于高斯噪声，确保生成的噪声在指定的百分比范围内。

        参数:
            lrs (list): 学习率列表。
            t (int): 当前训练周期。

        返回:
            list: 添加了噪声的学习率列表。
        """
        # 判断是否根据训练周期t应用噪声
        if self.noise_range_t is not None:
            # 判断noise_range_t类型并确定是否应用噪声
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            # 若决定应用噪声，则根据噪声类型生成噪声并添加到学习率中
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                # 根据噪声类型生成噪声
                if self.noise_type == 'normal':
                    # 仅当生成的噪声在指定百分比范围内时，结束循环
                    while True:
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    # 对于非'normal'的噪声类型，假设为均匀分布噪声
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                # 将噪声添加到学习率中
                lrs = [v + v * noise for v in lrs]
        return lrs

