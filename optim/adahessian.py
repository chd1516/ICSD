""" AdaHessian Optimizer

Lifted from https://github.com/davda54/ada-hessian/blob/master/ada_hessian.py
Originally licensed MIT, Copyright 2020, David Samuel
"""
import torch


class Adahessian(torch.optim.Optimizer):
    """
    Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 0.1)
        betas ((float, float), optional): coefficients used for computing running averages of gradient and the
            squared hessian trace (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional): exponent of the hessian trace (default: 1.0)
        update_each (int, optional): compute the hessian trace approximation only after *this* number of steps
            (to save time) (default: 1)
        n_samples (int, optional): how many times to sample `z` for the approximation of the hessian trace (default: 1)
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 hessian_power=1.0, update_each=1, n_samples=1, avg_conv_kernel=False):
        """
        初始化优化器Adahessian。

        参数:
        params: 可迭代的参数或参数组，将要优化的模型参数。
        lr (float): 学习率。默认值为0.1。
        betas (Tuple[float, float]): Adam风格优化器的系数。默认值为(0.9, 0.999)。
        eps (float): 添加到分母中的常数，用于数值稳定性。默认值为1e-8。
        weight_decay (float): 权重衰减（L2正则化强度）。默认值为0.0。
        hessian_power (float): 对角Hessian矩阵的幂次。默认值为1.0。
        update_each (int): 每次更新参数的频率。默认值为1。
        n_samples (int): 用于Hessian近似的样本数量。默认值为1。
        avg_conv_kernel (bool): 是否对卷积核进行平均。默认值为False。
        """
        # 验证学习率的有效性
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 验证epsilon的有效性
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 验证beta参数的有效性
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # 验证Hessian幂次的有效性
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.avg_conv_kernel = avg_conv_kernel

        # 使用一个独立的生成器，确保在分布式训练中所有GPU生成相同的`z`
        self.seed = 2147483647
        self.generator = torch.Generator().manual_seed(self.seed)

        # 将优化器的默认参数存储在字典中
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        # 调用父类构造方法，初始化优化器
        super(Adahessian, self).__init__(params, defaults)

        # 初始化每个参数的状态
        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0


    @property
    def is_second_order(self):
        """
        表示是否为二阶张量的属性。

        本属性用于指示当前张量是否为二阶张量。在各种张量操作和算法中，了解张量的阶数是非常重要的，
        因为不同的操作可能只适用于特定阶数的张量。通过将本属性设置为True，可以方便地告知使用方
        该张量的确切阶数，从而确保正确地执行后续操作。

        Returns:
            bool: 总是返回True，表示该张量是二阶的。
        """
        return True


    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        获取所有带梯度的param_groups中的所有参数

        Returns:
            Generator: All parameters requiring gradient calculation
            生成器:所有需要梯度计算的参数
        """

        # Use generator expression to iterate through all parameter groups and filter out parameters that require gradient
        # 使用生成器表达式来迭代所有参数组，并过滤掉需要梯度计算的参数
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """
        # 遍历所有参数，将每个参数的Hessian trace置零
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        计算哈希顿（Hutchinson）近似的哈essian迹，并为每个可训练参数累加。
        """
        # 初始化参数列表，用于存放需要计算哈essian的参数
        params = []
        # 过滤参数，只选择那些存在梯度的参数
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            # 仅在特定步骤计算迹，以减少计算成本
            if self.state[p]["hessian step"] % self.update_each == 0:
                params.append(p)
            # 更新参数的hessian step计数器
            self.state[p]["hessian step"] += 1

        # 如果没有需要处理的参数，则直接返回
        if len(params) == 0:
            return

        # 确保生成器与参数在相同的设备上
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)

        # 提取参数的梯度
        grads = [p.grad for p in params]

        # 对于每一个样本，计算哈essian-向量乘积
        for i in range(self.n_samples):
            # 生成参数的Rademacher分布向量
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            # 使用自动微分计算h_zs
            h_zs = torch.autograd.grad(
                grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            # 近似哈essian并累加到参数的hess属性上
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单个优化步骤。
        参数:
            closure (callable, 可选) -- 重新评估模型并返回损失的闭包函数 (默认: None)
        """

        # 初始化损失值为None，如果提供了closure函数，则通过调用closure来获取损失值
        loss = None
        if closure is not None:
            loss = closure()

        # 将Hessian矩阵置零
        self.zero_hessian()
        # 设置Hessian矩阵
        self.set_hessian()

        # 遍历每个参数组
        for group in self.param_groups:
            # 遍历组内的每个参数
            for p in group['params']:
                # 如果参数的梯度或Hessian值为None，则跳过该参数
                if p.grad is None or p.hess is None:
                    continue

                # 对于具有4维的参数（例如卷积核），对其Hessian矩阵的元素取绝对值并沿指定维度求平均后扩展为原尺寸
                if self.avg_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # 执行类似于AdamW的正确的步长衰减
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # 获取参数的状态
                state = self.state[p]

                # 状态初始化
                if len(state) == 1:
                    state['step'] = 0
                    # 梯度值的指数移动平均
                    state['exp_avg'] = torch.zeros_like(p)
                    # Hessian对角线平方值的指数移动平均
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p)

                # 从状态中获取指数移动平均值
                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                # 获取参数组的beta值
                beta1, beta2 = group['betas']
                # 更新步数
                state['step'] += 1

                # 衰减第一和第二矩的运行平均系数
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                # 计算偏差修正项
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 计算更新项的分母
                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # 执行更新
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        # 返回损失值
        return loss
