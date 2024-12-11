import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


class AttrDict(dict):
    """
       AttrDict是一个继承自dict的类，它允许用户通过对象属性的方式访问字典的值。
       当通过键名访问字典时，如果键存在，则返回该键对应的值；如果键不存在，则返回一个AttrDict实例。
       这种特性使得AttrDict可以在动态添加属性时，不需要提前定义这些属性。
       """

    def __init__(self, *args, **kwargs):
        """
        初始化AttrDict实例。

        参数:
        *args: 位置参数，传递给父类dict的构造方法。
        **kwargs: 关键字参数，传递给父类dict的构造方法。

        在初始化时，它首先调用父类(dict)的初始化方法对自身进行初始化，
        然后将自身的字典属性__dict__指向自己，从而实现通过属性名访问字典值的功能。
        """
        super(AttrDict, self).__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.__dict__ = self  # 将实例的__dict__属性指向实例本身，实现通过点符号访问字典值


def setup_for_distributed(is_master):
    """
    根据节点是否为主节点来设置分布式训练中的打印行为。

    在分布式训练环境中，只有主节点负责打印日志信息。此函数旨在通过重写print函数来实现这一行为，
    避免所有节点都进行输出，从而导致输出信息过多且混乱。

    参数:
    is_master (bool): 当前节点是否为主节点的标志。如果为True，表示当前节点是主节点，可以进行打印；
                      否则，应该禁止打印。
    """
    import builtins as __builtin__  # 导入内置模块，以便能够重写其print函数
    builtin_print = __builtin__.print  # 保存原始的print函数引用

    def print(*args, **kwargs):
        """
        重写的print函数，增加强制打印选项。

        通过检查is_master标志和force参数来决定是否执行打印。即使在非主节点上，
        如果force参数为True，则仍然执行打印。
        """
        force = kwargs.pop('force', False)  # 提取force参数，默认为False force参数是用来强制打印，用于在非主节点上强制打印信息。
        if is_master or force:
            builtin_print(*args, **kwargs)  # 使用原始的print函数进行打印

    __builtin__.print = print  # 重写内置的print函数为上面定义的print



def is_dist_avail_and_initialized():
    """
    检查分布式系统是否可用且已初始化
    dist是torch.distributed模块，用于分布式训练。
    返回:
        bool: 如果分布式系统可用且已初始化，则返回True，否则返回False
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



def get_world_size():
    """
    获取当前分布式环境的世界大小。

    世界大小是指参与分布式训练的设备数量。如果当前环境不是分布式环境，则返回1。

    返回:
        int: 当前分布式环境的世界大小。
    """
    # 如果分布式环境不可用或未初始化，则返回1
    if not is_dist_avail_and_initialized():
        return 1
    # 返回分布式环境的世界大小
    return dist.get_world_size()



def get_rank():
    # 当分布式环境不可用或未初始化时，返回0
    if not is_dist_avail_and_initialized():
        return 0
    # 返回当前进程在分布式环境中的排名
    return dist.get_rank()



def is_main_process():
    """
    检查当前进程是否为主进程。
    主进程是分布式环境中的第一个进程，其排名为0。如果当前环境不是分布式环境，则返回True。
    返回:
        bool: 如果当前进程为主进程，则返回True，否则返回False。
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    如果当前进程为主进程，则保存文件。
    参数:
        *args: 位置参数，传递给torch.save的参数。
        **kwargs: 关键字参数，传递给torch.save的参数。
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    初始化分布式模式。

    该函数通过检查环境变量来确定是否启用分布式模式。它支持两种环境变量组合：
    1. 'RANK' 和 'WORLD_SIZE' 用于设置全局排名和世界大小。
    2. 'SLURM_PROCID' 用于在SLURM集群中设置进程ID。

    如果以上环境变量均不可用，则禁用分布式模式。

    参数:
        args: 命令行参数，包括rank、world_size和gpu等属性。

    返回:
        None
    """
    # 尝试从环境变量中获取分布式模式的配置信息
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # 如果在SLURM集群中运行，则使用SLURM的进程ID
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # 如果无法获取到分布式模式的配置信息，则不使用分布式模式
        print('Not using distributed mode')
        args.distributed = False

        return

    # 初始化分布式模式的相关参数
    args.distributed = True

    # 设置GPU设备
    torch.cuda.set_device(args.gpu)
    # 设置分布式后端为NCCL，NCCL是NVIDIA提供的分布式计算库，支持GPU加速
    args.dist_backend = 'nccl'
    # 打印分布式模式的初始化信息
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # 初始化进程组
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # 设置分布式模式下的屏障，确保所有进程同步
    torch.distributed.barrier()
    # 配置分布式模式下的打印行为，只允许rank为0的进程打印日志
    setup_for_distributed(args.rank == 0)



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.

        该类实现了平滑移动平均的功能。

        它能够跟踪一系列数值，并提供这些数值的当前平滑平均值。这对于过滤短期波动、
        识别趋势或从噪声中提取信号特别有用。构造函数允许定义窗口大小以及定制平均值的格式化方式。

        参数:
        - window_size: 窗口大小，决定了考虑平均的数值数量。默认值为20。
        - fmt: 格式字符串，用于显示平均值信息。默认值为"{median:.4f} ({global_avg:.4f})"，
               其中{median}表示当前窗口的中位数，{global_avg}表示自对象创建以来的所有数值的全局平均值。
        """
    def __init__(self, window_size=20, fmt=None):
        # 如果未提供格式字符串，则使用默认格式
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        # 初始化deque作为存储数值的容器，其长度由window_size决定
        self.deque = deque(maxlen=window_size)
        # 初始化总计数器，用于计算全局平均值
        self.total = 0.0
        # 初始化计数器，跟踪加入总计数的数值数量
        self.count = 0
        # 保存用户定义的格式字符串
        self.fmt = fmt

    def update(self, value, n=1):
        """
        更新统计信息，通过添加新值并更新计数和总和。

        此方法用于向deque中添加一个新值，并根据给定的数量n更新元素计数和总和。
        它在需要快速根据新数据调整统计信息时非常有用，特别是在固定窗口或滑动窗口统计中。

        参数:
        value: 单个数值，代表要添加到统计信息中的新值。
        n: 可选参数，代表新值value出现的次数，默认为1。如果n大于1，则value将被添加多次。

        返回值:
        无返回值，但方法会更新对象的内部状态。
        """
        # 向deque中添加新值，这是为了维护一个有序的值序列，尽管在这个方法中它没有被直接用于计算
        self.deque.append(value)
        # 更新元素计数，考虑到n可能大于1的情况，这允许我们有效地增加计数值
        self.count += n
        # 更新总和，通过将新值乘以n然后加到总和中，确保总和准确反映新值的多次添加
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在多个进程中进行同步操作。

        警告: 不会同步deque(一种先进先出的数据结构)。

        这个函数的主要目的是在分布式训练环境中，同步不同进程间的计数和总和。
        它首先检查分布式环境是否可用并已初始化，然后通过torch.tensor将计数和总和
        发送到所有进程，利用dist.barrier进行同步屏障操作，确保所有进程都完成前一阶段
        的操作后再继续。之后，使用dist.all_reduce进行全局归约操作，将所有进程的数据
        聚合在一起。最后，更新本地的计数和总和数据。
        """
        # 检查分布式环境是否可用并已初始化
        if not is_dist_avail_and_initialized():
            return

        # 将计数和总和数据包装成tensor，方便后续的同步操作
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')

        # 使用分布式屏障，确保所有进程都到达这一点后再继续
        dist.barrier()

        # 执行全局归约操作，将所有进程的t数据聚合
        dist.all_reduce(t)

        # 将聚合后的tensor数据转换回列表，以便更新本地数据
        t = t.tolist()

        # 更新本地的计数和总和数据
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        # 通过计算deque容器的中位数来获取当前数据的中间值
        # 将deque转换为torch张量，以便利用其内置的median方法
        d = torch.tensor(list(self.deque))
        # 返回中位数的值，.item()用于从单个元素的张量中提取数值
        return d.median().item()

    @property
    def avg(self):
        # 计算并返回队列中元素的平均值
        # 将队列中的元素转换为torch张量，以便进行数学计算
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        # 使用张量的mean方法计算平均值，并通过item方法将结果转换为Python数值类型
        return d.mean().item()

    @property
    def global_avg(self):
        """
        计算并返回全局平均值。

        通过将总和除以计数来计算平均值。使用@property装饰器以便可以通过
        对象.属性的形式访问，而不是对象.方法的形式。

        返回:
        全局平均值。

        注意:
        如果计数为0，将会导致除以0的错误。因此在实际使用中需要
        增加对计数的检查，以避免此类错误的发生。
        """
        return self.total / self.count

    @property
    def max(self):
        # 返回deque中的最大值
        return max(self.deque)

    @property
    def value(self):
        # 返回deque中的最新值

        return self.deque[-1]

    def __str__(self):
        """
        重写str函数，用于返回统计信息的格式化字符串。

        返回:
            str: 包含中位数、平均数、全局平均数和当前值的字符串表示。
        """
        # 使用格式字符串fmt来生成包含统计信息的字符串
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    # 初始化SummaryWriter类的实例
    def __init__(self, delimiter="\t"):
        # 初始化字典，用于存储平滑值对象
        self.meters = defaultdict(SmoothedValue)
        # 设置分隔符，默认为制表符
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        使用关键字参数更新指标。

        本函数遍历关键字参数kwargs中的每个项目，如果值是torch.Tensor类型，则将其转换为标量。
        之后，断言该值必须是浮点数或整数，并更新相应指标的计量器。

        参数:
        - **kwargs: 关键字参数，包含要更新的指标及其新值。

        返回: 无
        """
        for k, v in kwargs.items():
            # 如果值是torch.Tensor类型，将其转换为标量
            if isinstance(v, torch.Tensor):
                v = v.item()

            # 断言值必须是浮点数或整数
            assert isinstance(v, (float, int))

            # 更新相应指标的计量器
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        动态属性获取方法。

        当尝试访问类的某个属性时，如果该属性在类的字典表示中（`self.meters` 或 `self.__dict__`）存在，
        则返回其值。如果不存在，则引发 AttributeError 异常。

        参数:
        - attr (str): 要访问的属性名称。

        返回:
        - 任意类型: 如果属性存在，则返回属性对应的值。

        异常:
        - AttributeError: 如果属性不存在于类中。
        """
        # 首先检查 attr 是否在 meters 字典中，如果在则返回对应值
        if attr in self.meters:
            return self.meters[attr]
        # 如果不在 meters 中，检查 attr 是否在实例的 __dict__ 中，如果在则返回对应值
        if attr in self.__dict__:
            return self.__dict__[attr]
        # 如果 attr 既不在 meters 中，也不在 __dict__ 中，则抛出 AttributeError 异常
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """
        将所有测量仪表的名称和值转换成一个字符串格式。

        这个方法遍历一个包含各种测量仪表（meters）的字典，这些仪表每个都有一个特定的名称。
        它从每个仪表中提取名称和值，并将它们格式化成一个字符串形式。最后，使用类定义的分隔符
        将所有的格式化字符串连接起来。

        返回:
            一个字符串，包含了所有测量仪表的名称和它们对应的值，各项之间使用定义的分隔符分隔。
        """
        # 初始化一个空列表来存储每项损失的字符串表示
        loss_str = []
        # 遍历字典中的每一对名称和仪表
        for name, meter in self.meters.items():
            # 将名称和仪表的值格式化成字符串，并添加到列表中
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        # 使用类的分隔符将所有的损失字符串连接起来，并返回最终的字符串
        return self.delimiter.join(loss_str)

    def global_avg(self):
        """
        获取所有计量器中记录的全局平均损失。

        遍历所有的计量器（meters），计算并格式化每个计量器的全局平均值，
        最终通过特定的分隔符连接成一个字符串返回。

        Returns:
            str: 包含所有计量器的全局平均损失的字符串，每个损失之间由特定的分隔符分隔。
        """
        # 初始化一个空列表，用于存储格式化的损失字符串
        loss_str = []
        # 遍历所有的计量器项，项包括计量器的名字（name）和计量器本身（meter）
        for name, meter in self.meters.items():
            # 将计量器的全局平均值格式化为字符串，并添加到列表中
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        # 使用类定义的分隔符连接所有损失字符串，并返回结果
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        在所有进程中同步指标数据。

        这个方法遍历self.meters字典中的所有计量器，并调用它们的synchronize_between_processes方法。
        其目的是确保在多进程环境下，各个进程之间的指标数据能够保持一致。
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        # 向字典meters中添加一个新的计量器
        # 参数name代表计量器的名称，用于作为字典的键
        # 参数meter代表计量器对象，用于作为字典的值
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, Eiters=0):
        """
        在迭代过程中记录日志信息。

        参数:
        - iterable: 可迭代对象，如数据集或列表。
        - print_freq: 控制日志打印频率，每print_freq次迭代打印一次。
        - header: 日志头部信息，默认为空。
        - Eiters: 初始迭代次数，用于继续之前的迭代计数，默认为0。

        该函数通过yield逐个返回iterable中的元素，并在特定频率下打印迭代日志，
        包括迭代进度、ETA（预计到达时间）、时间消耗、数据加载时间等信息。如果在CUDA环境下运行，
        还会打印最大显存使用量。
        """

        # 初始化迭代计数器
        i = 0

        # 确保header有默认值
        if not header:
            header = ''

        # 记录开始时间
        start_time = time.time()
        # 用于计算时间差
        end = time.time()

        # 初始化平滑值计算对象，用于时间计算
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        # 格式化字符串，确保迭代次数对齐
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        # 构建日志消息的模板
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]

        # 如果有GPU支持，添加显存使用到日志
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')

        # 将日志模板连接成字符串
        log_msg = self.delimiter.join(log_msg)

        # 定义MB为内存单位
        MB = 1024.0 * 1024.0

        # 遍历可迭代对象
        for obj in iterable:
            # 更新数据加载时间
            data_time.update(time.time() - end)

            # 返回当前元素
            yield obj

            # 更新迭代时间
            iter_time.update(time.time() - end)

            # 根据配置打印日志
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算剩余时间
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                # 打印日志信息
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

            # 更新计数器和结束时间
            i += 1
            end = time.time()

        # 计算并打印总时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
