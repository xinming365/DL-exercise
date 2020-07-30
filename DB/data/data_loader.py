import math
import bisect

import imgaug
import numpy as np

import torch
# 分布式pytorch。pytorch分布式目前只支持Linux。
import torch.distributed as dist
from torch.utils.data import Sampler, ConcatDataset, BatchSampler

from concern.config import Configurable, State


def default_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    imgaug.seed(worker_id)


class DataLoader(Configurable, torch.utils.data.DataLoader):
    dataset = State()
    batch_size = State(default=256)
    num_workers = State(default=10)
    is_train = State(default=True)
    collect_fn = State(default=None)
    drop_last = State(default=True)
    shuffle = State()

    # cmd表示什么？？cmd为字典
    # cmd的keys有'is_train', 'batch_size', 'num_workers', 'distributed', 'num_gups'.
    def __init__(self, **kwargs):
        self.load_all(**kwargs)
        if self.collect_fn is None:
            self.collect_fn = torch.utils.data.dataloader.default_collate
        cmd = kwargs.get('cmd', {})
        self.is_train = cmd['is_train']
        if 'batch_size' in cmd:
            self.batch_size = cmd['batch_size']
        if self.shuffle is None:
            self.shuffle = self.is_train
        self.num_workers = cmd.get('num_workers', self.num_workers)

        if cmd.get('distributed'):
            sampler = DistributedSampler(
                self.dataset, shuffle=self.shuffle,
                num_replicas=cmd['num_gpus'])
            batch_sampler = BatchSampler(
                sampler, self.batch_size // cmd['num_gpus'], False)
            torch.utils.data.DataLoader.__init__(
                self, self.dataset, batch_sampler=batch_sampler,
                num_workers=self.num_workers, pin_memory=False,
                drop_last=self.drop_last, collate_fn=self.collect_fn,
                worker_init_fn=default_worker_init_fn)
        else:
            torch.utils.data.DataLoader.__init__(
                self, self.dataset,
                batch_size=self.batch_size, num_workers=self.num_workers,
                drop_last=self.drop_last, shuffle=self.shuffle,
                pin_memory=True, collate_fn=self.collect_fn,
                worker_init_fn=default_worker_init_fn)
        self.collect_fn = str(self.collect_fn)


class SuccessiveRandomSampler(Sampler):
    '''Random Sampler that yields sorted data in successive ranges.
    Args:
        dataset: Dataset used for sampling.
    '''

    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    # randperm : Returns a random permutation of integers from 0 to n - 1.
    # manual_seed() ：Sets the seed for generating random numbers. 推荐设置大的种子数。神经网络中
    # 参数默认是随机初始化的。
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # assert: （断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        #
        # 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况，例如我们的代码只能在 Linux 系统下运行，可以先判断当前系统是否符合条件。

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
        # Currently, torch.distributed is available on Linux and MacOS.
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
        # get_world_size() Returns the number of processes in the current process group
        # 全局进程个数
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
        # Returns the rank of current process group
        # 表示进程序号，用于进程间通讯，表征进程优先级。rank=0的主机为master节点。一般而言，rank 均为从 0 到 world_size 的整数。
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # total_size 比dataset的数目大
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # 由于total_size 比dataset的数目大， 所以indices要加上多出来的索引。
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InfiniteOrderedSampler(Sampler):
    def __init__(self, data_source, limit_size):
        self.data_source = data_source
        self.limit_size = limit_size

    def __iter__(self):
        n = len(self.data_source)
    # 生成器函数，每次只产生一个元素，能够节省很多开销。
        def wrapper():
            cnt = 0
            while cnt < self.limit_size:
                if cnt % n == 0:
                    idx = torch.randperm(n).tolist()
                yield idx[cnt % n]
                cnt += 1

        return wrapper()

    def __len__(self):
        return self.limit_size

# python的多继承，需要注意圆括号中父类的顺序，若是父类中有相同的方法名，而在子类使用时未指定，python从左至右搜索
# 即方法在子类中未找到时，从左到右查找父类中是否包含方法。
class InfiniteDataLoader(Configurable, torch.utils.data.DataLoader):
    dataset = State()
    batch_size = State(default=256)
    num_workers = State(default=10)
    limit_size = State(default=2 ** 31)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'batch_size' in cmd:
            self.batch_size = cmd['batch_size']

        sampler = InfiniteOrderedSampler(self.dataset, self.limit_size)

        torch.utils.data.DataLoader.__init__(
            self, self.dataset,
            batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=sampler, worker_init_fn=default_worker_init_fn,
        )


class RandomSampleSampler(Sampler):
    def __init__(self, data_source, weights=None, size=2 ** 31):
        self.data_source = data_source
        if weights is None:
            self.probabilities = np.full(len(data_source), 1 / len(data_source))
        else:
            self.probabilities = np.array(weights) / np.sum(weights)
        self.cum_prob = np.cumsum(self.probabilities)
        self.size = size

    def __iter__(self):
        def wrapper():
            for i in range(self.size):
    # bisect模块，使用了基本的二分算法。bisect.bisect(,)查找该数值将会插入的位置并返回，而不会插入。
                yield bisect.bisect(self.cum_prob, torch.rand(1)[0], hi=len(self.data_source) - 1)

        return wrapper()

    def __len__(self):
        return self.size


class RandomSampleDataLoader(Configurable, torch.utils.data.DataLoader):
    datasets = State()
    weights = State()
    batch_size = State(default=256)
    num_workers = State(default=10)
    size = State(default=2 ** 31)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'batch_size' in cmd:
            self.batch_size = cmd['batch_size']

        probs = []
        for dataset, weight in zip(self.datasets, self.weights):
            probs.append(np.full(len(dataset), weight / len(dataset)))
    # pytorch自带的一种合并子数据集的方式，ConcatDataset类。
        dataset = ConcatDataset(self.datasets)
        probs = np.concatenate(probs)
        assert (len(dataset) == len(probs))

        sampler = RandomSampleSampler(dataset, probs, self.size)

        torch.utils.data.DataLoader.__init__(
            self, dataset,
            batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=sampler, worker_init_fn=default_worker_init_fn,
        )
