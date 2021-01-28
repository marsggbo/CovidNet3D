# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.datasets import MNIST as _MNIST

from .build import DATASET_REGISTRY
from .transforms import build_transforms

__all__ = [
    'MNIST',
    'CIFAR10',
    'FakeData',
    '_FakeData',
]

@DATASET_REGISTRY.register()
def MNIST(cfg):
    cfg.defrost()
    root = cfg.dataset.datapath

    datasets = []
    for is_train in [True, False]:
        cfg.dataset.is_train = is_train
        transform = build_transforms(cfg)
        datasets.append(_MNIST(root=root, train=is_train,
                        transform=transform.transform, download=True))
    dataset_train, dataset_valid = datasets
    cfg.freeze()
    return dataset_train, dataset_valid

@DATASET_REGISTRY.register()
def CIFAR10(cfg):
    cfg.defrost()
    root = cfg.dataset.datapath

    datasets = []
    for is_train in [True, False]:
        cfg.dataset.is_train = is_train
        transform = build_transforms(cfg)
        datasets.append(_CIFAR10(root=root, train=is_train,
                        transform=transform.transform, download=True))
    dataset_train, dataset_valid = datasets
    cfg.freeze()
    return dataset_train, dataset_valid

class _FakeData(torch.utils.data.Dataset):
    def __init__(self, size=32, classes=10, is_3d=False, depth=16):
        self.size = size
        self.classes = classes
        self.is_3d = is_3d
        self.depth = depth

    def get_img_size(self):
        return self.size, self.size

    def get_scale_size(self):
        return self.size, self.size

    def __getitem__(self, index):
        if self.is_3d:
            self.data = torch.rand(1, self.depth, self.size, self.size)
        else:
            self.data = torch.rand(3, self.size, self.size)
        self.labels = torch.randint(0, self.classes, (1,)).item()
        return self.data, self.labels

    def __len__(self):
        return 16

@DATASET_REGISTRY.register()
def FakeData(cfg):
    classes = cfg.model.classes
    size = cfg.input.size
    slice_num = cfg.dataset.slice_num
    is_3d = cfg.dataset.is_3d
    if isinstance(size, list) or isinstance(size, tuple):
        size = size[0]
    assert isinstance(size, int), "the type of data size must be integer."
    dataset_train = _FakeData(size, classes, is_3d, slice_num)
    dataset_valid = _FakeData(size, classes, is_3d, slice_num)
    return dataset_train, dataset_valid


@DATASET_REGISTRY.register()
def fakedata(cfg):
    return FakeData(cfg)
