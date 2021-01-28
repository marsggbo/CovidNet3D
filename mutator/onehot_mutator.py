# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutables import InputChoice, LayerChoice
from nni.nas.pytorch.mutator import Mutator

from .build import MUTATOR_REGISTRY
from nas.utils.gumbel_softmax import *

__all__ = [
    'OnehotMutator',
]


@MUTATOR_REGISTRY.register()
class OnehotMutator(Mutator):
    def __init__(self, model, cfg):
        super().__init__(model)
        self.choices = nn.ParameterDict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                # self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length+1))
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length))
            elif isinstance(mutable, InputChoice):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.n_candidates))

    def device(self):
        for v in self.choices.values():
            return v.device

    def sample_search(self):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                # result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).bool()[:-1]
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).bool()
            elif isinstance(mutable, InputChoice):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).bool()
        return result

    def sample_final(self):
        return self.sample_search()