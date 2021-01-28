import sys
sys.path.append('..')
from registry import Registry

MUTATOR_REGISTRY = Registry("MUTATOR")
MUTATOR_REGISTRY.__doc__ = """
Registry for mutator.

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'MUTATOR_REGISTRY',
    'build_mutator',
]

def build_mutator(model, cfg):
    """
    Built the mutator.
    """
    name = cfg.mutator.name
    return MUTATOR_REGISTRY.get(name)(model, cfg)
