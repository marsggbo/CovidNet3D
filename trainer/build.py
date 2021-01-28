import sys
sys.path.append('..')
from registry import Registry

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.__doc__ = """
Registry for trainer, i.e. the OnehotTrainer.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

__all__ = [
    'build_trainer',
    'TRAINER_REGISTRY',
]

def build_trainer(cfg):
    """
    Built the trainer, defined by `cfg.trainer.name`.
    """
    trainer = cfg.trainer.name
    return TRAINER_REGISTRY.get(trainer)(cfg)
    