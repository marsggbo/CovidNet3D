import sys
sys.path.append('..')
from registry import Registry

EVALUATOR_REGISTRY = Registry("TRAINER")
EVALUATOR_REGISTRY.__doc__ = """
Registry for evaluator, i.e. the DefaultEvaluator.

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'build_evaluator',
    'EVALUATOR_REGISTRY',
]

def build_evaluator(cfg):
    """
    Built the trainer, defined by `cfg.trainer.name`.
    """
    evaluator = cfg.evaluator.name
    return EVALUATOR_REGISTRY.get(evaluator)(cfg)
    