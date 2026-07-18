"""Standalone action-conditioned latent world model built on V-JEPA2.

The heavyweight Transformers/V-JEPA implementation is imported lazily.  This
keeps lightweight dataset indexing and metric tools usable from the LIBERO
environment, whose Transformers version intentionally predates
``AutoVideoProcessor``.
"""

from .evaluation import evaluate_latent_prediction

__all__ = ["LatentWorldModel", "LatentWorldModelConfig", "evaluate_latent_prediction"]


def __getattr__(name):
    if name in {"LatentWorldModel", "LatentWorldModelConfig"}:
        from .model import LatentWorldModel, LatentWorldModelConfig

        return {"LatentWorldModel": LatentWorldModel, "LatentWorldModelConfig": LatentWorldModelConfig}[name]
    raise AttributeError(name)
