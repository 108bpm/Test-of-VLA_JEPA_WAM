"""Representation-space metrics and lazy dataset-index helpers.

The LIBERO indexer depends on ``h5py``, which is intentionally not required in
the VLA-JEPA model environment.  Import it only when an index is requested.
"""

from .metrics import compute_prediction_metrics, evaluate_latent_prediction


def build_rollout_index(*args, **kwargs):
    from .index import build_rollout_index as _build_rollout_index

    return _build_rollout_index(*args, **kwargs)


__all__ = ["build_rollout_index", "compute_prediction_metrics", "evaluate_latent_prediction"]
