"""Small, numerically stable latent prediction metrics."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_latent_prediction(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute aggregate metrics and in-batch future-state retrieval accuracy.

    This is kept independent of HDF5/indexing dependencies so the model can be
    imported in the VLA-JEPA environment, while dataset indexing remains
    available lazily through :mod:`latent_world_model.evaluation`.
    """
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target must have identical [B, tokens, dim] shapes")
    pred = predicted.float()
    true = target.float()
    pred_norm = F.normalize(pred, dim=-1)
    true_norm = F.normalize(true, dim=-1)
    token_cosine = (pred_norm * true_norm).sum(dim=-1)
    summary_pred = F.normalize(pred_norm.mean(dim=1), dim=-1)
    summary_target = F.normalize(true_norm.mean(dim=1), dim=-1)
    similarity = summary_pred @ summary_target.T
    labels = torch.arange(predicted.size(0), device=predicted.device)
    return {
        "l1": F.l1_loss(pred, true),
        "mse": F.mse_loss(pred, true),
        "mean_token_cosine": token_cosine.mean(),
        "retrieval_accuracy": (similarity.argmax(dim=1) == labels).float().mean(),
    }


@torch.no_grad()
def compute_prediction_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute absolute, directional, and persistence-relative metrics.

    All three tensors must have shape ``[B, patches, channels]``.  Accumulation
    callers should convert returned values to Python scalars immediately so a
    long evaluation never retains an autograd graph or GPU tensor list.
    """
    if predicted.shape != target.shape or predicted.shape != current.shape or predicted.ndim != 3:
        raise ValueError("predicted, target, current must have identical [B, patches, channels] shapes")
    pred = predicted.float()
    true = target.float()
    now = current.float()
    pred_delta = pred - now
    true_delta = true - now
    eps = torch.finfo(pred.dtype).eps
    pred_flat = pred.flatten(1)
    true_flat = true.flatten(1)
    now_flat = now.flatten(1)
    pred_delta_flat = pred_delta.flatten(1)
    true_delta_flat = true_delta.flatten(1)
    persistence_error = (now_flat - true_flat).pow(2).mean(dim=1)
    prediction_error = (pred_flat - true_flat).pow(2).mean(dim=1)
    target_variance = true_flat.var(dim=1, unbiased=False)
    pred_norm = F.normalize(pred, dim=-1)
    true_norm = F.normalize(true, dim=-1)
    pred_delta_norm = F.normalize(pred_delta, dim=-1)
    true_delta_norm = F.normalize(true_delta, dim=-1)
    return {
        "l1": (pred_flat - true_flat).abs().mean(dim=1),
        "mse": prediction_error,
        "rmse": prediction_error.sqrt(),
        "normalized_mse": prediction_error / target_variance.clamp_min(eps),
        "target_variance": target_variance,
        "prediction_variance": pred_flat.var(dim=1, unbiased=False),
        "persistence_mse": persistence_error,
        "persistence_ratio": prediction_error / persistence_error.clamp_min(eps),
        "mean_token_cosine": (pred_norm * true_norm).sum(dim=-1).mean(dim=1),
        "delta_cosine": (pred_delta_norm * true_delta_norm).sum(dim=-1).mean(dim=1),
        "delta_norm_ratio": pred_delta.flatten(1).norm(dim=1) / true_delta.flatten(1).norm(dim=1).clamp_min(eps),
        "prediction_variance_ratio": pred_flat.var(dim=1, unbiased=False) / true_flat.var(dim=1, unbiased=False).clamp_min(eps),
    }
