"""Representation-space evaluation for future-latent predictions."""

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_latent_prediction(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Evaluate whether a prediction represents its matching future latent.

    Besides token-level L1/MSE and cosine similarity, ``retrieval_accuracy``
    asks whether each sample's predicted future is more similar to its own
    target than to other samples' target futures.  It detects a collapsed
    predictor that reports superficially low average error but loses sample-
    specific future information. Use batches with at least two examples for
    the retrieval metric.
    """
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target must have identical [B, tokens, dim] shapes")
    pred_norm = F.normalize(predicted.float(), dim=-1)
    target_norm = F.normalize(target.float(), dim=-1)
    token_cosine = (pred_norm * target_norm).sum(dim=-1)
    summary_pred = F.normalize(pred_norm.mean(dim=1), dim=-1)
    summary_target = F.normalize(target_norm.mean(dim=1), dim=-1)
    similarity = summary_pred @ summary_target.T
    labels = torch.arange(predicted.size(0), device=predicted.device)
    return {
        "l1": F.l1_loss(predicted.float(), target.float()),
        "mse": F.mse_loss(predicted.float(), target.float()),
        "mean_token_cosine": token_cosine.mean(),
        "retrieval_accuracy": (similarity.argmax(dim=1) == labels).float().mean(),
    }
