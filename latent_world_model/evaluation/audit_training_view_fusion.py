"""Reproduce the two-view batching logic used by VLA-JEPA training.

The source framework flattens a ``[B,V,...]`` tensor in sample-major order and
then applies ``torch.chunk(..., chunks=V, dim=0)`` as though the flattened data
were view-major.  For ``B>1`` this combines encoder features from different
samples/views.  This audit evaluates the released predictor under both that
legacy mapping and the intended per-sample mapping, without changing weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .. import LatentWorldModel
from .audit_model_protocol import _balanced_rows
from .runner import _joint_clip


def _encode_separate_views(
    model: LatentWorldModel,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    clip_batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for offset in range(0, len(rows), clip_batch_size):
        chunk = rows[offset : offset + clip_batch_size]
        clips = []
        for row in chunk:
            with h5py.File(row["record_path"], "r") as handle:
                clips.append(_joint_clip(handle, int(row["query_frame"])))
        videos = torch.stack(clips, dim=0).to(device)
        pixels = model.preprocess_video(videos)
        with torch.inference_mode():
            features = model.encoder.get_vision_features(
                pixel_values_videos=pixels.flatten(0, 1).to(model.encoder.device)
            )
        # Flattening [chunk,V] above is sample-major.  Preserve that ordering
        # across chunks so the global source-code mapping can be reproduced.
        outputs.append(features.detach().cpu().to(torch.float16))
        del videos, pixels, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def _predict_batched(
    model: LatentWorldModel,
    latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    latent_steps = model.latent_steps
    patch_count = latents.shape[1] // latent_steps
    context = latents[:, : patch_count * (latent_steps - 1)]
    outputs = []
    for offset in range(0, latents.shape[0], batch_size):
        with torch.inference_mode():
            predicted = model.predict_from_latents(
                context[offset : offset + batch_size].to(device),
                actions[offset : offset + batch_size].to(device),
            )
        outputs.append(predicted.detach().cpu())
    return torch.cat(outputs, dim=0)


def _per_sample_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    pred = predicted.float()
    true = target.float()
    reduce_dims = (1, 2)
    mse = (pred - true).pow(2).mean(dim=reduce_dims).numpy()
    l1 = (pred - true).abs().mean(dim=reduce_dims).numpy()
    cosine = F.cosine_similarity(pred, true, dim=-1).mean(dim=1).numpy()
    return {
        "mse_mean": float(mse.mean()),
        "mse_std": float(mse.std()),
        "l1_mean": float(l1.mean()),
        "l1_std": float(l1.std()),
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std()),
    }


def _mapping(batch_size: int, views: int) -> list[dict[str, Any]]:
    flattened = [(sample, view) for sample in range(batch_size) for view in range(views)]
    chunk_length = len(flattened) // views
    return [
        {
            "output_row": row,
            "action_sample": row,
            "concatenated_features": [list(flattened[row + view * chunk_length]) for view in range(views)],
        }
        for row in range(batch_size)
    ]


def run_audit(
    *,
    index_path: Path,
    encoder_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    windows: int,
    clip_batch_size: int,
    predictor_batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    if windows < 2 or windows % 2:
        raise ValueError("windows must be an even integer >= 2")
    device = torch.device(device_name)
    rows = _balanced_rows(index_path, windows)
    if len(rows) != windows:
        raise ValueError(f"requested {windows} rows but selected {len(rows)}")
    actions = []
    for row in rows:
        with h5py.File(row["record_path"], "r") as handle:
            actions.append(np.asarray(handle["latent_action_tokens"][int(row["query_row"])], dtype=np.float32))
    action_tensor = torch.from_numpy(np.stack(actions, axis=0))

    model = LatentWorldModel(encoder_path)
    checkpoint_load = model.load_predictor_checkpoint(checkpoint_path, strict=True)
    model.to(device).eval()
    raw = _encode_separate_views(
        model,
        rows,
        device=device,
        clip_batch_size=clip_batch_size,
    )
    batch_size = len(rows)
    views = model.config.num_views
    sequence_tokens, hidden = raw.shape[1:]
    correct = raw.view(batch_size, views, sequence_tokens, hidden).permute(0, 2, 1, 3).flatten(2)
    legacy = torch.cat(torch.chunk(raw, chunks=views, dim=0), dim=2)
    if correct.shape != legacy.shape:
        raise RuntimeError(f"fusion shape mismatch: correct={correct.shape}, legacy={legacy.shape}")

    patch_count = correct.shape[1] // model.latent_steps
    correct_target = correct[:, patch_count:]
    legacy_target = legacy[:, patch_count:]
    correct_prediction = _predict_batched(
        model, correct, action_tensor, device=device, batch_size=predictor_batch_size
    )
    legacy_prediction = _predict_batched(
        model, legacy, action_tensor, device=device, batch_size=predictor_batch_size
    )
    shuffled_actions = action_tensor.roll(shifts=-1, dims=0)
    legacy_shuffled_prediction = _predict_batched(
        model, legacy, shuffled_actions, device=device, batch_size=predictor_batch_size
    )

    mapping = _mapping(batch_size, views)
    result = {
        "index": str(index_path.resolve()),
        "encoder": str(encoder_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_load": checkpoint_load,
        "batch_size_reproduced": batch_size,
        "views": views,
        "source_flatten_order": "sample-major: b0v0,b0v1,b1v0,b1v1,...",
        "intended_fusion": _per_sample_metrics(correct_prediction, correct_target),
        "legacy_training_fusion": _per_sample_metrics(legacy_prediction, legacy_target),
        "legacy_training_fusion_shuffled_actions": _per_sample_metrics(legacy_shuffled_prediction, legacy_target),
        "legacy_prediction_change_after_shuffle_mse": float(
            F.mse_loss(legacy_prediction.float(), legacy_shuffled_prediction.float())
        ),
        "correct_vs_legacy_latent_mse": float(F.mse_loss(correct.float(), legacy.float())),
        "legacy_mapping_first_rows": mapping[: min(10, len(mapping))],
        "legacy_mapping_all_rows": mapping,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--clip-batch-size", type=int, default=5)
    parser.add_argument("--predictor-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_audit(
        index_path=args.index,
        encoder_path=args.encoder,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        windows=args.windows,
        clip_batch_size=args.clip_batch_size,
        predictor_batch_size=args.predictor_batch_size,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
