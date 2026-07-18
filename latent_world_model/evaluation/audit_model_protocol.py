"""Compare the released predictor under training-matched and shifted protocols.

The source VLA-JEPA objective jointly encodes one 8-frame clip, supplies three
latent context blocks and all 24 action tokens, then scores predictions against
the one-block-shifted latents from that *same* encoder call.  This diagnostic
keeps that exact path separate from strict-causal encodings used by the earlier
evaluation, so a protocol shift cannot be mistaken for predictor failure.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .. import LatentWorldModel
from .index import read_index
from .runner import _encode_clip_batch, _joint_clip, _pad_causal_clip


def _balanced_rows(index_path: Path, max_windows: int) -> list[dict[str, Any]]:
    """Select deterministic rows round-robin across suites."""
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_index(index_path):
        if not str(row["suite"]).startswith("_"):
            by_suite[str(row["suite"])].append(row)
    for rows in by_suite.values():
        rows.sort(key=lambda item: (int(item["task_id"]), int(item["episode_id"]), str(item["stage"])))
    suites = sorted(by_suite)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < max_windows:
        added = False
        for suite in suites:
            if cursor < len(by_suite[suite]):
                selected.append(by_suite[suite][cursor])
                added = True
                if len(selected) == max_windows:
                    break
        if not added:
            break
        cursor += 1
    return selected


def _one_row_per_task(index_path: Path) -> list[dict[str, Any]]:
    """Select the earliest deterministic window for every suite/task pair."""
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_index(index_path):
        if str(row["suite"]).startswith("_"):
            continue
        key = (str(row["suite"]), int(row["task_id"]))
        rank = (int(row["episode_id"]), int(row["query_frame"]), str(row["stage"]))
        if key not in chosen:
            chosen[key] = row
            continue
        current = chosen[key]
        current_rank = (int(current["episode_id"]), int(current["query_frame"]), str(current["stage"]))
        if rank < current_rank:
            chosen[key] = row
    return [chosen[key] for key in sorted(chosen)]


def _scalar_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = prediction.float()
    true = target.float()
    return {
        "l1": float(F.l1_loss(pred, true)),
        "mse": float(F.mse_loss(pred, true)),
        "cosine": float(F.cosine_similarity(pred, true, dim=-1).mean()),
    }


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.mse_loss(left.float(), right.float()))


def _predict(
    model: LatentWorldModel,
    context_blocks: Iterable[torch.Tensor],
    actions: np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    context = torch.cat(list(context_blocks), dim=0).unsqueeze(0).to(device)
    action_tensor = torch.as_tensor(actions).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model.predict_from_latents(context, action_tensor)
    return output[0].detach().cpu()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (float, int)) and key not in {"task_id", "episode_id", "query_frame", "query_row"}:
                numeric[key].append(float(value))
    return {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for key, values in sorted(numeric.items())
    }


def run_audit(
    *,
    index_path: Path,
    encoder_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    max_windows: int,
    clip_batch_size: int,
    device_name: str,
    one_per_task: bool = False,
) -> dict[str, Any]:
    device = torch.device(device_name)
    selected = _one_row_per_task(index_path) if one_per_task else _balanced_rows(index_path, max_windows)
    if len(selected) < 2:
        raise ValueError("protocol audit needs at least two windows for shuffled-action controls")

    action_rows: list[np.ndarray] = []
    for row in selected:
        with h5py.File(row["record_path"], "r") as handle:
            action_rows.append(np.asarray(handle["latent_action_tokens"][int(row["query_row"])], dtype=np.float32))

    model = LatentWorldModel(encoder_path)
    checkpoint_load = model.load_predictor_checkpoint(checkpoint_path, strict=True)
    model.to(device).eval()

    details: list[dict[str, Any]] = []
    patch_count: int | None = None
    for position, row in enumerate(selected):
        query_frame = int(row["query_frame"])
        with h5py.File(row["record_path"], "r") as handle:
            clips = [_joint_clip(handle, query_frame)]
            clips.extend(_pad_causal_clip(handle, query_frame + offset) for offset in (2, 4, 6, 8))
        encoded = _encode_clip_batch(model, clips, device=device, batch_size=clip_batch_size)
        joint_blocks = encoded[0]
        strict_blocks = [blocks[-1] for blocks in encoded[1:]]
        if patch_count is None:
            patch_count = int(joint_blocks.shape[1])

        correct_actions = action_rows[position]
        shuffled_actions = action_rows[(position + 1) % len(action_rows)]
        zero_actions = np.zeros_like(correct_actions)

        joint_correct = _predict(model, joint_blocks[:3], correct_actions, device)
        joint_zero = _predict(model, joint_blocks[:3], zero_actions, device)
        joint_shuffled = _predict(model, joint_blocks[:3], shuffled_actions, device)
        strict_correct = _predict(model, strict_blocks[:3], correct_actions, device)

        joint_target = torch.cat(list(joint_blocks[1:4]), dim=0)
        strict_target = torch.cat(list(strict_blocks[1:4]), dim=0)
        joint_metrics = _scalar_metrics(joint_correct, joint_target)
        strict_metrics = _scalar_metrics(strict_correct, strict_target)
        joint_zero_metrics = _scalar_metrics(joint_zero, joint_target)
        joint_shuffled_metrics = _scalar_metrics(joint_shuffled, joint_target)
        correct_action_tensor = torch.from_numpy(correct_actions)
        shuffled_action_tensor = torch.from_numpy(shuffled_actions)
        action_pair_mse = _mse(correct_action_tensor, shuffled_action_tensor)
        action_rms = float(correct_action_tensor.float().pow(2).mean().sqrt())

        detail: dict[str, Any] = {
            "suite": str(row["suite"]),
            "task_id": int(row["task_id"]),
            "episode_id": int(row["episode_id"]),
            "stage": str(row["stage"]),
            "query_row": int(row["query_row"]),
            "query_frame": query_frame,
            "train_matched_l1": joint_metrics["l1"],
            "train_matched_mse": joint_metrics["mse"],
            "train_matched_cosine": joint_metrics["cosine"],
            "strict_c3_l1": strict_metrics["l1"],
            "strict_c3_mse": strict_metrics["mse"],
            "strict_c3_cosine": strict_metrics["cosine"],
            "train_matched_zero_action_mse": joint_zero_metrics["mse"],
            "train_matched_shuffled_action_mse": joint_shuffled_metrics["mse"],
            "correct_minus_zero_target_mse": joint_metrics["mse"] - joint_zero_metrics["mse"],
            "correct_minus_shuffled_target_mse": joint_metrics["mse"] - joint_shuffled_metrics["mse"],
            "prediction_change_zero_mse": _mse(joint_correct, joint_zero),
            "prediction_change_shuffled_mse": _mse(joint_correct, joint_shuffled),
            "correct_vs_shuffled_action_mse": action_pair_mse,
            "correct_action_rms": action_rms,
            "correct_vs_shuffled_action_relative_rms": float(np.sqrt(action_pair_mse) / max(action_rms, 1e-12)),
            "correct_vs_shuffled_action_cosine": float(
                F.cosine_similarity(correct_action_tensor.float(), shuffled_action_tensor.float(), dim=-1).mean()
            ),
            "joint_vs_strict_context_mse": _mse(torch.cat(list(joint_blocks[:3]), dim=0), torch.cat(strict_blocks[:3], dim=0)),
            "joint_vs_strict_target_mse": _mse(joint_target, strict_target),
        }
        for horizon in range(3):
            start = horizon * patch_count
            end = start + patch_count
            matched_h = _scalar_metrics(joint_correct[start:end], joint_target[start:end])
            strict_h = _scalar_metrics(strict_correct[start:end], strict_target[start:end])
            detail[f"train_matched_h{horizon + 1}_mse"] = matched_h["mse"]
            detail[f"train_matched_h{horizon + 1}_l1"] = matched_h["l1"]
            detail[f"strict_c3_h{horizon + 1}_mse"] = strict_h["mse"]
            detail[f"strict_c3_h{horizon + 1}_l1"] = strict_h["l1"]
        details.append(detail)

        del encoded, joint_correct, joint_zero, joint_shuffled, strict_correct
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "index": str(index_path.resolve()),
        "encoder": str(encoder_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_load": checkpoint_load,
        "windows": len(details),
        "selection": "earliest deterministic window per suite/task" if one_per_task else "deterministic round-robin across suites",
        "protocols": {
            "train_matched": "one joint 8-frame encoding; C3 z0..z2; all 24 aligned tokens; targets z1..z3 from the same encoding",
            "strict_c3": "four independently causal 8-frame encodings ending at q+2,q+4,q+6,q+8; C3; same tokens; shifted strict targets",
        },
        "summary": _aggregate(details),
        "details": details,
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
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--clip-batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--one-per-task", action="store_true", help="ignore max-windows and select all 130 suite/task pairs")
    args = parser.parse_args()
    result = run_audit(
        index_path=args.index,
        encoder_path=args.encoder,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_windows=args.max_windows,
        clip_batch_size=args.clip_batch_size,
        device_name=args.device,
        one_per_task=args.one_per_task,
    )
    print(json.dumps({"windows": result["windows"], "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
