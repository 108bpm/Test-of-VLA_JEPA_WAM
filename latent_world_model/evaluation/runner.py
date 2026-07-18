"""Memory-bounded LIBERO latent-world-model evaluation runner.

The runner deliberately keeps dataset reading and model inference in one
process (the VLA_JEPA environment has the matching Transformers/CUDA stack).
HDF5 records are opened one at a time, causal encoder blocks are kept as
float16 CPU tensors for one rollout, and prediction metrics are written as
JSONL after every completed window.  A partially completed directory can
therefore be resumed without changing the deterministic index or model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from .. import LatentWorldModel
from .index import read_index
from .metrics import compute_prediction_metrics


ACTION_TOKENS_PER_STEP = 8
CAUSAL_BLOCKS_FOR_H3 = 6
CONDITION_SPECS: Mapping[str, Mapping[str, object]] = {
    # Screening funnel (10 pre-registered conditions).
    "S0": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "correct", "view": "both"},
    "S1": {"protocol": "strict_causal", "context": 3, "horizon": 1, "action": "correct", "view": "both"},
    "S2": {"protocol": "strict_causal", "context": 1, "horizon": 3, "action": "correct", "view": "both"},
    "S3": {"protocol": "strict_causal", "context": 3, "horizon": 3, "action": "correct", "view": "both"},
    "S4": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "zero", "view": "both"},
    "S5": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "shuffled", "view": "both"},
    "S6": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "offset_plus_one", "view": "both"},
    "S7": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "correct", "view": "agentview"},
    "S8": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "correct", "view": "wrist"},
    "S9": {"protocol": "original_joint", "context": 1, "horizon": 1, "action": "correct", "view": "both"},
    # Targeted post-screening control: original joint encoding with C3 history.
    "X0": {"protocol": "original_joint", "context": 3, "horizon": 1, "action": "correct", "view": "both"},
    # Formal funnel (six conditions on all 1300 rollouts).
    "F0": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "correct", "view": "both"},
    "F1": {"protocol": "strict_causal", "context": 3, "horizon": 1, "action": "correct", "view": "both"},
    "F2": {"protocol": "strict_causal", "context": 1, "horizon": 3, "action": "correct", "view": "both"},
    "F3": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "zero", "view": "both"},
    "F4": {"protocol": "strict_causal", "context": 1, "horizon": 1, "action": "shuffled", "view": "both"},
    "F5": {"protocol": "original_joint", "context": 1, "horizon": 1, "action": "correct", "view": "both"},
}


@dataclass(frozen=True)
class RunnerConfig:
    dataset_root: Path
    index_path: Path
    output_dir: Path
    encoder_path: Path
    checkpoint_path: Path
    conditions: Tuple[str, ...]
    max_windows: Optional[int] = None
    clip_batch_size: int = 6
    device: str = "cuda"
    seed: int = 20260718
    bootstrap_replicates: int = 1000
    shard_id: int = 0
    num_shards: int = 1
    screening: bool = False
    rollouts_per_task: Optional[int] = None


def stable_row_id(row: Mapping[str, object]) -> str:
    return f"{row['suite']}/task{int(row['task_id']):03d}/episode{int(row['episode_id']):03d}/{row['stage']}"


def _git_provenance() -> Dict[str, object]:
    result: Dict[str, object] = {}
    for name, path in (
        ("latent_world_model", Path(__file__).resolve().parents[3]),
        ("VLA_JEPA", Path(__file__).resolve().parents[3] / "VLA-JEPA"),
    ):
        try:
            result[f"{name}_commit"] = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            result[f"{name}_status"] = subprocess.check_output(
                ["git", "-C", str(path), "status", "--short"], text=True, stderr=subprocess.DEVNULL
            ).splitlines()[:40]
        except (OSError, subprocess.CalledProcessError):
            result[f"{name}_commit"] = None
            result[f"{name}_status"] = []
    return result


def _load_rows(
    index_path: Path,
    conditions: Sequence[str],
    max_windows: Optional[int],
    *,
    shard_id: int = 0,
    num_shards: int = 1,
    screening: bool = False,
    rollouts_per_task: Optional[int] = None,
) -> List[dict]:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    rows = list(read_index(index_path))
    if screening or any(name.startswith("S") for name in conditions):
        # One deterministic rollout per task for the screening stage.  The
        # smallest episode id is selected before the three stage rows.
        chosen: Dict[Tuple[str, int], int] = {}
        for row in rows:
            key = (str(row["suite"]), int(row["task_id"]))
            chosen[key] = min(chosen.get(key, int(row["episode_id"])), int(row["episode_id"]))
        rows = [
            row
            for row in rows
            if not row["suite"].startswith("_")
            and int(row["episode_id"]) == chosen[(str(row["suite"]), int(row["task_id"]))]
        ]
    elif rollouts_per_task is not None:
        if rollouts_per_task < 1:
            raise ValueError("rollouts_per_task must be positive")
        # Deterministically subsample episodes per task.  For the requested
        # 130*5 setting this is episode ids 0,2,4,6,8 when ten episodes are
        # available, avoiding a first-half collection-order bias.
        episode_pool: MutableMapping[Tuple[str, int], List[int]] = defaultdict(list)
        for row in rows:
            key = (str(row["suite"]), int(row["task_id"]))
            episode_pool[key].append(int(row["episode_id"]))
        selected_episodes = {}
        for key, values in episode_pool.items():
            unique = sorted(set(values))
            if len(unique) <= rollouts_per_task:
                selected_episodes[key] = set(unique)
                continue
            if rollouts_per_task == 5 and len(unique) == 10:
                selected_episodes[key] = set(unique[::2][:5])
            else:
                positions = np.linspace(0, len(unique) - 1, rollouts_per_task).round().astype(int)
                selected_episodes[key] = {unique[int(position)] for position in positions}
        rows = [
            row
            for row in rows
            if int(row["episode_id"]) in selected_episodes[(str(row["suite"]), int(row["task_id"]))]
        ]
    rows.sort(key=lambda row: (str(row["suite"]), int(row["task_id"]), int(row["episode_id"]), str(row["stage"])))
    if max_windows is not None:
        rows = rows[:max_windows]
    if num_shards > 1:
        rows = rows[shard_id::num_shards]
    return rows


def _pad_causal_clip(handle: h5py.File, end_frame: int, view: str = "both") -> torch.Tensor:
    """Return a left-padded 8-frame clip ending at ``end_frame`` (exclusive)."""
    if end_frame <= 0 or end_frame > int(handle.attrs["num_frames"]):
        raise ValueError(f"causal endpoint {end_frame} outside record")
    start = max(0, end_frame - 8)
    agent = np.asarray(handle["frames/agentview_rgb"][start:end_frame])
    wrist = np.asarray(handle["frames/eye_in_hand_rgb"][start:end_frame])
    if agent.shape[0] < 8:
        pad = 8 - agent.shape[0]
        agent = np.concatenate([np.repeat(agent[:1], pad, axis=0), agent], axis=0)
        wrist = np.concatenate([np.repeat(wrist[:1], pad, axis=0), wrist], axis=0)
    if view == "agentview":
        wrist = agent.copy()
    elif view == "wrist":
        agent = wrist.copy()
    elif view != "both":
        raise ValueError(f"unknown view mode: {view}")
    # [V,T,C,H,W], matching the public model interface after batching.
    return torch.from_numpy(np.stack([agent, wrist], axis=0)).permute(0, 1, 4, 2, 3).contiguous()


def _joint_clip(handle: h5py.File, query_frame: int, view: str = "both") -> torch.Tensor:
    return _pad_causal_clip(handle, query_frame + 8, view=view)


def _encode_clip_batch(
    model: LatentWorldModel,
    clips: Sequence[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> List[torch.Tensor]:
    """Encode clips in small GPU batches and return blocks on CPU float16."""
    outputs: List[torch.Tensor] = []
    patch_count: Optional[int] = None
    for offset in range(0, len(clips), batch_size):
        batch = torch.stack(clips[offset : offset + batch_size], dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            encoded = model.encode_video(batch)
        if patch_count is None:
            patch_count = encoded.shape[1] // model.latent_steps
        blocks = encoded.view(encoded.shape[0], model.latent_steps, patch_count, encoded.shape[2])
        outputs.extend(blocks.detach().to("cpu", dtype=torch.float16).unbind(0))
        del batch, encoded, blocks
    return outputs


def _action_groups(action_row: np.ndarray, *, dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    arr = np.asarray(action_row)
    if arr.shape[0] != ACTION_TOKENS_PER_STEP * 3:
        raise ValueError(f"expected 24 latent action tokens, got {arr.shape}")
    return [torch.from_numpy(arr[i : i + ACTION_TOKENS_PER_STEP]).to(dtype=dtype) for i in range(0, arr.shape[0], ACTION_TOKENS_PER_STEP)]


def _read_action_row(path: str, query_row: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["latent_action_tokens"][query_row])


def _metrics_for_horizons(
    predicted: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    current: torch.Tensor,
    *,
    device: torch.device,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    current_b = current.unsqueeze(0).to(device)
    target_summaries = []
    prediction_summaries = []
    for horizon, (pred, target) in enumerate(zip(predicted, targets), start=1):
        pred_b = pred.to(device).unsqueeze(0)
        target_b = target.to(device).unsqueeze(0)
        values = compute_prediction_metrics(pred_b, target_b, current_b)
        for key, value in values.items():
            result[f"h{horizon}_{key}"] = float(value[0].detach().cpu())
        target_summaries.append(target.float().mean(dim=0).mean(dim=0))
        prediction_summaries.append(pred.float().mean(dim=0).mean(dim=0))
    # The unsuffixed fields always refer to the last requested horizon (H1 or
    # H3), making condition-level tables straightforward to consume.
    if predicted:
        for key in ("l1", "mse", "rmse", "normalized_mse", "target_variance", "prediction_variance", "persistence_mse", "persistence_ratio", "mean_token_cosine", "delta_cosine", "delta_norm_ratio", "prediction_variance_ratio"):
            result[key] = result[f"h{len(predicted)}_{key}"]
    result["prediction_summary"] = prediction_summaries[-1]
    result["target_summary"] = target_summaries[-1]
    return result


def _predict_h1(
    model: LatentWorldModel,
    context_blocks: Sequence[torch.Tensor],
    action_groups: Sequence[torch.Tensor],
    target: torch.Tensor,
    *,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor]:
    context = torch.cat(list(context_blocks), dim=0).unsqueeze(0).to(device)
    actions = torch.cat(list(action_groups), dim=0).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model.predict_from_latents(context, actions)
    prediction = output[0, -context_blocks[0].shape[0] :].detach().to("cpu", dtype=torch.float16)
    values = _metrics_for_horizons([prediction], [target], context_blocks[-1], device=device)
    return values, prediction


def _predict_ar(
    model: LatentWorldModel,
    initial_context: Sequence[torch.Tensor],
    action_windows: Sequence[Sequence[torch.Tensor]],
    targets: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> Tuple[Dict[str, float], List[torch.Tensor]]:
    """Autoregressively predict H3 while retaining C1 or C3 history."""
    history = list(initial_context)
    predictions: List[torch.Tensor] = []
    patch_count = history[0].shape[0]
    for action_window in action_windows:
        context = torch.cat(history[-len(action_window) :], dim=0).unsqueeze(0).to(device)
        actions = torch.cat(list(action_window), dim=0).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model.predict_from_latents(context, actions)
        prediction = output[0, -patch_count:].detach().to("cpu", dtype=torch.float16)
        predictions.append(prediction)
        history.append(prediction)
    values = _metrics_for_horizons(predictions, targets, initial_context[-1], device=device)
    return values, predictions


def _condition_action_data(
    spec: Mapping[str, object],
    current_groups: Sequence[torch.Tensor],
    next_groups: Optional[Sequence[torch.Tensor]],
    *,
    partner_groups: Optional[Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]] = None,
    offset_groups: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    mode = str(spec["action"])
    if mode == "zero":
        zero = [torch.zeros_like(group) for group in current_groups]
        return zero, [torch.zeros_like(group) for group in (next_groups or current_groups)]
    if mode == "shuffled":
        if partner_groups is None:
            raise ValueError("shuffled condition requires a partner action row")
        return list(partner_groups[0]), list(partner_groups[1])
    if mode == "offset_plus_one":
        if offset_groups is None:
            raise ValueError("offset_plus_one condition requires the next query action")
        return list(offset_groups), list(offset_groups)
    return list(current_groups), list(next_groups or current_groups)


def _run_condition(
    model: LatentWorldModel,
    condition: str,
    spec: Mapping[str, object],
    strict_blocks: Sequence[torch.Tensor],
    joint_blocks: Optional[Sequence[torch.Tensor]],
    current_groups: Sequence[torch.Tensor],
    next_groups: Optional[Sequence[torch.Tensor]],
    target_blocks: Sequence[torch.Tensor],
    *,
    device: torch.device,
    partner_groups: Optional[Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]] = None,
    offset_groups: Optional[Sequence[torch.Tensor]] = None,
) -> Dict[str, object]:
    context_steps = int(spec["context"])
    horizon = int(spec["horizon"])
    if str(spec["protocol"]) == "original_joint":
        if joint_blocks is None:
            raise ValueError("joint blocks are required for original_joint")
        blocks = joint_blocks
    else:
        blocks = strict_blocks
    truth_blocks = target_blocks
    cur, nxt = _condition_action_data(
        spec, current_groups, next_groups, partner_groups=partner_groups, offset_groups=offset_groups
    )
    if horizon == 1:
        # The current state is z2 and the target is z3 for both C1 and C3;
        # C3 additionally receives z0,z1 as historical context.
        context = [blocks[2]] if context_steps == 1 else [blocks[0], blocks[1], blocks[2]]
        actions = [cur[2]] if context_steps == 1 else cur
        values, _ = _predict_h1(model, context, actions, truth_blocks[3], device=device)
    else:
        if next_groups is None:
            raise ValueError("H3 conditions require a following policy query")
        # C1 and C3 share the same action trajectory: g2 at the current query,
        # then g0/g1 from the immediately following query.
        if context_steps == 1:
            initial = [blocks[2]]
            windows = [[cur[2]], [nxt[0]], [nxt[1]]]
        else:
            initial = [blocks[0], blocks[1], blocks[2]]
            windows = [[cur[0], cur[1], cur[2]], [cur[1], cur[2], nxt[0]], [cur[2], nxt[0], nxt[1]]]
        values, _ = _predict_ar(model, initial, windows, [truth_blocks[3], truth_blocks[4], truth_blocks[5]], device=device)
    return values


def _summarize_target(blocks: Sequence[torch.Tensor], current: torch.Tensor) -> float:
    delta = blocks[3].float() - current.float()
    return float(delta.pow(2).mean().sqrt())


def _safe_gripper_type(action_chunk: np.ndarray) -> str:
    arr = np.asarray(action_chunk)
    if arr.size == 0:
        return "unknown"
    # The last action dimension is the recorded gripper command.  Keep the
    # categorical label coarse so it remains meaningful across task suites.
    value = float(arr.reshape(-1, arr.shape[-1])[:, -1].mean())
    return "open" if value >= 0.5 else "closed"


def _load_completed(path: Path) -> set[Tuple[str, str]]:
    completed: set[Tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                completed.add((str(row["row_id"]), str(row["condition"])))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def _prepare_embedding_memmaps(output_dir: Path, conditions: Sequence[str], row_count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for condition in conditions:
        pred_path = output_dir / f"embeddings_{condition}_pred.npy"
        target_path = output_dir / f"embeddings_{condition}_target.npy"
        valid_path = output_dir / f"embeddings_{condition}_valid.npy"
        if pred_path.exists() and target_path.exists() and valid_path.exists():
            pred = np.lib.format.open_memmap(pred_path, mode="r+")
            target = np.lib.format.open_memmap(target_path, mode="r+")
            valid = np.lib.format.open_memmap(valid_path, mode="r+")
            if pred.shape != (row_count, 2048) or target.shape != (row_count, 2048) or valid.shape != (row_count,):
                raise ValueError(f"embedding cache shape mismatch for {condition}")
        else:
            pred = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.float16, shape=(row_count, 2048))
            target = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float16, shape=(row_count, 2048))
            valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=(row_count,))
            valid[:] = False
        arrays[condition] = (pred, target, valid)
    return arrays


def _write_config(config: RunnerConfig, rows: Sequence[dict]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_root": str(config.dataset_root.resolve()),
        "index_path": str(config.index_path.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "encoder_path": str(config.encoder_path.resolve()),
        "checkpoint_path": str(config.checkpoint_path.resolve()),
        "conditions": list(config.conditions),
        "max_windows": config.max_windows,
        "clip_batch_size": config.clip_batch_size,
        "device": config.device,
        "seed": config.seed,
        "shard_id": config.shard_id,
        "num_shards": config.num_shards,
        "screening": config.screening,
        "rollouts_per_task": config.rollouts_per_task,
        "row_count": len(rows),
        "created_unix": time.time(),
        "provenance": _git_provenance(),
    }
    (config.output_dir / "config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_evaluation(config: RunnerConfig) -> dict:
    """Run the requested screening/formal conditions and return a summary."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(config.device)
    rows = _load_rows(
        config.index_path,
        config.conditions,
        config.max_windows,
        shard_id=config.shard_id,
        num_shards=config.num_shards,
        screening=config.screening,
        rollouts_per_task=config.rollouts_per_task,
    )
    _write_config(config, rows)
    row_positions = {stable_row_id(row): i for i, row in enumerate(rows)}
    result_path = config.output_dir / "metrics.jsonl"
    completed = _load_completed(result_path)
    embeddings = _prepare_embedding_memmaps(config.output_dir, config.conditions, len(rows))

    # Build deterministic same-task/stage shuffles.  Each item points to a
    # different rollout; cyclic pairing avoids random-state dependence.
    by_group: MutableMapping[Tuple[str, int, str], List[dict]] = defaultdict(list)
    # Screening evaluates one rollout per task, but its shuffled-action control
    # still needs a distinct same-task partner.  Build the partner pool from
    # the complete index (the partner need not itself be evaluated).
    partner_pool_rows = list(read_index(config.index_path))
    for row in partner_pool_rows:
        by_group[(str(row["suite"]), int(row["task_id"]), str(row["stage"]))].append(row)
    partner_for = {}
    for group_rows in by_group.values():
        group_rows.sort(key=lambda row: int(row["episode_id"]))
        for i, row in enumerate(group_rows):
            partner_for[stable_row_id(row)] = group_rows[(i + 1) % len(group_rows)]

    model = LatentWorldModel(config.encoder_path)
    checkpoint_load = model.load_predictor_checkpoint(config.checkpoint_path, strict=True)
    model.to(device).eval()
    if checkpoint_load["missing_keys"] or checkpoint_load["unexpected_keys"]:
        raise RuntimeError(f"predictor checkpoint mismatch: {checkpoint_load}")
    (config.output_dir / "model_load.json").write_text(
        json.dumps({"checkpoint": str(config.checkpoint_path), "load": checkpoint_load}, indent=2) + "\n",
        encoding="utf-8",
    )

    grouped: MutableMapping[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(Path(row["record_path"]).resolve())].append(row)
    metrics_stream = result_path.open("a", encoding="utf-8")
    processed = 0
    skipped = 0
    errors = []
    try:
        for record_path, record_rows in grouped.items():
            record_rows.sort(key=lambda row: int(row["query_row"]))
            with h5py.File(record_path, "r") as handle:
                # Every requested H3 trajectory needs the six causal state
                # endpoints.  Sharing them across the three stage rows avoids
                # repeated encoder work while retaining strict causality.
                endpoints = sorted({int(row["query_frame"]) + 2 * (j + 1) for row in record_rows for j in range(CAUSAL_BLOCKS_FOR_H3)})
                strict_by_view: Dict[str, Dict[int, torch.Tensor]] = {}
                needed_views = {str(CONDITION_SPECS[c]["view"]) for c in config.conditions if str(CONDITION_SPECS[c]["protocol"]) == "strict_causal"}
                if any(str(CONDITION_SPECS[c]["protocol"]) == "original_joint" for c in config.conditions):
                    # Original-joint inputs are scored against their strict
                    # causal target, so the two-view strict target is needed
                    # even when no strict condition was requested explicitly.
                    needed_views.add("both")
                for view in sorted(needed_views):
                    clips = [_pad_causal_clip(handle, endpoint, view=view) for endpoint in endpoints]
                    encoded = _encode_clip_batch(model, clips, device=device, batch_size=config.clip_batch_size)
                    strict_by_view[view] = {endpoint: blocks[-1] for endpoint, blocks in zip(endpoints, encoded)}
                joint_by_row: Dict[str, Sequence[torch.Tensor]] = {}
                if any(str(CONDITION_SPECS[c]["protocol"]) == "original_joint" for c in config.conditions):
                    clips = [_joint_clip(handle, int(row["query_frame"]), view="both") for row in record_rows]
                    encoded = _encode_clip_batch(model, clips, device=device, batch_size=config.clip_batch_size)
                    for row, blocks in zip(record_rows, encoded):
                        joint_by_row[stable_row_id(row)] = blocks

                action_table = np.asarray(handle["latent_action_tokens"][:])
                action_chunks = np.asarray(handle["unnormalized_action_chunks"][:])
                for row in record_rows:
                    row_id = stable_row_id(row)
                    qrow = int(row["query_row"])
                    qframe = int(row["query_frame"])
                    strict_blocks_by_view = {
                        view: [strict_by_view[view][qframe + 2 * (j + 1)] for j in range(CAUSAL_BLOCKS_FOR_H3)]
                        for view in needed_views
                    }
                    current_groups = _action_groups(action_table[qrow])
                    next_groups = None
                    if qrow + 1 < action_table.shape[0]:
                        next_groups = _action_groups(action_table[qrow + 1])
                    offset_groups = _action_groups(action_table[qrow + 1]) if qrow + 1 < action_table.shape[0] else None
                    partner_row = partner_for.get(row_id)
                    partner_groups = None
                    if partner_row is not None:
                        partner_current = _action_groups(_read_action_row(str(partner_row["record_path"]), int(partner_row["query_row"])))
                        partner_next_row = int(partner_row["query_row"]) + 1
                        with h5py.File(str(partner_row["record_path"]), "r") as partner_handle:
                            if partner_next_row < partner_handle["latent_action_tokens"].shape[0]:
                                partner_next = _action_groups(np.asarray(partner_handle["latent_action_tokens"][partner_next_row]))
                            else:
                                partner_next = partner_current
                        partner_groups = (partner_current, partner_next)

                    action_norm = float(np.asarray(action_chunks[qrow]).reshape(-1).astype(np.float64).mean() ** 2) ** 0.5
                    # Use the actual action vector RMS as an interpretable
                    # scale rather than the high-dimensional token norm.
                    action_norm = float(np.asarray(action_chunks[qrow], dtype=np.float64).reshape(-1).std())
                    base_metadata = {
                        "row_id": row_id,
                        "suite": str(row["suite"]),
                        "task_id": int(row["task_id"]),
                        "episode_id": int(row["episode_id"]),
                        "success": bool(row["success"]),
                        "stage": str(row["stage"]),
                        "query_row": qrow,
                        "query_frame": qframe,
                        "action_norm": action_norm,
                        "gripper_type": _safe_gripper_type(action_chunks[qrow]),
                    }
                    for condition in config.conditions:
                        if (row_id, condition) in completed:
                            skipped += 1
                            continue
                        spec = CONDITION_SPECS[condition]
                        view = str(spec["view"])
                        strict_blocks = strict_blocks_by_view.get(view)
                        if strict_blocks is None:
                            raise RuntimeError(f"missing encoded view {view}")
                        joint_blocks = joint_by_row.get(row_id)
                        try:
                            values = _run_condition(
                                model,
                                condition,
                                spec,
                                strict_blocks,
                                joint_blocks,
                                current_groups,
                                next_groups,
                                strict_blocks,
                                device=device,
                                partner_groups=partner_groups,
                                offset_groups=offset_groups,
                            )
                            pred_summary = values.pop("prediction_summary")
                            target_summary = values.pop("target_summary")
                            values.update(base_metadata)
                            values.update(
                                {
                                    "condition": condition,
                                    "protocol": str(spec["protocol"]),
                                    "context_steps": int(spec["context"]),
                                    "horizon": int(spec["horizon"]),
                                    "action_mode": str(spec["action"]),
                                    "view": view,
                                    "target_delta_rms": _summarize_target(strict_blocks, strict_blocks[2]),
                                    "timestamp": time.time(),
                                }
                            )
                            position = row_positions[row_id]
                            pred_mm, target_mm, valid_mm = embeddings[condition]
                            pred_mm[position] = pred_summary.detach().cpu().numpy().astype(np.float16)
                            target_mm[position] = target_summary.detach().cpu().numpy().astype(np.float16)
                            valid_mm[position] = True
                            pred_mm.flush(); target_mm.flush(); valid_mm.flush()
                            metrics_stream.write(json.dumps(values, ensure_ascii=False, sort_keys=True) + "\n")
                            metrics_stream.flush()
                            completed.add((row_id, condition))
                            processed += 1
                        except (RuntimeError, ValueError, KeyError) as exc:
                            errors.append({"row_id": row_id, "condition": condition, "error": repr(exc)})
                            if "outside record" in str(exc):
                                continue
                            raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        metrics_stream.close()
        for pred_mm, target_mm, valid_mm in embeddings.values():
            pred_mm.flush(); target_mm.flush(); valid_mm.flush()

    summary = {
        "rows": len(rows),
        "conditions": list(config.conditions),
        "processed": processed,
        "skipped_existing": skipped,
        "errors": errors,
        "metrics_path": str(result_path),
        "completed": len(completed),
        "expected": len(rows) * len(config.conditions),
        "checkpoint_load": checkpoint_load,
    }
    (config.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", dest="index_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--encoder", dest="encoder_path", type=Path, required=True)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--clip-batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--screening", action="store_true", help="select one deterministic rollout per task")
    parser.add_argument("--rollouts-per-task", type=int, default=None, help="deterministically retain this many episodes per task")
    args = parser.parse_args(argv)
    conditions = tuple(args.conditions)
    unknown = sorted(set(conditions) - set(CONDITION_SPECS))
    if unknown:
        parser.error(f"unknown conditions: {unknown}")
    return RunnerConfig(
        dataset_root=args.dataset_root,
        index_path=args.index_path,
        output_dir=args.output_dir,
        encoder_path=args.encoder_path,
        checkpoint_path=args.checkpoint_path,
        conditions=conditions,
        max_windows=args.max_windows,
        clip_batch_size=args.clip_batch_size,
        device=args.device,
        seed=args.seed,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        screening=args.screening,
        rollouts_per_task=args.rollouts_per_task,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = _parse_args(argv)
    summary = run_evaluation(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
