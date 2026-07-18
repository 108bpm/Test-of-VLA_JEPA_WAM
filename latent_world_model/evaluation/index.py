"""Deterministic, memory-bounded indexing for the v3 LIBERO rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "frames/agentview_rgb",
    "frames/eye_in_hand_rgb",
    "states",
    "executed_actions",
    "query_frame_index",
    "latent_action_tokens",
    "unnormalized_action_chunks",
)
STAGES = {"early": 0.20, "middle": 0.50, "late": 0.80}


def _valid_record(handle: h5py.File) -> bool:
    if not all(path in handle for path in REQUIRED_DATASETS):
        return False
    if "num_frames" not in handle.attrs or "success" not in handle.attrs:
        return False
    frames = int(handle.attrs["num_frames"])
    if handle["frames/agentview_rgb"].shape[0] != frames:
        return False
    if handle["frames/eye_in_hand_rgb"].shape[0] != frames:
        return False
    query_count = handle["query_frame_index"].shape[0]
    return (
        handle["latent_action_tokens"].shape[0] == query_count
        and handle["unnormalized_action_chunks"].shape[0] == query_count
        and np.isfinite(handle["latent_action_tokens"][: min(query_count, 1)]).all()
    )


def _select_queries(
    query_indices: np.ndarray,
    num_frames: int,
    stages: Mapping[str, float],
    *,
    required_future_frames: int = 12,
) -> List[tuple[str, int]]:
    """Select at most one query per stage with enough room for all H3 targets."""
    legal = np.flatnonzero(query_indices + required_future_frames <= num_frames)
    if legal.size == 0:
        return []
    selected = []
    used = set()
    for stage, fraction in stages.items():
        order = sorted(legal.tolist(), key=lambda i: (abs(float(query_indices[i]) / max(num_frames - 1, 1) - fraction), i))
        choice = next((i for i in order if i not in used), order[0])
        selected.append((stage, int(choice)))
        used.add(choice)
    return selected


def build_rollout_index(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    stages: Optional[Mapping[str, float]] = None,
    required_future_frames: int = 12,
) -> dict:
    """Build a JSONL query index without copying video/latent data.

    The index is deterministic and stores only metadata.  Every selected query
    has an eight-frame future window and a corresponding latent-action row.
    Existing output is replaced only when the caller explicitly invokes this
    function; evaluation commands should use a separate output directory.
    """
    root = Path(dataset_root).resolve()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage_spec = dict(stages or STAGES)
    if required_future_frames < 8:
        raise ValueError("required_future_frames must leave room for the 8-frame encoder clip")
    records = sorted((root / "records").glob("*/*.hdf5"))
    rows = []
    invalid = []
    for record_path in records:
        try:
            with h5py.File(record_path, "r") as handle:
                if not _valid_record(handle):
                    invalid.append(str(record_path))
                    continue
                suite = str(handle.attrs["task_suite"])
                task_id = int(handle.attrs["task_id"])
                episode_id = int(handle.attrs["episode_id"])
                num_frames = int(handle.attrs["num_frames"])
                success = bool(handle.attrs["success"])
                query_indices = np.asarray(handle["query_frame_index"][:], dtype=np.int64)
                token_count = int(handle["latent_action_tokens"].shape[1])
                token_dim = int(handle["latent_action_tokens"].shape[2])
                action_shape = list(handle["unnormalized_action_chunks"].shape[1:])
                instruction = str(handle.attrs.get("instruction", ""))
            video_dir = root / "videos" / suite
            stem = record_path.stem
            video_matches = sorted(video_dir.glob(f"{stem}_*.mp4"))
            if len(video_matches) != 1:
                invalid.append(str(record_path))
                continue
            for stage, query_row in _select_queries(
                query_indices, num_frames, stage_spec, required_future_frames=required_future_frames
            ):
                rows.append(
                    {
                        "record_path": str(record_path),
                        "video_path": str(video_matches[0]),
                        "suite": suite,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "success": success,
                        "instruction": instruction,
                        "stage": stage,
                        "query_row": query_row,
                        "query_frame": int(query_indices[query_row]),
                        "num_frames": num_frames,
                        "latent_action_shape": [token_count, token_dim],
                        "action_chunk_shape": action_shape,
                    }
                )
        except (OSError, ValueError, KeyError):
            invalid.append(str(record_path))
    rows.sort(key=lambda row: (row["suite"], row["task_id"], row["episode_id"], row["stage"]))
    with out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "dataset_root": str(root),
        "index_path": str(out),
        "records_seen": len(records),
        "records_invalid": len(invalid),
        "windows": len(rows),
        "stages": stage_spec,
        "required_future_frames": required_future_frames,
        "invalid_records": invalid,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def read_index(path: str | Path) -> Iterable[dict]:
    """Stream index rows without loading all metadata into memory."""
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
