"""Audit temporal and artifact integrity of collected VLA-JEPA rollouts.

This module checks invariants that are stronger than file-count validation:

* policy queries occur at the expected control-frame indices;
* every stored 7-step action chunk reproduces the actions actually sent to
  LIBERO (including the collector's gripper binarization);
* latent-action tensors are finite, non-degenerate, and aligned one-to-one
  with policy queries;
* the status-labelled MP4 has the same number of frames as the lossless HDF5
  observation stream, and sampled decoded frames match up to video compression.

Run with::

    python -m latent_world_model.evaluation.audit_collection \
        --dataset-root datasets/vla_jepa_libero130_v3 \
        --output evaluation_outputs/audit/collection_integrity.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


REQUIRED_DATASETS = {
    "frames/agentview_rgb",
    "frames/eye_in_hand_rgb",
    "states",
    "executed_actions",
    "query_frame_index",
    "latent_action_tokens",
    "unnormalized_action_chunks",
}


def _video_path(dataset_root: Path, suite: str, record: Path, success: bool) -> Path:
    suffix = "success" if success else "failure"
    return dataset_root / "videos" / suite / f"{record.stem}_{suffix}.mp4"


def _sample_indices(length: int, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=np.int64))


def _decode_selected_rgb(video_path: Path, indices: np.ndarray) -> dict[int, np.ndarray]:
    """Decode selected frames sequentially to avoid unreliable random seeks."""
    wanted = {int(index) for index in indices}
    decoded: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_index = 0
        while wanted:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                decoded[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                wanted.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    return decoded


def audit_collection(
    dataset_root: Path,
    *,
    chunk_size: int = 7,
    video_samples_per_rollout: int = 3,
    pixel_mae_limit: float = 12.0,
) -> dict[str, Any]:
    records = sorted((dataset_root / "records").rglob("*.hdf5"))
    failures: list[dict[str, Any]] = []
    suites: Counter[str] = Counter()
    identities: set[tuple[str, int, int]] = set()

    total_frames = 0
    total_queries = 0
    total_action_rows = 0
    action_mismatch_rows = 0
    gripper_mismatch_rows = 0
    max_continuous_action_abs_error = 0.0
    latent_nonfinite_values = 0
    latent_all_zero_queries = 0
    latent_sum = 0.0
    latent_squared_sum = 0.0
    latent_value_count = 0

    videos_checked = 0
    video_frame_count_mismatches = 0
    video_sample_decode_misses = 0
    video_pixel_mae_values: list[float] = []

    def fail(record: Path, check: str, detail: Any) -> None:
        failures.append({"record": str(record.relative_to(dataset_root)), "check": check, "detail": detail})

    for record in records:
        try:
            with h5py.File(record, "r") as handle:
                missing = sorted(path for path in REQUIRED_DATASETS if path not in handle)
                if missing:
                    fail(record, "required_datasets", missing)
                    continue

                suite = str(handle.attrs["task_suite"])
                task_id = int(handle.attrs["task_id"])
                episode_id = int(handle.attrs["episode_id"])
                success = bool(handle.attrs["success"])
                identity = (suite, task_id, episode_id)
                if identity in identities:
                    fail(record, "duplicate_identity", identity)
                identities.add(identity)
                suites[suite] += 1

                frame_count = int(handle["frames/agentview_rgb"].shape[0])
                query_count = int(handle["query_frame_index"].shape[0])
                total_frames += frame_count
                total_queries += query_count

                frame_lengths = {
                    path: int(handle[path].shape[0])
                    for path in ("frames/agentview_rgb", "frames/eye_in_hand_rgb", "states", "executed_actions")
                }
                if len(set(frame_lengths.values())) != 1:
                    fail(record, "frame_dataset_lengths", frame_lengths)
                if int(handle.attrs.get("num_frames", -1)) != frame_count:
                    fail(record, "num_frames_attribute", {"attribute": int(handle.attrs.get("num_frames", -1)), "actual": frame_count})

                query_lengths = {
                    path: int(handle[path].shape[0])
                    for path in ("query_frame_index", "latent_action_tokens", "unnormalized_action_chunks")
                }
                if len(set(query_lengths.values())) != 1:
                    fail(record, "query_dataset_lengths", query_lengths)
                if int(handle.attrs.get("num_policy_queries", -1)) != query_count:
                    fail(
                        record,
                        "num_policy_queries_attribute",
                        {"attribute": int(handle.attrs.get("num_policy_queries", -1)), "actual": query_count},
                    )

                query_indices = handle["query_frame_index"][:]
                expected_indices = np.arange(0, frame_count, chunk_size, dtype=np.int64)
                if not np.array_equal(query_indices, expected_indices):
                    fail(
                        record,
                        "query_frame_index",
                        {"actual": query_indices.tolist(), "expected": expected_indices.tolist()},
                    )

                executed = handle["executed_actions"][:]
                chunks = handle["unnormalized_action_chunks"][:]
                for query_number, query_frame in enumerate(query_indices):
                    available = min(chunk_size, frame_count - int(query_frame))
                    if available <= 0:
                        fail(record, "query_out_of_range", int(query_frame))
                        continue
                    stored_chunk = chunks[query_number, :available]
                    actual = executed[int(query_frame) : int(query_frame) + available]
                    continuous_error = np.abs(stored_chunk[:, :6] - actual[:, :6])
                    max_continuous_action_abs_error = max(
                        max_continuous_action_abs_error,
                        float(continuous_error.max(initial=0.0)),
                    )
                    expected_gripper = 1.0 - 2.0 * (stored_chunk[:, 6] > 0.5)
                    continuous_bad = np.any(continuous_error > 1e-6, axis=1)
                    gripper_bad = actual[:, 6] != expected_gripper
                    total_action_rows += available
                    action_mismatch_rows += int(np.count_nonzero(continuous_bad | gripper_bad))
                    gripper_mismatch_rows += int(np.count_nonzero(gripper_bad))

                latents = handle["latent_action_tokens"][:].astype(np.float32)
                finite = np.isfinite(latents)
                latent_nonfinite_values += int(latents.size - np.count_nonzero(finite))
                if latents.size:
                    safe = np.where(finite, latents, 0.0).astype(np.float64)
                    latent_sum += float(safe.sum())
                    latent_squared_sum += float(np.square(safe).sum())
                    latent_value_count += int(np.count_nonzero(finite))
                    latent_all_zero_queries += int(np.count_nonzero(np.all(safe == 0.0, axis=(1, 2))))

                video_path = _video_path(dataset_root, suite, record, success)
                if not video_path.exists():
                    fail(record, "status_matched_video_missing", str(video_path.relative_to(dataset_root)))
                    continue

                capture = cv2.VideoCapture(str(video_path))
                reported_video_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()
                videos_checked += 1
                if reported_video_frames != frame_count:
                    video_frame_count_mismatches += 1
                    fail(
                        record,
                        "video_frame_count",
                        {"video": reported_video_frames, "hdf5": frame_count},
                    )

                indices = _sample_indices(frame_count, video_samples_per_rollout)
                decoded = _decode_selected_rgb(video_path, indices)
                for index in indices:
                    index = int(index)
                    if index not in decoded:
                        video_sample_decode_misses += 1
                        fail(record, "video_sample_decode", index)
                        continue
                    reference = handle["frames/agentview_rgb"][index].astype(np.float32)
                    frame = decoded[index].astype(np.float32)
                    if frame.shape != reference.shape:
                        fail(record, "video_frame_shape", {"index": index, "video": frame.shape, "hdf5": reference.shape})
                        continue
                    mae = float(np.mean(np.abs(frame - reference)))
                    video_pixel_mae_values.append(mae)
                    if mae > pixel_mae_limit:
                        fail(record, "video_pixel_mae", {"index": index, "mae": mae, "limit": pixel_mae_limit})
        except (OSError, KeyError, ValueError) as error:
            fail(record, "record_open_or_schema", repr(error))

    latent_mean = latent_sum / latent_value_count if latent_value_count else math.nan
    latent_variance = (
        latent_squared_sum / latent_value_count - latent_mean * latent_mean
        if latent_value_count
        else math.nan
    )
    video_mae = np.asarray(video_pixel_mae_values, dtype=np.float64)
    return {
        "dataset_root": str(dataset_root.resolve()),
        "passed": not failures,
        "records_checked": len(records),
        "unique_rollout_identities": len(identities),
        "suite_record_counts": dict(sorted(suites.items())),
        "total_frames": total_frames,
        "total_policy_queries": total_queries,
        "action_alignment": {
            "rows_checked": total_action_rows,
            "mismatch_rows": action_mismatch_rows,
            "gripper_mismatch_rows": gripper_mismatch_rows,
            "max_continuous_abs_error": max_continuous_action_abs_error,
        },
        "latent_actions": {
            "finite_values_checked": latent_value_count,
            "nonfinite_values": latent_nonfinite_values,
            "all_zero_queries": latent_all_zero_queries,
            "mean": latent_mean,
            "std": math.sqrt(max(latent_variance, 0.0)),
        },
        "videos": {
            "checked": videos_checked,
            "frame_count_mismatches": video_frame_count_mismatches,
            "sampled_frames": int(video_mae.size),
            "sample_decode_misses": video_sample_decode_misses,
            "pixel_mae_mean": float(video_mae.mean()) if video_mae.size else math.nan,
            "pixel_mae_p99": float(np.quantile(video_mae, 0.99)) if video_mae.size else math.nan,
            "pixel_mae_max": float(video_mae.max()) if video_mae.size else math.nan,
            "pixel_mae_limit": pixel_mae_limit,
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=7)
    parser.add_argument("--video-samples-per-rollout", type=int, default=3)
    parser.add_argument("--pixel-mae-limit", type=float, default=12.0)
    args = parser.parse_args()

    result = audit_collection(
        args.dataset_root,
        chunk_size=args.chunk_size,
        video_samples_per_rollout=args.video_samples_per_rollout,
        pixel_mae_limit=args.pixel_mae_limit,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
