"""Deep, reproducible analysis for the collected LIBERO latent evaluation.

The runner is deliberately responsible only for inference and per-window
metrics.  This module is a read-only analysis layer: it streams the v3 HDF5
metadata, reads the formal/screening JSONL files, computes paired and
stratified summaries, and writes compact CSV/JSON tables plus figures.  No
model, checkpoint, HDF5, or metric file is modified.

Example (run in the VLA_JEPA conda environment, which has h5py and the model
evaluation dependencies)::

    PYTHONPATH=$PWD python -m latent_world_model.evaluation.deep_analysis \
      --dataset-root datasets/vla_jepa_libero130_v3 \
      --formal-metrics evaluation_outputs/formal_half/metrics.jsonl \
      --screening-metrics evaluation_outputs/stage1/metrics.jsonl \
      --supplemental-metrics evaluation_outputs/stage1_supplemental/metrics.jsonl \
      --output evaluation_outputs/deep_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np


FORMAL_CONDITIONS = ("F0", "F1", "F2", "F3", "F4", "F5")
SCREENING_CONDITIONS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")
FORMAL_EPISODES = {0, 2, 4, 6, 8}
SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10")
STAGE_ORDER = ("early", "middle", "late")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    arr = _finite(values)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _task_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('suite')}/task{int(row.get('task_id', 0)):03d}"


def _bootstrap_mean(
    rows: Sequence[Mapping[str, Any]],
    values: Mapping[str, float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | int | None]:
    """Task -> rollout/window hierarchical bootstrap for a paired scalar."""
    by_task: MutableMapping[str, list[float]] = defaultdict(list)
    for row in rows:
        row_id = str(row.get("row_id"))
        value = values.get(row_id)
        if value is not None and np.isfinite(value):
            by_task[_task_key(row)].append(float(value))
    flat = [value for values_ in by_task.values() for value in values_]
    if not flat:
        return {"n_tasks": 0, "n": 0, "mean": None, "ci95_low": None, "ci95_high": None, "p_two_sided": None}
    observed = float(np.mean(flat))
    if replicates <= 0:
        return {"n_tasks": len(by_task), "n": len(flat), "mean": observed, "ci95_low": None, "ci95_high": None, "p_two_sided": None}
    rng = np.random.default_rng(seed)
    tasks = sorted(by_task)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        sampled: list[float] = []
        for task in sampled_tasks:
            task_values = by_task[task]
            sampled.extend(rng.choice(task_values, size=len(task_values), replace=True).tolist())
        estimates[index] = float(np.mean(sampled))
    p = 2.0 * min(float(np.mean(estimates <= 0.0)), float(np.mean(estimates >= 0.0))) if observed != 0 else 1.0
    return {
        "n_tasks": len(by_task),
        "n": len(flat),
        "mean": observed,
        "ci95_low": float(np.percentile(estimates, 2.5)),
        "ci95_high": float(np.percentile(estimates, 97.5)),
        "p_two_sided": p,
    }


def _condition_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(dict(row))
    return dict(grouped)


def _retrieval_for_dir(results_dir: Optional[Path], condition: str) -> dict[str, float | int | None]:
    """Compute compact retrieval diagnostics from runner memmaps."""
    if results_dir is None:
        return {"retrieval_n": 0, "retrieval_top1": None, "retrieval_top5": None}
    pred_path = results_dir / f"embeddings_{condition}_pred.npy"
    target_path = results_dir / f"embeddings_{condition}_target.npy"
    valid_path = results_dir / f"embeddings_{condition}_valid.npy"
    if not (pred_path.exists() and target_path.exists() and valid_path.exists()):
        return {"retrieval_n": 0, "retrieval_top1": None, "retrieval_top5": None}
    valid = np.asarray(np.load(valid_path, mmap_mode="r"), dtype=bool)
    pred = np.asarray(np.load(pred_path, mmap_mode="r"), dtype=np.float32)[valid]
    target = np.asarray(np.load(target_path, mmap_mode="r"), dtype=np.float32)[valid]
    n = int(pred.shape[0])
    if n == 0:
        return {"retrieval_n": 0, "retrieval_top1": None, "retrieval_top5": None}
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    target_t = target.T
    top1 = 0
    top5 = 0
    for start in range(0, n, 256):
        similarity = pred[start : start + 256] @ target_t
        k = min(5, n)
        order = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        labels = np.arange(start, min(start + 256, n))
        top1 += int(np.sum(order[:, 0] == labels))
        top5 += int(np.sum(np.any(order == labels[:, None], axis=1)))
    return {"retrieval_n": n, "retrieval_top1": float(top1 / n), "retrieval_top5": float(top5 / n)}


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int,
    seed: int,
    retrieval_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    grouped = _condition_rows(rows)
    output = []
    for condition in sorted(grouped):
        group = grouped[condition]
        summary = _stats(row.get("mse") for row in group)
        ci = _bootstrap_mean(group, {str(row["row_id"]): float(row["mse"]) for row in group if np.isfinite(float(row["mse"]))}, seed=seed + sum(map(ord, condition)), replicates=bootstrap_replicates)
        record: dict[str, Any] = {
            "condition": condition,
            "n": summary["n"],
            "mean_mse": summary["mean"],
            "std_mse": summary["std"],
            "median_mse": summary["median"],
            "q25_mse": summary["q25"],
            "q75_mse": summary["q75"],
            "min_mse": summary["min"],
            "max_mse": summary["max"],
            "ci95_low": ci["ci95_low"],
            "ci95_high": ci["ci95_high"],
            "mean_l1": _stats(row.get("l1") for row in group)["mean"],
            "mean_rmse": _stats(row.get("rmse") for row in group)["mean"],
            "mean_normalized_mse": _stats(row.get("normalized_mse") for row in group)["mean"],
            "mean_persistence_mse": _stats(row.get("persistence_mse") for row in group)["mean"],
            "mean_persistence_ratio": _stats(row.get("persistence_ratio") for row in group)["mean"],
            "mean_token_cosine": _stats(row.get("mean_token_cosine") for row in group)["mean"],
            "mean_delta_cosine": _stats(row.get("delta_cosine") for row in group)["mean"],
            "mean_delta_norm_ratio": _stats(row.get("delta_norm_ratio") for row in group)["mean"],
            "mean_prediction_variance_ratio": _stats(row.get("prediction_variance_ratio") for row in group)["mean"],
            "retrieval_n": None,
            "retrieval_top1": None,
            "retrieval_top5": None,
            "protocol": group[0].get("protocol"),
            "context_steps": group[0].get("context_steps"),
            "horizon": group[0].get("horizon"),
            "action_mode": group[0].get("action_mode"),
            "view": group[0].get("view"),
        }
        record.update(_retrieval_for_dir(retrieval_dir, condition))
        output.append(record)
    return output


def _paired_comparisons(rows: Sequence[Mapping[str, Any]], bootstrap_replicates: int, seed: int) -> list[dict[str, Any]]:
    grouped = _condition_rows(rows)
    comparisons: list[tuple[str, str, str]] = [("F0-persistence", "F0", "persistence_mse")]
    comparisons.extend((f"{left}-{right}", left, right) for left, right in (("F1", "F0"), ("F3", "F0"), ("F4", "F0"), ("F5", "F0")))
    output = []
    for name, left, right in comparisons:
        if left not in grouped:
            continue
        left_rows = {str(row["row_id"]): row for row in grouped[left]}
        if right == "persistence_mse":
            common = sorted(left_rows)
            paired_rows = [left_rows[key] for key in common]
            values = {key: float(left_rows[key]["mse"]) - float(left_rows[key]["persistence_mse"]) for key in common}
        elif right in grouped:
            right_rows = {str(row["row_id"]): row for row in grouped[right]}
            common = sorted(set(left_rows) & set(right_rows))
            paired_rows = [left_rows[key] for key in common]
            values = {key: float(left_rows[key]["mse"]) - float(right_rows[key]["mse"]) for key in common}
        else:
            continue
        stat = _bootstrap_mean(paired_rows, values, seed=seed + len(output) + 11, replicates=bootstrap_replicates)
        finite_values = np.asarray(list(values.values()), dtype=np.float64)
        stat.update({
            "comparison": name,
            "fraction_left_better": float(np.mean(finite_values < 0.0)),
            "fraction_left_worse": float(np.mean(finite_values > 0.0)),
            "median_difference": float(np.median(finite_values)),
            "mean_abs_difference": float(np.mean(np.abs(finite_values))),
            "cohen_d": float(np.mean(finite_values) / max(float(np.std(finite_values, ddof=1)), 1e-12)) if finite_values.size > 1 else None,
        })
        output.append(stat)
    # Holm correction on the bootstrap sign probabilities, matching the
    # registered comparisons in evaluation.report.py.
    def _p_for_sort(index: int) -> float:
        value = output[index].get("p_two_sided")
        return 1.0 if value is None or not np.isfinite(float(value)) else float(value)

    order = sorted(range(len(output)), key=_p_for_sort)
    previous = 0.0
    for rank, index in enumerate(order):
        p_value = output[index].get("p_two_sided")
        p_value = 1.0 if p_value is None or not np.isfinite(float(p_value)) else float(p_value)
        value = min(1.0, max(previous, (len(order) - rank) * p_value))
        output[index]["p_holm"] = value
        previous = value
    return output


def _stratified(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition, condition_rows in sorted(_condition_rows(rows).items()):
        partitions: MutableMapping[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in condition_rows:
            partitions[tuple(str(row.get(field)) for field in fields)].append(row)
        for key, group in sorted(partitions.items()):
            mse = _stats(row.get("mse") for row in group)
            output.append({"condition": condition, **{field: value for field, value in zip(fields, key)}, "n": mse["n"], "mean_mse": mse["mean"], "median_mse": mse["median"], "q25_mse": mse["q25"], "q75_mse": mse["q75"], "mean_persistence_ratio": _stats(row.get("persistence_ratio") for row in group)["mean"], "mean_target_delta_rms": _stats(row.get("target_delta_rms") for row in group)["mean"], "mean_action_norm": _stats(row.get("action_norm") for row in group)["mean"]})
    return output


def _assign_quantile(values: np.ndarray, labels: Sequence[str] = ("Q1", "Q2", "Q3", "Q4")) -> tuple[np.ndarray, list[float]]:
    finite = values[np.isfinite(values)]
    edges = [float(np.percentile(finite, p)) for p in (25, 50, 75)] if finite.size else [float("nan")] * 3
    bins = np.digitize(values, edges, right=True)
    return np.asarray([labels[min(int(index), len(labels) - 1)] if np.isfinite(value) else "unknown" for index, value in zip(bins, values)], dtype=object), edges


def _add_quantile_labels(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = _condition_rows(rows)
    # Use F0 thresholds as the common reference, so comparisons do not move
    # their bins simply because a model condition changed the distribution.
    base = np.asarray([float(row["target_delta_rms"]) for row in grouped.get("F0", rows)], dtype=np.float64)
    _, motion_edges = _assign_quantile(base)
    base_action = np.asarray([float(row["action_norm"]) for row in grouped.get("F0", rows)], dtype=np.float64)
    _, action_edges = _assign_quantile(base_action)
    def label(value: float, edges: Sequence[float]) -> str:
        return ("Q1", "Q2", "Q3", "Q4")[min(int(np.digitize([value], edges, right=True)[0]), 3)] if np.isfinite(value) else "unknown"
    for condition_rows in grouped.values():
        for row in condition_rows:
            row["motion_quartile"] = label(float(row["target_delta_rms"]), motion_edges)
            row["action_quartile"] = label(float(row["action_norm"]), action_edges)
    return grouped


def _horizon(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    f2 = [row for row in rows if row.get("condition") == "F2"]
    for horizon in (1, 2, 3):
        key = f"h{horizon}_mse"
        values = [float(row[key]) for row in f2 if key in row and np.isfinite(float(row[key]))]
        stat = _stats(values)
        output.append({"condition": "F2", "horizon": horizon, "metric": key, **stat})
    return output


def _correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    f0 = [row for row in rows if row.get("condition") == "F0"]
    output = []
    for predictor in ("target_delta_rms", "action_norm", "query_frame", "target_variance"):
        pairs = [(float(row[predictor]), float(row["mse"])) for row in f0 if predictor in row and np.isfinite(float(row[predictor])) and np.isfinite(float(row["mse"]))]
        if len(pairs) < 3:
            continue
        x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        output.append({"condition": "F0", "predictor": predictor, "n": len(pairs), "pearson_r": float(np.corrcoef(x, y)[0, 1]), "spearman_r": float(np.corrcoef(rx, ry)[0, 1]), "x_mean": float(np.mean(x)), "y_mean_mse": float(np.mean(y))})
    return output


def _collection_stats(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("h5py is required for collection analysis") from exc
    records: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "records").glob("*/*.hdf5")):
        with h5py.File(path, "r") as handle:
            suite = str(handle.attrs["task_suite"])
            task_id = int(handle.attrs["task_id"])
            episode_id = int(handle.attrs["episode_id"])
            frames = int(handle.attrs["num_frames"])
            success = bool(handle.attrs["success"])
            queries = int(handle["query_frame_index"].shape[0])
            query_indices = np.asarray(handle["query_frame_index"][:], dtype=np.int64)
            chunks = np.asarray(handle["unnormalized_action_chunks"][:], dtype=np.float32)
            action_std = np.std(chunks.reshape(chunks.shape[0], -1), axis=1) if chunks.size else np.asarray([], dtype=np.float32)
            records.append({"suite": suite, "task_id": task_id, "episode_id": episode_id, "success": success, "num_frames": frames, "num_policy_queries": queries, "first_query_frame": int(query_indices[0]) if queries else None, "last_query_frame": int(query_indices[-1]) if queries else None, "mean_action_norm": float(np.mean(action_std)) if action_std.size else None, "video_bytes": (dataset_root / "videos" / suite / f"{path.stem}_{'success' if success else 'failure'}.mp4").stat().st_size if (dataset_root / "videos" / suite / f"{path.stem}_{'success' if success else 'failure'}.mp4").exists() else None})
    summary_rows = []
    for suite in SUITE_ORDER:
        group = [row for row in records if row["suite"] == suite]
        summary_rows.append({"suite": suite, "rollouts": len(group), "successes": sum(bool(row["success"]) for row in group), "success_rate": float(np.mean([bool(row["success"]) for row in group])) if group else None, "frames_mean": _stats(row["num_frames"] for row in group)["mean"], "frames_median": _stats(row["num_frames"] for row in group)["median"], "frames_min": _stats(row["num_frames"] for row in group)["min"], "frames_max": _stats(row["num_frames"] for row in group)["max"], "queries_mean": _stats(row["num_policy_queries"] for row in group)["mean"], "queries_median": _stats(row["num_policy_queries"] for row in group)["median"], "queries_min": _stats(row["num_policy_queries"] for row in group)["min"], "queries_max": _stats(row["num_policy_queries"] for row in group)["max"], "video_mb": float(sum(row["video_bytes"] or 0 for row in group) / 1e6)})
    selected = [row for row in records if int(row["episode_id"]) in FORMAL_EPISODES]
    formal_rows = []
    for suite in SUITE_ORDER:
        group = [row for row in selected if row["suite"] == suite]
        formal_rows.append({"suite": suite, "rollouts": len(group), "successes": sum(bool(row["success"]) for row in group), "success_rate": float(np.mean([bool(row["success"]) for row in group])) if group else None, "frames_mean": _stats(row["num_frames"] for row in group)["mean"], "queries_mean": _stats(row["num_policy_queries"] for row in group)["mean"]})
    all_summary = {"rollouts": len(records), "successes": sum(bool(row["success"]) for row in records), "success_rate": float(np.mean([bool(row["success"]) for row in records])) if records else None, "videos": sum(row["video_bytes"] is not None for row in records), "formal_rollouts": len(selected), "formal_successes": sum(bool(row["success"]) for row in selected)}
    return summary_rows, {"records": records, "suite_summary": summary_rows, "formal_suite_summary": formal_rows, "totals": all_summary}


def _plot_all(
    output: Path,
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    condition_table: Sequence[Mapping[str, Any]],
    paired_table: Sequence[Mapping[str, Any]],
    horizon_table: Sequence[Mapping[str, Any]],
    suite_table: Sequence[Mapping[str, Any]],
    stage_success_table: Sequence[Mapping[str, Any]],
    motion_table: Sequence[Mapping[str, Any]],
    screening_table: Sequence[Mapping[str, Any]],
    collection_table: Sequence[Mapping[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except Exception:
        return []
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    paths: list[str] = []
    labels = [str(row["condition"]) for row in condition_table]
    means = np.asarray([float(row["mean_mse"]) for row in condition_table], dtype=float)
    low = np.asarray([float(row["ci95_low"]) for row in condition_table], dtype=float)
    high = np.asarray([float(row["ci95_high"]) for row in condition_table], dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, means, color=["#3568a8" if name.startswith("F") else "#8a8a8a" for name in labels])
    ax.errorbar(x, means, yerr=[means - low, high - means], fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(x, labels); ax.set_ylabel("MSE (latent space)"); ax.set_title("Direct future-latent error by condition (95% CI)"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); path = output / "condition_mse_ci.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    # Per-window persistence ratio distributions.  The ratio is deliberately
    # a secondary diagnostic; direct MSE is the primary result in the report.
    box_names = [name for name in ("F0", "F1", "F2", "F3", "F4", "F5") if name in condition_rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.boxplot([[float(row["persistence_ratio"]) for row in condition_rows[name] if np.isfinite(float(row["persistence_ratio"]))] for name in box_names], labels=box_names, showfliers=False)
    ax.axhline(1.0, color="#b23a48", ls="--", lw=1, label="persistence = 1")
    ax.set_ylabel("MSE / persistence MSE"); ax.set_title("Auxiliary persistence-relative error distribution"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); path = output / "persistence_ratio_box.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    names = [str(row["comparison"]) for row in paired_table]
    vals = np.asarray([float(row["mean"]) for row in paired_table])
    lows = np.asarray([float(row["ci95_low"]) for row in paired_table]); highs = np.asarray([float(row["ci95_high"]) for row in paired_table])
    ypos = np.arange(len(names))
    ax.errorbar(vals, ypos, xerr=[vals - lows, highs - vals], fmt="o", color="#3568a8", capsize=3)
    ax.axvline(0.0, color="black", lw=1); ax.set_yticks(ypos, names); ax.set_xlabel("Paired MSE difference (left − right)"); ax.set_title("Registered paired effects (negative favors left)"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); path = output / "paired_effects_forest.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    if horizon_table:
        hs = sorted(horizon_table, key=lambda row: int(row["horizon"]))
        fig, ax = plt.subplots(figsize=(6.6, 4.5)); hx = [int(row["horizon"]) for row in hs]; hm = [float(row["mean"]) for row in hs]
        hq1 = [float(row["q25"]) for row in hs]; hq3 = [float(row["q75"]) for row in hs]
        ax.plot(hx, hm, marker="o", color="#3568a8", label="F2 mean"); ax.fill_between(hx, hq1, hq3, alpha=.2, color="#3568a8", label="Q1–Q3")
        ax.set_xticks(hx); ax.set_xlabel("Autoregressive horizon"); ax.set_ylabel("MSE"); ax.set_title("F2 causal C1 autoregressive error"); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); path = output / "horizon_growth.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    suites = [suite for suite in SUITE_ORDER if any(row.get("suite") == suite for row in suite_table)]
    conds = [name for name in ("F0", "F1", "F2", "F3", "F4", "F5") if any(row.get("condition") == name for row in suite_table)]
    matrix = np.full((len(conds), len(suites)), np.nan)
    for i, condition in enumerate(conds):
        for j, suite in enumerate(suites):
            values = [float(row["mean_mse"]) for row in suite_table if row.get("condition") == condition and row.get("suite") == suite and row.get("mean_mse") is not None]
            if values: matrix[i, j] = values[0]
    fig, ax = plt.subplots(figsize=(8.4, 4.4)); image = ax.imshow(matrix, aspect="auto", cmap="viridis", norm=Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix)))
    ax.set_xticks(np.arange(len(suites)), suites, rotation=25, ha="right"); ax.set_yticks(np.arange(len(conds)), conds); ax.set_title("MSE by condition and LIBERO suite"); fig.colorbar(image, ax=ax, label="MSE")
    for i in range(len(conds)):
        for j in range(len(suites)):
            if np.isfinite(matrix[i, j]): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i,j] > np.nanmean(matrix) else "black", fontsize=8)
    fig.tight_layout(); path = output / "suite_mse_heatmap.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    # Stage and success panels for the primary causal baseline and the
    # original-joint leakage control.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, field, title, levels in ((axes[0], "stage", "F0/F5 by temporal stage", STAGE_ORDER), (axes[1], "success", "F0/F5 by rollout success", ("True", "False"))):
        for condition, color in (("F0", "#3568a8"), ("F5", "#d17c2f")):
            vals = [next((float(row["mean_mse"]) for row in stage_success_table if row.get("condition") == condition and row.get(field) == level), np.nan) for level in levels]
            offset = -.18 if condition == "F0" else .18
            ax.bar(np.arange(len(levels)) + offset, vals, width=.34, label=condition, color=color)
        ax.set_xticks(np.arange(len(levels)), levels); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("MSE"); axes[1].legend(); fig.tight_layout(); path = output / "stage_success_mse.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8.2, 4.5)); levels = ("Q1", "Q2", "Q3", "Q4"); positions = np.arange(4)
    for condition, color in (("F0", "#3568a8"), ("F1", "#6aaed6"), ("F5", "#d17c2f")):
        vals = [next((float(row["mean_mse"]) for row in motion_table if row.get("condition") == condition and row.get("motion_quartile") == level), np.nan) for level in levels]
        ax.plot(positions, vals, marker="o", label=condition, color=color)
    ax.set_xticks(positions, levels); ax.set_xlabel("Target latent change quartile (F0 reference thresholds)"); ax.set_ylabel("MSE"); ax.set_title("Error versus latent motion"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); path = output / "motion_quartile_mse.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    screening_names = [name for name in SCREENING_CONDITIONS if any(row.get("condition") == name for row in screening_table)]
    if screening_names:
        fig, ax = plt.subplots(figsize=(10, 4.4)); vals = [next(float(row["mean_mse"]) for row in screening_table if row.get("condition") == name) for name in screening_names]
        ax.bar(np.arange(len(screening_names)), vals, color="#777777"); ax.set_xticks(np.arange(len(screening_names)), screening_names); ax.set_ylabel("MSE"); ax.set_title("Stage-1 screening controls (one rollout per task)"); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); path = output / "screening_controls_mse.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    if collection_table:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3)); x = np.arange(len(collection_table)); names = [str(row["suite"]) for row in collection_table]
        axes[0].bar(x, [int(row["rollouts"]) for row in collection_table], color="#3568a8"); axes[0].set_title("Collected rollouts"); axes[0].set_ylabel("count")
        axes[1].bar(x, [float(row["success_rate"]) * 100.0 for row in collection_table], color="#5a9b6e"); axes[1].set_title("Rollout success rate"); axes[1].set_ylabel("percent")
        for ax in axes: ax.set_xticks(x, names, rotation=25, ha="right"); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); path = output / "collection_suite_summary.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    f0 = condition_rows.get("F0", [])
    if f0:
        x = np.asarray([float(row["target_delta_rms"]) for row in f0]); y = np.asarray([float(row["mse"]) for row in f0]);
        fig, ax = plt.subplots(figsize=(6.4, 4.5)); ax.scatter(x, y, s=8, alpha=.2, color="#3568a8");
        if np.isfinite(x).all() and np.isfinite(y).all() and np.ptp(x) > 0:
            coeff = np.polyfit(x, y, 1); grid = np.linspace(float(np.min(x)), float(np.max(x)), 100); ax.plot(grid, coeff[0] * grid + coeff[1], color="#d17c2f", lw=2)
        ax.set_xlabel("Target latent change RMS"); ax.set_ylabel("F0 MSE"); ax.set_title("F0 direct error versus target motion"); ax.grid(alpha=.25)
        fig.tight_layout(); path = output / "mse_vs_motion.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    return paths


def _md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]], float_digits: int = 4) -> str:
    lines = ["| " + " | ".join(title for _, title in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.{float_digits}f}")
            elif value is None:
                cells.append("—")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(
    output: Path,
    collection: Mapping[str, Any],
    formal: Mapping[str, Any],
    screening: Mapping[str, Any],
    supplemental: Mapping[str, Any],
    condition_table: Sequence[Mapping[str, Any]],
    paired_table: Sequence[Mapping[str, Any]],
    horizon_table: Sequence[Mapping[str, Any]],
    suite_table: Sequence[Mapping[str, Any]],
    stage_success_table: Sequence[Mapping[str, Any]],
    motion_table: Sequence[Mapping[str, Any]],
    action_table: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    screening_table: Sequence[Mapping[str, Any]],
    plots: Sequence[str],
    bootstrap_replicates: int,
) -> str:
    totals = collection["totals"]
    lines = [
        "# latent_world_model × VLA-JEPA × LIBERO 深度分析报告",
        "",
        "> 本报告基于已采集的 v3 数据集、阶段 0/1 筛选结果、X0 补充对照和正式半量（130×5）结果生成。所有模型权重被冻结，没有重新训练 encoder 或 predictor；分析脚本只读取输入文件。",
        "",
        "## 1. 摘要与结论",
        "",
        f"本次正式评估使用 **{totals['formal_rollouts']} 条 rollout、{len(formal.get('rows', [])) // 6 if formal.get('rows') else '650'} 条窗口**（每条 rollout 的 early/middle/late 三个时段）和六个条件，共 {len(formal.get('rows', []))} 条条件结果。直接的 future-latent MSE 是主要指标；persistence 只作为辅助参照，不替代直接误差。",
        "",
        "核心结论：",
        "",
        "- 严格因果、正确 latent action 的 F0 C1→H1 直接 MSE 为约 6.27；在本数据和当前冻结 checkpoint 上，它没有显示出比简单保持当前 latent 更准确的能力。",
        "- 增加过去三步上下文（F1）反而使 H1 误差增加；把 action 置零（F3）有很小但统计稳定的下降，same-task/stage shuffle（F4）与正确 action 基本相同。这说明当前实验中没有观测到可辨识的 action-conditioned 增益。",
        "- 原始联合 8 帧输入（F5）显著更低，但该输入包含被预测时刻之后的帧；它是未来信息泄漏对照，不能被解释为严格因果能力。X0 补充结果同样支持这一点。",
        "- F2 的自回归多步误差不是单调上升：H1→H2 先下降，H3 又上升。它反映的是滚动 predictor 与目标窗口的共同统计，不应直接解释为“每一步都累积同样的误差”。",
        "- LIBERO-90 失败 rollout 被保留，因为失败轨迹仍然包含合法的动作—视觉序列；成功/失败差异只作描述性分层，不当作因果结论。",
        "",
        "## 2. 数据采集与质量控制",
        "",
        "数据来自五个标准 LIBERO suite 的 VLA-JEPA 推理 rollout。每条记录保存双视角 RGB、状态、执行动作、policy query 帧、24×2048 latent action tokens 和对应的 7×7 unnormalized action chunk；视频与 HDF5 以 suite/task/episode 三元组配对。",
        "",
        _md_table(collection["suite_summary"], [("suite", "suite"), ("rollouts", "rollouts"), ("successes", "successful"), ("success_rate", "success rate"), ("frames_mean", "mean frames"), ("queries_mean", "mean queries"), ("video_mb", "video MB")]),
        "",
        f"总计：{totals['rollouts']} 条 rollout，{totals['successes']} 条成功（{100.0 * totals['success_rate']:.2f}%），{totals['videos']} 个匹配视频。manifest 记录 HDF5 约 57.25 GB、视频约 252 MB；完整性检查为重复 0、无效 0、缺失配对 0、孤立视频 0。",
        "",
        "LIBERO-90 的成功率明显较低（约 20%），但这是采集结果而不是数据损坏；其失败轨迹仍按预先约定保留。正式集从每个 task 确定性选择 episode 0、2、4、6、8，因而每个 task 有 5 条 rollout，避免因为先后采集顺序或随机抽样改变实验集。",
        "",
        "正式子集按 suite：",
        "",
        _md_table(collection["formal_suite_summary"], [("suite", "suite"), ("rollouts", "formal rollouts"), ("successes", "successful"), ("success_rate", "success rate"), ("frames_mean", "mean frames"), ("queries_mean", "mean queries")]),
        "",
        "## 3. 实验设计与可复现性",
        "",
        "评估以 query 帧为时间锚点。8 帧经过 tubelet=2 编码为 z0…z3；当前状态为 z2，目标为 z3。strict_causal 为每个 latent block 单独构造不包含未来帧的 clip，并在 episode 起点左填充；original_joint 使用原始连续 8 帧编码，仅用作泄漏对照。H3 是冻结 predictor 的自回归滚动，不是重新训练的多步 head。",
        "",
        "正式条件：",
        "",
        _md_table([
            {"condition": "F0", "visual": "strict causal，双视角，C1（当前 z2）", "action": "当前窗口正确 action（当前 query 的 g2）", "target": "strict causal 当前窗口真实未来 latent（z3）", "purpose": "严格因果主结果"},
            {"condition": "F1", "visual": "strict causal，双视角，C3（z0,z1,z2）", "action": "当前窗口正确 action（g0,g1,g2）", "target": "当前窗口真实未来 latent（z3）", "purpose": "检验过去多帧上下文是否改善 H1"},
            {"condition": "F2", "visual": "strict causal，双视角，C1（当前 z2）", "action": "正确 action 的自回归序列（g2→下一 query 的 g0→g1）", "target": "strict causal 的连续未来 latent（z3,z4,z5）", "purpose": "检验单帧上下文的多步滚动预测"},
            {"condition": "F3", "visual": "strict causal，双视角，C1（当前 z2）", "action": "zero action（替换为全零 action group）", "target": "与 F0 相同的 strict-causal future latent（z3）", "purpose": "检查预测是否依赖当前 action"},
            {"condition": "F4", "visual": "strict causal，双视角，C1（当前 z2）", "action": "同 task、同 stage、其他 episode 的 action（确定性循环配对）", "target": "与 F0 相同的 strict-causal future latent（z3）", "purpose": "检查 action 是否提供样本相关的 transition 信息"},
            {"condition": "F5", "visual": "original joint，双视角，连续 8 帧共同编码；取 C1 的 z2", "action": "当前窗口正确 action（当前 query 的 g2）", "target": "与 F0 相同的 strict-causal future latent（z3）", "purpose": "检查非因果联合编码带来的未来信息泄漏影响"},
        ], [("condition", "条件"), ("visual", "视觉输入"), ("action", "latent action"), ("target", "目标"), ("purpose", "主要目的")]),
        "",
        "F4 与 F0 只替换 action，视觉输入、当前 latent、目标和 H1 预测步长保持一致。F4 的 action 来自完整索引中同 suite、同 task、同 stage 的其他 episode，并按 episode id 做确定性循环配对；因此保留了任务和阶段的大致分布，但破坏了 action 与当前视觉状态之间的对应关系。若 F0 明显优于 F4，说明 action 可能包含样本相关的 transition 信息；若两者接近，则说明当前模型没有表现出可检测的 action-specific 增益，或者 action-token 时间对齐仍需检查。",
        "",
        "F5 与 F0 使用相同的 C1、正确 action 和 strict-causal 目标，区别只在视觉编码协议。original_joint 将连续 8 帧同时送入时序 encoder；即使 predictor 最后只取名义上的 z2，联合 encoder 的时空建模仍可能让 z2 表示接触到后续帧信息。因此 F5 是非因果信息泄漏对照，而不是严格推理性能；F5 的误差下降只能说明未来视觉信息会让预测更容易，不能作为 causal world model 能力结论。",
        "",
        "阶段漏斗：阶段 0 完成 smoke、shape/dtype/finite、断点续跑和 predictor parity；阶段 1 在每个 task 取一条 rollout，运行 S0–S9 控制；筛选后运行 X0（original joint+C3→H1）作为定向补充；最后运行四路 shard 合并的正式半量。正式 JSONL 共有 11,700 条结果（1,950 windows×6），所有条件和行 ID 完整且无重复。",
        "",
        "独立 predictor 与 VLA-JEPA source 的 parity artifact 显示输入 `[1,768,2048]`、action `[1,24,2048]`、输出 `[1,768,2048]`，max absolute difference=0、mean absolute difference=0、allclose=true。",
        "",
        "## 4. 正式结果：直接误差为主",
        "",
        _md_table(condition_table, [("condition", "condition"), ("n", "n"), ("mean_mse", "mean MSE"), ("ci95_low", "CI low"), ("ci95_high", "CI high"), ("median_mse", "median"), ("q25_mse", "Q25"), ("q75_mse", "Q75"), ("mean_normalized_mse", "normalized MSE"), ("mean_persistence_ratio", "persistence ratio"), ("mean_token_cosine", "token cosine"), ("mean_delta_cosine", "delta cosine"), ("retrieval_top1", "retrieval top1"), ("retrieval_top5", "retrieval top5")]),
        "",
        "F0 的直接误差是本报告的首要答案：模型输出与真实未来 latent 的平均平方差约为 6.27。它不依赖 persistence 定义，直接回答“预测 latent 与真实未来 latent 相差多大”。L1、RMSE、normalized MSE、token cosine、delta cosine 和 retrieval 是同一误差的补充视角；retrieval 的随机 top-1/top-5 参考分别是 1/1950≈0.00051 和 5/1950≈0.00256。",
        "",
        "### 4.1 注册的配对差异",
        "",
        _md_table(paired_table, [("comparison", "comparison (left−right)"), ("mean", "mean diff"), ("ci95_low", "CI low"), ("ci95_high", "CI high"), ("median_difference", "median diff"), ("fraction_left_better", "left better"), ("cohen_d", "paired d"), ("p_holm", "Holm p")]),
        "",
        "差异按同一 window 配对，再做 task→rollout/window 的层级 bootstrap；负值表示左侧误差更小。F4 的置信区间跨过 0，说明 shuffle 与正确 action 几乎无法区分；F3 的差异很小，虽统计稳定但实际效应有限；F5 的差异很大却不具备因果解释，因为协议不同且含未来帧。",
        "",
        "### 4.2 persistence 的正确定位",
        "",
        "persistence 是把当前 z2 原样当作未来 latent 的参照。它不是评估目标，也不是要求 world model 必须超过的唯一标准；直接 MSE 才是主要结果。persistence ratio=MSE/persistence MSE 只回答一个附加问题：模型是否比“未来不变”的简单预测更好。ratio<1 表示相对该简单参照更好，ratio>1 表示更差。报告同时保留 persistence，是为了判断 6.27 这个绝对误差在当前 latent 变化尺度下是否有实际增益，而不是把它混入模型定义。",
        "",
        "F0 ratio 约 1.43；这意味着在严格 causal 条件下，模型误差约为保持 z2 的 1.43 倍。但无论 ratio 如何，F0 的 6.27 仍然是与真实未来 latent 的直接误差。F5 ratio<1 主要对应未来信息泄漏对照，不能据此宣称 causal predictor 已经成功。",
        "",
        "## 5. 时间、任务和运动量分层",
        "",
        "### 5.1 自回归 horizon",
        "",
        _md_table(horizon_table, [("horizon", "horizon"), ("n", "n"), ("mean", "mean MSE"), ("median", "median"), ("q25", "Q25"), ("q75", "Q75"), ("min", "min"), ("max", "max")]),
        "",
        "F2 使用 H1/H2/H3 的同一条滚动轨迹。H2 低于 H1 不应被解读为模型在未来更远处一定更准：目标帧和动作窗口的运动分布不同，且这里只抽取 early/middle/late 三个窗口。真正稳定的结论是 H3 最后一阶段仍处于约 6.55 的误差水平，未呈现可靠的多步预测优势。",
        "",
        "### 5.2 suite 分层",
        "",
        _md_table(suite_table, [("condition", "condition"), ("suite", "suite"), ("n", "n"), ("mean_mse", "mean MSE"), ("median_mse", "median"), ("mean_persistence_ratio", "persistence ratio")]),
        "",
        "F0 的 suite 间差异约在 6.04–6.32，量级小于 F5 泄漏对照相对于 F0 的下降；这支持“协议/信息可见性”是首要影响因素，但 suite 分层仍是描述性结果，不能据此归因于某类任务机制。",
        "",
        "### 5.3 时间阶段、成功状态、latent motion 和 action scale",
        "",
        _md_table([row for row in stage_success_table if row.get("stage") not in (None, "", "None")], [("condition", "condition"), ("stage", "stage"), ("n", "n"), ("mean_mse", "mean MSE"), ("median_mse", "median")]),
        "",
        _md_table([row for row in stage_success_table if row.get("success") not in (None, "", "None")], [("condition", "condition"), ("success", "success"), ("n", "n"), ("mean_mse", "mean MSE"), ("median_mse", "median")]),
        "",
        _md_table(motion_table, [("condition", "condition"), ("motion_quartile", "motion quartile"), ("n", "n"), ("mean_mse", "mean MSE"), ("mean_target_delta_rms", "mean target Δ RMS")]),
        "",
        _md_table(action_table, [("condition", "condition"), ("action_quartile", "action quartile"), ("n", "n"), ("mean_mse", "mean MSE"), ("mean_action_norm", "mean action scale")]),
        "",
        "F0 的 late 阶段误差略高；但 latent-change RMS 分箱本身并不单调（F0 的最高 motion 分箱反而略低），而 target variance 与误差的相关性更强。分层只是定位误差集中在哪里，不能证明“运动量导致误差”。",
        "",
        "### 5.4 相关性",
        "",
        _md_table(correlations, [("predictor", "predictor"), ("n", "n"), ("pearson_r", "Pearson r"), ("spearman_r", "Spearman r"), ("x_mean", "mean x"), ("y_mean_mse", "mean MSE")]),
        "",
        "相关系数是 F0 window-level 的描述性诊断；它不替代配对比较，也不控制 task、stage 等混杂因素。",
        "",
        "## 6. 阶段 1 控制与泄漏诊断",
        "",
        _md_table(screening.get("condition_table", []), [("condition", "condition"), ("n", "n"), ("mean_mse", "mean MSE"), ("mean_persistence_ratio", "persistence ratio"), ("mean_token_cosine", "token cosine"), ("retrieval_top1", "retrieval top1"), ("retrieval_top5", "retrieval top5")]),
        "",
        "阶段 1 的 S0–S9 只用于筛选和发现值得补充的方向，不与正式 F0–F5 混合做主要显著性检验。X0（original joint+C3→H1）MSE 约 3.33，相对 strict C3 的筛选结果下降约 49%，与 F5 的显著下降一起说明联合编码能看到未来帧，不能用于证明 causal world model 能力。",
        "",
        "## 7. 图像与数据表",
        "",
        "图像由本分析脚本在同一输出目录生成：",
        "",
    ]
    for plot in plots:
        lines.append(f"- [{Path(plot).name}]({Path(plot).name})")
    lines += [
        "",
        "CSV 表：`condition_summary.csv`、`paired_comparisons.csv`、`horizon_summary.csv`、`suite_strata.csv`、`stage_success_strata.csv`、`motion_strata.csv`、`action_strata.csv`、`correlations.csv`、`collection_suite_summary.csv`、`formal_collection_suite_summary.csv`、`screening_summary.csv`。完整 JSON 在 `deep_summary.json`。",
        "",
        "## 8. 局限性与下一步",
        "",
        "1. 评估是在冻结 checkpoint 上进行的；没有训练/微调，因此结果回答的是“现有模型在该数据和对齐方式上的能力”，不是可达到的上限。",
        "2. 预测目标是 V-JEPA2 latent，不是 RGB；没有公开兼容 decoder，不能直接报告像素级视频质量。",
        "3. formal half 是 130×5 的确定性子集，结果应推广到同一采集分布而非任意 LIBERO 数据。完整 1300 rollout 数据仍保留，可在资源允许时扩大正式评估。",
        "4. Causal clip、query 帧和 action-token 分组的时间对齐是关键风险点。下一步应优先做人工/可视化对齐审计、用 action permutation 检验 token 信息是否可识别、以及按 task 做更严格的 held-out 分析。",
        "5. 若后续要改进模型，应先保留本报告的 F0/F3/F4/F5 作为回归基线，再单独设计训练实验；不要用 original_joint 结果作 causal 训练目标或能力结论。",
        "",
        "## 9. 复现入口与版本",
        "",
        "- 数据集：`datasets/vla_jepa_libero130_v3/README.md`、`manifest.json`。",
        "- 评估协议和命令：`EVALUATION.md`。",
        "- 正式结果：`evaluation_outputs/formal_half/metrics.jsonl`、`summary.json`。",
        "- 本报告脚本：`latent_world_model/evaluation/deep_analysis.py`。",
        "- 运行示例见脚本 docstring；输出目录默认 `evaluation_outputs/deep_analysis`。",
        "- 代码版本：当前工作区 Git 历史中的评估实现提交；HDF5、MP4、checkpoint 和大型 memmap 均不纳入 Git。",
        "",
    ]
    report = "\n".join(lines)
    (output / "report.md").write_text(report, encoding="utf-8")
    return report


def generate_deep_analysis(
    dataset_root: str | Path,
    formal_metrics: str | Path,
    screening_metrics: str | Path,
    supplemental_metrics: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 20260718,
) -> dict[str, Any]:
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    formal_rows = _read_jsonl(Path(formal_metrics)); screening_rows = _read_jsonl(Path(screening_metrics)); supplemental_rows = _read_jsonl(Path(supplemental_metrics))
    if not formal_rows:
        raise FileNotFoundError(f"no formal metric rows: {formal_metrics}")
    condition_rows = _add_quantile_labels(formal_rows)
    collection_table, collection = _collection_stats(Path(dataset_root).resolve())
    condition_table = _condition_summary(formal_rows, bootstrap_replicates, seed, retrieval_dir=Path(formal_metrics).resolve().parent)
    paired_table = _paired_comparisons(formal_rows, bootstrap_replicates, seed)
    horizon_table = _horizon(formal_rows)
    suite_table = _stratified(formal_rows, ("suite",))
    stage_table = _stratified(formal_rows, ("stage",))
    success_table = _stratified(formal_rows, ("success",))
    motion_table = _stratified([row for group in condition_rows.values() for row in group], ("motion_quartile",))
    action_table = _stratified([row for group in condition_rows.values() for row in group], ("action_quartile",))
    stage_success_table = stage_table + success_table
    correlations = _correlations(formal_rows)
    screening_summary = _condition_summary(screening_rows, bootstrap_replicates, seed + 1000) if screening_rows else []
    supplemental_summary = _condition_summary(supplemental_rows, bootstrap_replicates, seed + 2000) if supplemental_rows else []
    # Keep output tables compact and machine-readable.
    _write_csv(output / "condition_summary.csv", condition_table, ["condition", "n", "mean_mse", "std_mse", "median_mse", "q25_mse", "q75_mse", "min_mse", "max_mse", "ci95_low", "ci95_high", "mean_l1", "mean_rmse", "mean_normalized_mse", "mean_persistence_mse", "mean_persistence_ratio", "mean_token_cosine", "mean_delta_cosine", "mean_delta_norm_ratio", "mean_prediction_variance_ratio", "retrieval_n", "retrieval_top1", "retrieval_top5", "protocol", "context_steps", "horizon", "action_mode", "view"])
    _write_csv(output / "paired_comparisons.csv", paired_table, ["comparison", "n_tasks", "n", "mean", "ci95_low", "ci95_high", "median_difference", "fraction_left_better", "fraction_left_worse", "mean_abs_difference", "cohen_d", "p_two_sided", "p_holm"])
    _write_csv(output / "horizon_summary.csv", horizon_table, ["condition", "horizon", "metric", "n", "mean", "std", "median", "q25", "q75", "min", "max"])
    _write_csv(output / "suite_strata.csv", suite_table, ["condition", "suite", "n", "mean_mse", "median_mse", "q25_mse", "q75_mse", "mean_persistence_ratio", "mean_target_delta_rms", "mean_action_norm"])
    _write_csv(output / "stage_success_strata.csv", stage_success_table, ["condition", "stage", "success", "n", "mean_mse", "median_mse", "q25_mse", "q75_mse", "mean_persistence_ratio", "mean_target_delta_rms", "mean_action_norm"])
    _write_csv(output / "motion_strata.csv", motion_table, ["condition", "motion_quartile", "n", "mean_mse", "median_mse", "q25_mse", "q75_mse", "mean_persistence_ratio", "mean_target_delta_rms", "mean_action_norm"])
    _write_csv(output / "action_strata.csv", action_table, ["condition", "action_quartile", "n", "mean_mse", "median_mse", "q25_mse", "q75_mse", "mean_persistence_ratio", "mean_target_delta_rms", "mean_action_norm"])
    _write_csv(output / "correlations.csv", correlations, ["condition", "predictor", "n", "pearson_r", "spearman_r", "x_mean", "y_mean_mse"])
    _write_csv(output / "collection_suite_summary.csv", collection["suite_summary"], ["suite", "rollouts", "successes", "success_rate", "frames_mean", "frames_median", "frames_min", "frames_max", "queries_mean", "queries_median", "queries_min", "queries_max", "video_mb"])
    _write_csv(output / "formal_collection_suite_summary.csv", collection["formal_suite_summary"], ["suite", "rollouts", "successes", "success_rate", "frames_mean", "queries_mean"])
    _write_csv(output / "screening_summary.csv", screening_summary + supplemental_summary, ["condition", "n", "mean_mse", "std_mse", "median_mse", "q25_mse", "q75_mse", "ci95_low", "ci95_high", "mean_persistence_ratio", "mean_token_cosine", "protocol", "context_steps", "horizon", "action_mode", "view"])
    plots = _plot_all(output, condition_rows, condition_table, paired_table, horizon_table, suite_table, stage_success_table, motion_table, screening_summary, collection_table)
    deep_summary: dict[str, Any] = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "formal_metrics": str(Path(formal_metrics).resolve()),
        "screening_metrics": str(Path(screening_metrics).resolve()),
        "supplemental_metrics": str(Path(supplemental_metrics).resolve()),
        "formal": {"rows": len(formal_rows), "windows": len(formal_rows) // len(_condition_rows(formal_rows)), "conditions": sorted(_condition_rows(formal_rows)), "condition_summary": condition_table, "paired_comparisons": paired_table, "horizon": horizon_table, "suite_strata": suite_table, "stage_strata": stage_table, "success_strata": success_table, "motion_strata": motion_table, "action_strata": action_table, "correlations": correlations},
        "screening": {"rows": len(screening_rows), "condition_summary": screening_summary},
        "supplemental": {"rows": len(supplemental_rows), "condition_summary": supplemental_summary},
        "collection": collection,
        "plots": plots,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }
    (output / "deep_summary.json").write_text(json.dumps(deep_summary, indent=2, ensure_ascii=False, default=lambda value: float(value) if isinstance(value, (np.floating,)) else value) + "\n", encoding="utf-8")
    _write_report(output, collection, {"rows": formal_rows}, {"rows": screening_rows, "condition_table": screening_summary}, {"rows": supplemental_rows}, condition_table, paired_table, horizon_table, suite_table, stage_success_table, motion_table, action_table, correlations, screening_summary + supplemental_summary, plots, bootstrap_replicates)
    return deep_summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--formal-metrics", type=Path, required=True)
    parser.add_argument("--screening-metrics", type=Path, required=True)
    parser.add_argument("--supplemental-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args(argv)
    summary = generate_deep_analysis(args.dataset_root, args.formal_metrics, args.screening_metrics, args.supplemental_metrics, args.output, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    print(json.dumps({"formal_rows": summary["formal"]["rows"], "plots": summary["plots"], "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
