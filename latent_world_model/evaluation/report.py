"""Hierarchical bootstrap summaries and plots for runner JSONL outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _read_metrics(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(rows: Sequence[dict], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(row[key])]
    return float(np.mean(values)) if values else None


def _bootstrap_hierarchical(
    rows: Sequence[dict],
    values: Mapping[str, float],
    *,
    seed: int,
    replicates: int,
) -> Dict[str, float]:
    """Task -> rollout hierarchical bootstrap for a scalar per-rollout value."""
    by_task: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in values and np.isfinite(values[row_id]):
            by_task[f"{row['suite']}/task{int(row['task_id']):03d}"].append(float(values[row_id]))
    if not by_task:
        return {"n_tasks": 0, "n_rollouts": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    tasks = sorted(by_task)
    estimates = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        sampled_values = []
        for task in sampled_tasks:
            rollouts = by_task[task]
            sampled_values.extend(rng.choice(rollouts, size=len(rollouts), replace=True).tolist())
        estimates[i] = np.mean(sampled_values)
    observed = float(np.mean([value for values_ in by_task.values() for value in values_]))
    p = 2.0 * min(float(np.mean(estimates <= 0.0)), float(np.mean(estimates >= 0.0))) if observed != 0 else 1.0
    # The sign probability above is appropriate for paired differences centered
    # at zero; for ordinary means retain a descriptive NaN p-value.
    return {
        "n_tasks": len(tasks),
        "n_rollouts": sum(len(v) for v in by_task.values()),
        "mean": observed,
        "ci95_low": float(np.percentile(estimates, 2.5)),
        "ci95_high": float(np.percentile(estimates, 97.5)),
        "p_two_sided": p,
    }


def _paired_values(rows_a: Sequence[dict], rows_b: Sequence[dict], key: str) -> Tuple[List[dict], Dict[str, float]]:
    a = {str(row["row_id"]): float(row[key]) for row in rows_a if key in row and np.isfinite(row[key])}
    b = {str(row["row_id"]): float(row[key]) for row in rows_b if key in row and np.isfinite(row[key])}
    common = sorted(set(a) & set(b))
    synthetic = [{"row_id": row_id, "suite": row_id.split("/", 1)[0], "task_id": int(row_id.split("/task", 1)[1].split("/", 1)[0])} for row_id in common]
    return synthetic, {row_id: a[row_id] - b[row_id] for row_id in common}


def _holm(p_values: Mapping[str, float]) -> Dict[str, float]:
    finite = sorted(((name, float(p)) for name, p in p_values.items() if np.isfinite(p)), key=lambda x: x[1])
    adjusted: Dict[str, float] = {}
    previous = 0.0
    total = len(finite)
    for rank, (name, p) in enumerate(finite):
        value = min(1.0, max(previous, (total - rank) * p))
        adjusted[name] = value
        previous = value
    return adjusted


def _retrieval(pred_path: Path, target_path: Path, valid_path: Path) -> Dict[str, float]:
    if not pred_path.exists() or not target_path.exists() or not valid_path.exists():
        return {"n": 0, "top1": float("nan"), "top5": float("nan")}
    valid = np.asarray(np.load(valid_path, mmap_mode="r"), dtype=bool)
    pred = np.asarray(np.load(pred_path, mmap_mode="r"), dtype=np.float32)[valid]
    target = np.asarray(np.load(target_path, mmap_mode="r"), dtype=np.float32)[valid]
    n = pred.shape[0]
    if n == 0:
        return {"n": 0, "top1": float("nan"), "top5": float("nan")}
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    top1 = 0
    top5 = 0
    target_t = target.T
    for start in range(0, n, 256):
        similarity = pred[start : start + 256] @ target_t
        order = np.argpartition(-similarity, kth=min(4, n - 1), axis=1)[:, : min(5, n)]
        labels = np.arange(start, min(start + 256, n))
        top1 += int(np.sum(order[:, 0] == labels))
        top5 += int(np.sum(np.any(order == labels[:, None], axis=1)))
    return {"n": n, "top1": top1 / n, "top5": top5 / n}


def _plot(summary: Mapping[str, object], output_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    aggregates = summary["conditions"]
    names = list(aggregates)
    means = [aggregates[name].get("mse") for name in names]
    low = [aggregates[name].get("mse_ci95_low") for name in names]
    high = [aggregates[name].get("mse_ci95_high") for name in names]
    if not names or any(value is None for value in means):
        return []
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 4.5))
    ax.bar(x, means, color="#4776a8")
    if all(value is not None for value in low + high):
        ax.errorbar(x, means, yerr=[np.asarray(means) - np.asarray(low), np.asarray(high) - np.asarray(means)], fmt="none", color="black", capsize=3)
    ax.set_xticks(x, names)
    ax.set_ylabel("MSE (latent space)")
    ax.set_title("Latent prediction error by condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "mse_by_condition.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    horizon_names = [name for name in names if aggregates[name].get("horizon") == 3]
    if horizon_names:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in horizon_names:
            curve = aggregates[name].get("horizon_mse", {})
            ax.plot([1, 2, 3], [curve.get(str(i), np.nan) for i in [1, 2, 3]], marker="o", label=name)
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel("MSE")
        ax.set_title("Autoregressive error growth")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path2 = output_dir / "horizon_error.png"
        fig.savefig(path2, dpi=160)
        plt.close(fig)
        return [str(path), str(path2)]
    return [str(path)]


def generate_report(results_dir: str | Path, *, bootstrap_replicates: int = 1000, seed: int = 20260718) -> dict:
    output_dir = Path(results_dir)
    rows = _read_metrics(output_dir / "metrics.jsonl")
    by_condition: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    conditions: Dict[str, dict] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        metrics = {"n": len(condition_rows), "horizon": int(condition_rows[0].get("horizon", 1))}
        for key in ("mse", "l1", "rmse", "persistence_ratio", "mean_token_cosine", "delta_cosine", "delta_norm_ratio", "prediction_variance_ratio", "normalized_mse"):
            value = _mean(condition_rows, key)
            if value is not None:
                metrics[key] = value
        horizon_mse = {}
        for h in (1, 2, 3):
            value = _mean(condition_rows, f"h{h}_mse")
            if value is not None:
                horizon_mse[str(h)] = value
        if horizon_mse:
            metrics["horizon_mse"] = horizon_mse
        # Hierarchical bootstrap CI for the ordinary condition mean.
        if "mse" in metrics:
            values = {str(row["row_id"]): float(row["mse"]) for row in condition_rows}
            stable_offset = sum((i + 1) * ord(char) for i, char in enumerate(condition)) % 10000
            ci = _bootstrap_hierarchical(condition_rows, values, seed=seed + stable_offset, replicates=bootstrap_replicates)
            metrics["mse_ci95_low"] = ci["ci95_low"]
            metrics["mse_ci95_high"] = ci["ci95_high"]
            metrics["bootstrap"] = ci
        retrieval = _retrieval(output_dir / f"embeddings_{condition}_pred.npy", output_dir / f"embeddings_{condition}_target.npy", output_dir / f"embeddings_{condition}_valid.npy")
        metrics["retrieval"] = retrieval
        conditions[condition] = metrics

    comparisons: Dict[str, dict] = {}
    p_values: Dict[str, float] = {}
    # The formal paired comparisons are pre-registered.  Missing F* rows are
    # reported rather than silently substituting a different condition.
    if "F0" in by_condition:
        values = {str(row["row_id"]): float(row["mse"]) - float(row["persistence_mse"]) for row in by_condition["F0"]}
        paired_rows = by_condition["F0"]
        stat = _bootstrap_hierarchical(paired_rows, values, seed=seed + 1, replicates=bootstrap_replicates)
        comparisons["F0-persistence"] = stat
        p_values["F0-persistence"] = stat["p_two_sided"]
    for left, right in (("F1", "F0"), ("F3", "F0"), ("F4", "F0"), ("F5", "F0")):
        if left in by_condition and right in by_condition:
            paired_rows, values = _paired_values(by_condition[left], by_condition[right], "mse")
            stat = _bootstrap_hierarchical(paired_rows, values, seed=seed + len(comparisons) + 2, replicates=bootstrap_replicates)
            comparisons[f"{left}-{right}"] = stat
            p_values[f"{left}-{right}"] = stat["p_two_sided"]
    comparisons_holm = _holm(p_values)
    for name, p in comparisons_holm.items():
        comparisons[name]["p_holm"] = p

    strata: Dict[str, dict] = {}
    for field in ("suite", "success", "stage"):
        field_out = {}
        for condition, condition_rows in by_condition.items():
            groups = defaultdict(list)
            for row in condition_rows:
                groups[str(row.get(field))].append(row)
            field_out[condition] = {group: {"n": len(group_rows), "mse": _mean(group_rows, "mse"), "persistence_ratio": _mean(group_rows, "persistence_ratio")} for group, group_rows in sorted(groups.items())}
        strata[field] = field_out

    summary = {
        "results_dir": str(output_dir.resolve()),
        "rows": len(rows),
        "conditions": conditions,
        "comparisons": comparisons,
        "strata": strata,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }
    summary["plots"] = _plot(summary, output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=lambda value: float(value)) + "\n", encoding="utf-8")

    lines = [
        "# Latent World Model LIBERO 评估报告",
        "",
        f"样本行数：{len(rows)}；bootstrap：task→rollout，{bootstrap_replicates} 次；所有数值均在 V-JEPA2 latent 空间计算。",
        "",
        "## 条件汇总",
        "",
        "| 条件 | n | horizon | MSE | 95% CI | persistence ratio | token cosine | retrieval top1/top5 |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, metrics in conditions.items():
        ci = f"[{metrics.get('mse_ci95_low', float('nan')):.4f}, {metrics.get('mse_ci95_high', float('nan')):.4f}]"
        retrieval = metrics.get("retrieval", {})
        lines.append(f"| {name} | {metrics.get('n', 0)} | {metrics.get('horizon', '')} | {metrics.get('mse', float('nan')):.4f} | {ci} | {metrics.get('persistence_ratio', float('nan')):.4f} | {metrics.get('mean_token_cosine', float('nan')):.4f} | {retrieval.get('top1', float('nan')):.4f}/{retrieval.get('top5', float('nan')):.4f} |")
    lines += ["", "## 预注册配对比较", "", "| 比较（左−右，负值更好） | mean | 95% CI | Holm p |", "|---|---:|---|---:|"]
    for name, stat in comparisons.items():
        lines.append(f"| {name} | {stat.get('mean', float('nan')):.4f} | [{stat.get('ci95_low', float('nan')):.4f}, {stat.get('ci95_high', float('nan')):.4f}] | {stat.get('p_holm', float('nan')):.4f} |")
    lines += ["", "## 解读规则", "", "- `persistence_ratio < 1` 表示优于保持当前 latent 的基线；`history_gain` 和 `action_gain` 由配对 MSE 差异计算。", "- H3 条件同时报告 H1/H2/H3，不能把 H3 的最后一步误读为 direct multi-horizon head；这里是冻结 predictor 的自回归滚动。", "- `original_joint` 使用联合 8 帧编码作为输入，但统一用 strict-causal target 比较，用于暴露未来帧泄漏。", "- 分层表和图表位于同一结果目录；若某条件缺失，报告保留为空而不替换实验定义。", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args(argv)
    summary = generate_report(args.results_dir, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    print(json.dumps({"rows": summary["rows"], "conditions": sorted(summary["conditions"]), "comparisons": sorted(summary["comparisons"]), "plots": summary["plots"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
