"""Validate and inventory the finalized evaluation archive.

Large rollout HDF5 files, checkpoints, NumPy caches, and generated result
tables stay outside Git.  This command records their paths, sizes, SHA-256
digests, row counts, and condition coverage so the tracked reports can point
to an auditable local archive without duplicating tens of gigabytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FIRST_CONDITIONS = {f"F{index}": 1950 for index in range(6)}
SECOND_CONDITIONS = {f"J{index}": 1950 for index in range(3)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _scan_jsonl(path: Path) -> dict[str, Any]:
    conditions: Counter[str] = Counter()
    row_ids: set[str] = set()
    keys: set[tuple[str, str]] = set()
    duplicate_keys = 0
    nonfinite_metrics = 0
    lines = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            lines += 1
            row = json.loads(line)
            row_id = str(row["row_id"])
            condition = str(row["condition"])
            key = (row_id, condition)
            if key in keys:
                duplicate_keys += 1
            keys.add(key)
            row_ids.add(row_id)
            conditions[condition] += 1
            for metric in ("mse", "l1", "mean_token_cosine"):
                if metric not in row or not math.isfinite(float(row[metric])):
                    nonfinite_metrics += 1
    return {
        "lines": lines,
        "unique_keys": len(keys),
        "unique_windows": len(row_ids),
        "conditions": dict(sorted(conditions.items())),
        "duplicate_keys": duplicate_keys,
        "nonfinite_required_metrics": nonfinite_metrics,
    }


def _add_files(
    output: dict[str, dict[str, Any]],
    root: Path,
    paths: Iterable[Path],
) -> None:
    for path in paths:
        output[str(path.relative_to(root))] = _artifact(path, root)


def _directory_inventory(path: Path, root: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "file_count": len(files),
        "logical_size_bytes": sum(item.stat().st_size for item in files),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    dataset_manifest_path = root / "datasets/vla_jepa_libero130_v3/manifest.json"
    index_path = root / "evaluation_outputs/stage0/index.jsonl"
    first_metrics_path = root / "evaluation_outputs/formal_half/metrics.jsonl"
    second_metrics_path = root / "evaluation_outputs/joint_c3_full/metrics.jsonl"
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    first = _scan_jsonl(first_metrics_path)
    second = _scan_jsonl(second_metrics_path)
    index_rows = sum(1 for line in index_path.open("r", encoding="utf-8") if line.strip())

    errors: list[str] = []
    if int(dataset.get("total_rollouts", -1)) != 1300:
        errors.append("dataset manifest does not contain 1300 rollouts")
    if index_rows != 3900:
        errors.append(f"evaluation index has {index_rows} rows, expected 3900")
    for name, scan, expected in (
        ("first", first, FIRST_CONDITIONS),
        ("second", second, SECOND_CONDITIONS),
    ):
        if scan["conditions"] != expected:
            errors.append(f"{name} condition counts {scan['conditions']} != {expected}")
        if scan["unique_windows"] != 1950:
            errors.append(f"{name} has {scan['unique_windows']} unique windows, expected 1950")
        if scan["duplicate_keys"]:
            errors.append(f"{name} has {scan['duplicate_keys']} duplicate result keys")
        if scan["nonfinite_required_metrics"]:
            errors.append(f"{name} has {scan['nonfinite_required_metrics']} non-finite/missing metrics")

    artifacts: dict[str, dict[str, Any]] = {}
    _add_files(
        artifacts,
        root,
        [
            dataset_manifest_path,
            index_path,
            root / "evaluation_outputs/formal_half/config.json",
            first_metrics_path,
            root / "evaluation_outputs/formal_half/summary.json",
            root / "evaluation_outputs/formal_half/report.md",
            root / "evaluation_outputs/joint_c3_full/config.json",
            second_metrics_path,
            root / "evaluation_outputs/joint_c3_full/summary.json",
            root / "evaluation_outputs/joint_c3_full/report.md",
            root / "latent_world_model/evaluation/runner.py",
            root / "latent_world_model/evaluation/report.py",
            root / "latent_world_model/evaluation/archive_manifest.py",
            root / "COMPREHENSIVE_REPORT.md",
            root / "EXPERIMENT_AUDIT_REPORT.md",
            root / "SECOND_EXPERIMENT_REPORT.md",
            root / "FINAL_REPORT.md",
        ],
    )
    audit_dir = root / "evaluation_outputs/audit"
    _add_files(artifacts, root, sorted(audit_dir.glob("*.json")))
    # Hash every authoritative generated artifact, including plots and compact
    # embedding summaries.  Historical shards and smoke/screening directories
    # remain inventoried below without bloating the tracked manifest with a
    # digest entry for every resumability cache.
    for directory in (
        root / "evaluation_outputs/formal_half",
        root / "evaluation_outputs/deep_analysis",
        root / "evaluation_outputs/joint_c3_full",
    ):
        _add_files(artifacts, root, sorted(path for path in directory.rglob("*") if path.is_file()))

    local_output_dirs = [
        path
        for path in sorted((root / "evaluation_outputs").iterdir())
        if path.is_dir()
    ]

    return {
        "format_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "status": "complete" if not errors else "invalid",
        "validation_errors": errors,
        "dataset": {
            "name": dataset.get("name"),
            "rollouts": dataset.get("total_rollouts"),
            "videos": dataset.get("total_videos"),
            "index_windows": index_rows,
        },
        "first_experiment": first,
        "second_experiment": second,
        "local_result_directories": {
            str(path.relative_to(root)): _directory_inventory(path, root)
            for path in local_output_dirs
        },
        "artifacts": dict(sorted(artifacts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="exit nonzero when archive invariants fail")
    args = parser.parse_args()
    manifest = build_manifest(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "dataset", "first_experiment", "second_experiment")}, indent=2, ensure_ascii=False))
    if args.strict and manifest["validation_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
