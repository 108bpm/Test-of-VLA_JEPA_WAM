"""Merge independent runner shards without loading latent tensors in RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def merge_shards(shard_dirs: Sequence[str | Path], output_dir: str | Path) -> dict:
    shards = [Path(path) for path in shard_dirs]
    if not shards:
        raise ValueError("at least one shard is required")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = [json.loads((path / "config.json").read_text(encoding="utf-8")) for path in shards]
    conditions = list(configs[0]["conditions"])
    if any(list(config["conditions"]) != conditions for config in configs[1:]):
        raise ValueError("shards do not use the same condition list")

    rows = []
    seen = set()
    for path in shards:
        with (path / "metrics.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["row_id"]), str(row["condition"]))
                if key in seen:
                    raise ValueError(f"duplicate shard result: {key}")
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (str(row["row_id"]), str(row["condition"])))
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    row_counts = []
    for path in shards:
        count = int(json.loads((path / "config.json").read_text(encoding="utf-8")).get("row_count", 0))
        row_counts.append(count)
    total_rows = sum(row_counts)
    for condition in conditions:
        pred_out = np.lib.format.open_memmap(out / f"embeddings_{condition}_pred.npy", mode="w+", dtype=np.float16, shape=(total_rows, 2048))
        target_out = np.lib.format.open_memmap(out / f"embeddings_{condition}_target.npy", mode="w+", dtype=np.float16, shape=(total_rows, 2048))
        valid_out = np.lib.format.open_memmap(out / f"embeddings_{condition}_valid.npy", mode="w+", dtype=np.bool_, shape=(total_rows,))
        offset = 0
        for path, count in zip(shards, row_counts):
            pred = np.load(path / f"embeddings_{condition}_pred.npy", mmap_mode="r")
            target = np.load(path / f"embeddings_{condition}_target.npy", mmap_mode="r")
            valid = np.load(path / f"embeddings_{condition}_valid.npy", mmap_mode="r")
            if pred.shape != (count, 2048) or target.shape != (count, 2048) or valid.shape != (count,):
                raise ValueError(f"embedding shape mismatch in {path} for {condition}")
            pred_out[offset : offset + count] = pred
            target_out[offset : offset + count] = target
            valid_out[offset : offset + count] = valid
            offset += count
        pred_out.flush(); target_out.flush(); valid_out.flush()

    summary = {
        "shards": [str(path.resolve()) for path in shards],
        "output_dir": str(out.resolve()),
        "conditions": conditions,
        "rows": total_rows,
        "metric_rows": len(rows),
        "expected_metric_rows": total_rows * len(conditions),
    }
    (out / "config.json").write_text(json.dumps({**configs[0], "merged_shards": summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "merge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(merge_shards(args.shards, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
