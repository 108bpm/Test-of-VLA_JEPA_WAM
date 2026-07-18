"""Recompute latent actions from recorded observations using a live VLA server.

The request payload intentionally mirrors ``M1Inference.step`` from the LIBERO
collector.  Equality is checked after converting the fresh float32 response to
the float16 storage dtype, which isolates alignment/reproducibility from the
documented quantization used to reduce dataset size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


def recompute(
    records: list[Path],
    *,
    source_root: Path,
    host: str,
    port: int,
    query_row: int,
    repeats: int,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    sys.path.insert(0, str(source_root.resolve()))
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(host, port, request_timeout=180.0)
    results: list[dict[str, Any]] = []
    try:
        for record in records:
            with h5py.File(record, "r") as handle:
                count = int(handle["query_frame_index"].shape[0])
                selected_row = query_row if query_row >= 0 else count // 2
                if not 0 <= selected_row < count:
                    raise IndexError(f"query row {selected_row} outside {record} with {count} queries")
                frame_index = int(handle["query_frame_index"][selected_row])
                agent = cv2.resize(
                    np.asarray(handle["frames/agentview_rgb"][frame_index]),
                    (224, 224),
                    interpolation=cv2.INTER_AREA,
                )
                wrist = cv2.resize(
                    np.asarray(handle["frames/eye_in_hand_rgb"][frame_index]),
                    (224, 224),
                    interpolation=cv2.INTER_AREA,
                )
                state = np.asarray(handle["states"][frame_index])
                instruction = str(handle.attrs["instruction"])
                stored = np.asarray(handle["latent_action_tokens"][selected_row])

            # During collection, observation.state was [1, 8] and
            # M1Inference wrapped it in one additional list before RPC.
            payload = {
                "batch_images": [[agent, wrist]],
                "instructions": [instruction],
                "unnorm_key": None,
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": 10,
                "return_latent_action_tokens": True,
                "state": [state[None, :]],
            }
            responses = [client.infer(payload) for _ in range(repeats)]
            if not all(response.get("ok", False) for response in responses):
                raise RuntimeError(f"policy server rejected {record}: {responses}")
            fresh_latents = [np.asarray(response["data"]["latent_action_tokens"])[0] for response in responses]
            normalized_actions = [np.asarray(response["data"]["normalized_actions"])[0] for response in responses]
            fresh = fresh_latents[0]
            fresh_stored_dtype = fresh.astype(np.float16)
            delta = fresh.astype(np.float32) - stored.astype(np.float32)
            storage_equal = bool(np.array_equal(fresh_stored_dtype, stored))
            repeated_latent_max_abs = max(
                (float(np.max(np.abs(item.astype(np.float32) - fresh.astype(np.float32)))) for item in fresh_latents[1:]),
                default=0.0,
            )
            repeated_action_mse = max(
                (float(np.mean(np.square(item.astype(np.float32) - normalized_actions[0].astype(np.float32)))) for item in normalized_actions[1:]),
                default=0.0,
            )
            results.append(
                {
                    "record": str(record.resolve()),
                    "query_row": selected_row,
                    "query_frame": frame_index,
                    "stored_shape": list(stored.shape),
                    "fresh_shape": list(fresh.shape),
                    "float16_storage_exact": storage_equal,
                    "float32_vs_stored_mae": float(np.mean(np.abs(delta))),
                    "float32_vs_stored_max_abs": float(np.max(np.abs(delta))),
                    "mismatched_float16_values": int(np.count_nonzero(fresh_stored_dtype != stored)),
                    "repeats": repeats,
                    "repeated_latent_max_abs": repeated_latent_max_abs,
                    "repeated_normalized_action_max_mse": repeated_action_mse,
                    "same_latent_different_policy_action": bool(
                        repeats > 1 and repeated_latent_max_abs == 0.0 and repeated_action_mse > 0.0
                    ),
                }
            )
    finally:
        client.close()

    return {
        "passed": all(item["float16_storage_exact"] for item in results),
        "records_checked": len(results),
        "host": host,
        "port": port,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--query-row", type=int, default=0, help="negative selects each record's middle query")
    parser.add_argument("--repeats", type=int, default=1, help="repeat identical RPCs to test action stochasticity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = recompute(
        args.record,
        source_root=args.source_root,
        host=args.host,
        port=args.port,
        query_row=args.query_row,
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
