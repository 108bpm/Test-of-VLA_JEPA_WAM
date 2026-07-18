"""Replay recorded actions in LIBERO and compare simulator observations.

This is an environment-level audit: it reconstructs the exact task and initial
state, performs the ten stabilization steps used during collection, then sends
the stored ``executed_actions`` back to LIBERO.  Before every action it compares
the simulator state and both rendered cameras with the lossless HDF5 record.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * np.arccos(quat[3]) / denominator).astype(np.float64)


def replay_record(record_path: Path, *, warmup_steps: int = 10, resolution: int = 256) -> dict[str, Any]:
    with h5py.File(record_path, "r") as handle:
        suite_name = str(handle.attrs["task_suite"])
        task_id = int(handle.attrs["task_id"])
        episode_id = int(handle.attrs["episode_id"])
        seed = int(handle.attrs["seed"])
        expected_success = bool(handle.attrs["success"])
        expected_instruction = str(handle.attrs["instruction"])
        actions = np.asarray(handle["executed_actions"][:], dtype=np.float64)
        expected_states = np.asarray(handle["states"][:], dtype=np.float64)
        expected_agent = handle["frames/agentview_rgb"]
        expected_wrist = handle["frames/eye_in_hand_rgb"]

        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        env.seed(seed)

        state_abs_max = 0.0
        state_abs_sum = 0.0
        state_value_count = 0
        agent_abs_max = 0
        wrist_abs_max = 0
        agent_abs_sum = 0
        wrist_abs_sum = 0
        pixel_value_count = 0
        first_state_mismatch: int | None = None
        first_agent_mismatch: int | None = None
        first_wrist_mismatch: int | None = None
        done_steps: list[int] = []
        final_success = False
        try:
            env.reset()
            obs = env.set_init_state(initial_states[episode_id])
            for _ in range(warmup_steps):
                obs, _, _, _ = env.step(DUMMY_ACTION)

            for step, action in enumerate(actions):
                agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                state = np.concatenate(
                    [
                        np.asarray(obs["robot0_eef_pos"]),
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        np.asarray(obs["robot0_gripper_qpos"]),
                    ]
                )

                state_error = np.abs(state - expected_states[step])
                agent_error = np.abs(agent.astype(np.int16) - expected_agent[step].astype(np.int16))
                wrist_error = np.abs(wrist.astype(np.int16) - expected_wrist[step].astype(np.int16))
                state_abs_max = max(state_abs_max, float(state_error.max(initial=0.0)))
                state_abs_sum += float(state_error.sum())
                state_value_count += int(state_error.size)
                agent_abs_max = max(agent_abs_max, int(agent_error.max(initial=0)))
                wrist_abs_max = max(wrist_abs_max, int(wrist_error.max(initial=0)))
                agent_abs_sum += int(agent_error.sum())
                wrist_abs_sum += int(wrist_error.sum())
                pixel_value_count += int(agent_error.size)
                if first_state_mismatch is None and np.any(state_error > 1e-5):
                    first_state_mismatch = step
                if first_agent_mismatch is None and np.any(agent_error != 0):
                    first_agent_mismatch = step
                if first_wrist_mismatch is None and np.any(wrist_error != 0):
                    first_wrist_mismatch = step

                obs, _, done, _ = env.step(action.tolist())
                if done:
                    done_steps.append(step)
            final_success = bool(env.check_success())
        finally:
            env.close()

    state_mae = state_abs_sum / state_value_count if state_value_count else math.nan
    agent_mae = agent_abs_sum / pixel_value_count if pixel_value_count else math.nan
    wrist_mae = wrist_abs_sum / pixel_value_count if pixel_value_count else math.nan
    passed = (
        state_abs_max <= 1e-5
        and agent_abs_max == 0
        and wrist_abs_max == 0
        and final_success == expected_success
        and str(task.language) == expected_instruction
    )
    return {
        "record": str(record_path.resolve()),
        "passed": passed,
        "suite": suite_name,
        "task_id": task_id,
        "episode_id": episode_id,
        "seed": seed,
        "steps_replayed": int(actions.shape[0]),
        "instruction_matches": str(task.language) == expected_instruction,
        "expected_success": expected_success,
        "replay_success": final_success,
        "done_steps": done_steps,
        "state": {
            "mae": state_mae,
            "max_abs_error": state_abs_max,
            "first_mismatch_over_1e-5": first_state_mismatch,
        },
        "agentview_rgb": {
            "mae": agent_mae,
            "max_abs_error": agent_abs_max,
            "first_nonidentical_frame": first_agent_mismatch,
        },
        "eye_in_hand_rgb": {
            "mae": wrist_mae,
            "max_abs_error": wrist_abs_max,
            "first_nonidentical_frame": first_wrist_mismatch,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    results = [
        replay_record(record, warmup_steps=args.warmup_steps, resolution=args.resolution)
        for record in args.record
    ]
    payload = {
        "passed": all(result["passed"] for result in results),
        "records_replayed": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
