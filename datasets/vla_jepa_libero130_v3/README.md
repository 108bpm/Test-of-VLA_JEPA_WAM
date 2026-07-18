# VLA-JEPA × LIBERO latent-world-model rollouts (v3)

This directory is the canonical, deduplicated rollout dataset for evaluating the
action-conditioned `latent_world_model`.  It contains VLA-JEPA policy rollouts
in the five standard LIBERO suites, together with the VLA-JEPA latent action
tokens used to produce the policy action chunks.

Read [`manifest.json`](manifest.json) for machine-readable counts, schema, and
collection configuration.

## Dataset summary

| Suite | Rollouts | Successful | Success rate |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 100 | 100 | 100.00% |
| `libero_object` | 100 | 100 | 100.00% |
| `libero_goal` | 100 | 100 | 100.00% |
| `libero_90` | 900 | 182 | 20.22% |
| `libero_10` | 100 | 98 | 98.00% |
| **Total** | **1300** | **480** | **36.92%** |

There are exactly 1300 complete HDF5 records and 1300 matching MP4 videos.
Each HDF5 is unique by `(task_suite, task_id, episode_id)`; no duplicate keys,
invalid HDF5 files, unmatched videos, or active writers remained at final
validation.  The records occupy about 57.25 GB and videos about 252 MB.

`libero_90` includes failed rollouts intentionally: they are still valid
action-conditioned trajectories for latent-dynamics evaluation.  Filter on the
HDF5 `success` attribute only when a success-only subset is required.

## Layout and identity

```text
vla_jepa_libero130_v3/
├── README.md
├── manifest.json
├── records/<suite>/rollout_taskNNN_episodeM.hdf5
└── videos/<suite>/rollout_taskNNN_episodeM_{success|failure}.mp4
```

`taskNNN` and `episodeM` are the only rollout identity fields.  Do not use
natural-language instructions as file identities: LIBERO contains distinct
tasks with identical instructions.

## HDF5 contents

Each record has the following datasets.

| HDF5 path | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `frames/agentview_rgb` | `[T, 256, 256, 3]` | `uint8` | Main camera RGB, vertically and horizontally rotated to match policy preprocessing. |
| `frames/eye_in_hand_rgb` | `[T, 256, 256, 3]` | `uint8` | Wrist-camera RGB with the same rotation. |
| `states` | `[T, 8]` | `float32` | End-effector position (3), axis-angle orientation (3), and gripper state (2). |
| `executed_actions` | `[T, 7]` | `float32` | Executed LIBERO action: translation (3), rotation (3), binarized gripper (1). |
| `query_frame_index` | `[N]` | `int64` | Frame index at which a policy query was made. |
| `latent_action_tokens` | `[N, 24, 2048]` | `float16` | VLA-JEPA latent action tokens returned at each policy query. |
| `unnormalized_action_chunks` | `[N, 7, 7]` | `float32` | Corresponding unnormalized 7-step policy action chunks. |

Frame and latent-token storage uses LZF compression where applicable.  `T` and
`N` vary by rollout.  Across the final dataset, `T` ranges from 69 to 520 and
`N` ranges from 10 to 75 (suite-specific statistics are in the manifest).

Root attributes are:

```text
format_version, task_suite, task_id, episode_id, seed, instruction,
success, num_frames, num_policy_queries
```

## Temporal alignment

At every `query_frame_index[i]`, the policy observes the two recorded camera
frames and produces `latent_action_tokens[i]` plus a 7-step action chunk.  The
frame at that index is therefore the observation aligned with that latent
action.  For world-model evaluation, choose an 8-frame multi-view window whose
first frame is a query frame, encode it with the frozen V-JEPA2 encoder, and
provide its aligned `[24, 2048]` latent action token tensor to the model.

The standalone model's default configuration has 8 video frames, V-JEPA2
tubelet size 2, and 3 context latent steps.  Its predictor consumes `z0..z2`
and predicts the aligned future blocks `z1..z3`.

## Minimal reader

```python
from pathlib import Path
import h5py

root = Path("latent_world_model/datasets/vla_jepa_libero130_v3")
path = next((root / "records" / "libero_10").glob("*.hdf5"))

with h5py.File(path, "r") as record:
    print(record.attrs["instruction"], record.attrs["success"])
    agentview = record["frames/agentview_rgb"][:]       # [T, 256, 256, 3]
    wrist = record["frames/eye_in_hand_rgb"][:]         # [T, 256, 256, 3]
    query_indices = record["query_frame_index"][:]      # [N]
    latent_actions = record["latent_action_tokens"][:]  # [N, 24, 2048]
```

For a chosen query `i`, ensure the required 8-frame window is in range before
creating a sample: `query_indices[i] + 8 <= num_frames`.

## Collection and provenance

- Simulator: LIBERO native suites `LIBERO_SPATIAL`, `LIBERO_OBJECT`,
  `LIBERO_GOAL`, `LIBERO_90`, and `LIBERO_10`.
- Policy: the local VLA-JEPA LIBERO checkpoint
  `VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`.
- Visual encoder/action source: V-JEPA2 ViT-L at 256px; latent action tokens
  have hidden size 2048.
- Seed: 7, recorded per rollout.
- File schema version: 1.

The dataset was validated after collection by opening every HDF5, checking all
required datasets and final attributes, verifying `num_frames`, verifying the
one-to-one token/query count, checking unique rollout keys, and matching each
record with exactly one status-consistent MP4.
