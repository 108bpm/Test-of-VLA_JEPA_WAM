# VLA-JEPA pretrain zero-shot LIBERO rollouts

This dataset is the second evaluation collection for `latent_world_model`. It
uses the released `VLA-JEPA-pretrain.pt` checkpoint before any LIBERO
fine-tuning and evaluates it zero-shot in four LIBERO suites.

## Scope

| Suite | Tasks | Rollouts per task | Expected rollouts |
|---|---:|---:|---:|
| `libero_spatial` | 10 | 10 | 100 |
| `libero_object` | 10 | 10 | 100 |
| `libero_goal` | 10 | 10 | 100 |
| `libero_10` | 10 | 10 | 100 |
| **Total** | **40** | **10** | **400** |

`libero_90` is intentionally excluded.

## Model and zero-shot policy contract

- Checkpoint: `VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt`.
- Checkpoint SHA-256:
  `fd929c79d9bbd0bda56c0b952c7acb470d93c6241a519013fe5248c3f3ea5fab`.
- Pretraining robot data: DROID; auxiliary video data: Something-Something-v2.
- No LIBERO fine-tuning or parameter update is performed.
- Actions use the checkpoint's native DROID `franka` normalization statistics.
- Robot state is saved in each HDF5 but is not supplied to the policy action
  head, matching the pretraining configuration `with_state: false`.
- Seed: 7; policy action chunk size: 7.

The complete machine-readable provenance is in
[`collection_config.json`](collection_config.json).

## Stored data

The HDF5 and MP4 schema matches `vla_jepa_libero130_v3`: two RGB views, robot
state, executed actions, exact policy-query frame indices, `[N,24,2048]`
latent-action tokens, `[N,7,7]` unnormalized action chunks, task identity,
instruction, and rollout success.

Large records, videos, and worker logs remain local and are excluded from Git.
The finalized dataset manifest will be generated only after all 400 HDF5/MP4
pairs pass coverage, uniqueness, shape, finite-value, and frame-count checks.

## Pre-collection validation

A separate 40-control-step smoke rollout was completed before formal
collection. It produced one matching HDF5/MP4 pair with 40 video/HDF5 frames,
6 strictly increasing query indices, finite `[6,24,2048]` latent tokens, finite
`[6,7,7]` action chunks, and matching query counts. The truncated smoke output
is not part of this dataset.

## Collection commands

Start one GPU policy server:

```bash
scripts/start_pretrain_libero40_server.sh 15193
```

Then start one or more disjoint workers. For two workers:

```bash
scripts/collect_pretrain_libero40.sh 15193 0 2
scripts/collect_pretrain_libero40.sh 15193 1 2
```

Workers split task IDs modulo `NUM_WORKERS`, write separate logs, and resume
only complete HDF5/MP4 pairs. The collector rejects any suite outside the four
listed above, preventing accidental LIBERO-90 collection.
