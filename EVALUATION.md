# VLA-JEPA × LIBERO latent prediction evaluation

The high-level, result-free experiment framework is documented in
[`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md). This file records the
implementation-oriented evaluation details and commands.

This evaluation is representation-space only. V-JEPA2 is an encoder/teacher;
there is no compatible public latent-to-RGB decoder. The predictor and encoder
are never trained or fine-tuned by these commands.

## Data and environments

Use only `datasets/vla_jepa_libero130_v3` and its deterministic index. The v3
contract contains 1300 validated HDF5/video pairs (10+10+10+100+100 rollouts)
and three early/middle/late query windows per rollout. The index requires 12
future frames so both H1 and autoregressive H3 can be evaluated; it contains
3900 windows. HDF5 reading and indexing work in `libero`; model inference
requires `VLA_JEPA` (Transformers ≥4.57, CUDA, and h5py).

```bash
PYTHONPATH=$PWD conda run -n VLA_JEPA python - <<'PY'
from latent_world_model.evaluation.index import build_rollout_index
build_rollout_index(
    "datasets/vla_jepa_libero130_v3",
    "evaluation_outputs/index.jsonl",
    required_future_frames=12,
)
PY
```

## Causal protocols

With an 8-frame clip and tubelet size 2, the encoder yields `z0...z3` (256
patches per block). The current state is `z2`, and H1 predicts `z3` using the
last action-token group. C3 supplies `z0,z1,z2` and all three action groups.

* `strict_causal`: each state block is encoded from a separate 8-frame clip
  ending at that block; left padding is used at episode start. No encoder call
  sees a frame after the represented state.
* `original_joint`: the original contiguous 8-frame joint encoding is used as
  predictor input. Its prediction is always scored against the corresponding
  strict-causal target, exposing any future-frame leakage.
* H3 is autoregressive, not a new multi-horizon head. C1 rolls `z2` forward;
  C3 rolls a three-state history with action windows `[g0,g1,g2]`,
  `[g1,g2,next-g0]`, `[g2,next-g0,next-g1]`.

Single-view controls duplicate the selected RGB view into the second input
slot because the released predictor has a fixed two-view 2048-dimensional
input. They answer whether the selected camera is sufficient without changing
weights.

## Compact funnel

Stage 0 is a 20-window smoke run covering shape/dtype/finite checks, checkpoint
loading, all condition code paths, and resumability. Stage 1 selects the lowest
episode id per task (one rollout/task, 390 windows) and runs `S0...S9`:

`S0` strict C1-H1 correct action; `S1` strict C3-H1; `S2` strict C1-AR-H3;
`S3` strict C3-AR-H3; `S4` zero action; `S5` deterministic same-task/stage
shuffled action from the full index; `S6` action from the next policy query;
`S7` agentview only; `S8` wrist only; `S9` original joint encoding.

Stage 2 uses the user-requested half collection: five deterministic episodes
per task (episode ids `0,2,4,6,8`), i.e. 650 rollouts and 1950 windows. It
runs the six pre-registered formal conditions:
`F0` strict C1-H1; `F1` strict C3-H1; `F2` strict C1-AR-H3; `F3` zero
action; `F4` same-task/stage shuffled action; and `F5` original joint encoding.

Example commands (the checkpoint paths are local and are not committed):

```bash
COMMON="--dataset-root datasets/vla_jepa_libero130_v3 \
  --index evaluation_outputs/index.jsonl \
  --encoder ../VLA-JEPA/checkpoints/vjepa2-vitl-fpc64-256 \
  --checkpoint ../VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt \
  --clip-batch-size 6 --device cuda"

PYTHONPATH=$PWD conda run -n VLA_JEPA python -m latent_world_model.evaluation.runner \
  $COMMON --output evaluation_outputs/stage1 --conditions S0 S1 S2 S3 S4 S5 S6 S7 S8 S9

PYTHONPATH=$PWD conda run -n VLA_JEPA python -m latent_world_model.evaluation.runner \
  $COMMON --output evaluation_outputs/formal_shard0 --conditions F0 F1 F2 F3 F4 F5 \
  --rollouts-per-task 5 --num-shards 4 --shard-id 0

# Launch shard ids 1, 2, and 3 in separate terminals/processes. Each shard
# writes its own JSONL/memmaps; then merge without loading latent tensors:
PYTHONPATH=$PWD conda run -n VLA_JEPA python -m latent_world_model.evaluation.merge \
  --output evaluation_outputs/formal \
  evaluation_outputs/formal_shard0 evaluation_outputs/formal_shard1 \
  evaluation_outputs/formal_shard2 evaluation_outputs/formal_shard3

PYTHONPATH=$PWD conda run -n VLA_JEPA python -m latent_world_model.evaluation.report \
  evaluation_outputs/formal --bootstrap-replicates 1000
```

The runner writes one JSON object per `(window, condition)`, flushes after each
row, and stores compact float16 summary embeddings in NumPy memmaps for
retrieval metrics. Re-running the same command skips completed pairs. The
output includes `config.json`, checkpoint load status, metrics, summaries,
plots, and `report.md`.

## Deep analysis and report

After the formal half run has been merged and the stage-1/X0 outputs exist,
generate the collection statistics, paired effects, strata, correlations,
figures, and the machine-readable `deep_summary.json` with:

```bash
MPLCONFIGDIR=/tmp/lwm_mpl PYTHONPATH=$PWD \
  conda run --no-capture-output -n VLA_JEPA \
  python -m latent_world_model.evaluation.deep_analysis \
  --dataset-root datasets/vla_jepa_libero130_v3 \
  --formal-metrics evaluation_outputs/formal_half/metrics.jsonl \
  --screening-metrics evaluation_outputs/stage1/metrics.jsonl \
  --supplemental-metrics evaluation_outputs/stage1_supplemental/metrics.jsonl \
  --output evaluation_outputs/deep_analysis \
  --bootstrap-replicates 1000
```

The tracked interpretation is [`COMPREHENSIVE_REPORT.md`](COMPREHENSIVE_REPORT.md);
the generated report and CSV/PNG artifacts are in the output directory. The
script streams all 1300 HDF5 records and never loads the video frames or latent
memmaps into one large array.

## Metrics and inference rules

Primary metrics are MSE, `persistence_ratio` (MSE divided by keeping `z2`),
history/action gains, and H1/H2/H3 error growth. Auxiliary metrics are L1,
normalized MSE, token/delta cosine, delta-norm ratio, prediction/target
variance, and retrieval top-1/top-5. Stratify only by suite, task, success,
stage, latent-change quartile, action scale, and gripper category; do not form
an uncontrolled Cartesian product.

Means and paired differences use rollout as the observational unit. Confidence
intervals use a deterministic task→rollout hierarchical bootstrap (1000
replicates). Holm correction is applied only to the registered comparisons
`F0−persistence`, `F1−F0`, `F3−F0`, `F4−F0`, and `F5−F0`.
