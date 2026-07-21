# V-JEPA latent world model

[English](README.md) | [中文](README_CN.md)

Standalone, action-conditioned latent world model extracted from VLA-JEPA. It contains no Qwen, robot dataset, trainer, or action-policy dependency. The frozen V-JEPA2 encoder maps a multi-view video to latent patch tokens; the trainable predictor maps context tokens plus **external latent actions** to a time-aligned sequence of next latent states.

The finalized VLA-JEPA × LIBERO rollout dataset is available on Hugging Face:

**[Monita108/VLA_JEPA-on-libero](https://huggingface.co/datasets/Monita108/VLA_JEPA-on-libero)**

It contains 1,300 validated multi-view rollouts across all 130 tasks in
LIBERO-SPATIAL, LIBERO-OBJECT, LIBERO-GOAL, LIBERO-90, and LIBERO-10. The local
schema and collection statistics are documented in
[`datasets/vla_jepa_libero130_v3/README.md`](datasets/vla_jepa_libero130_v3/README.md).

## Install

```bash
git clone git@github.com:108bpm/Test-of-VLA_JEPA_WM.git latent_world_model
cd latent_world_model
pip install -e .
```

The encoder weights are not committed. Create `checkpoints/vjepa2-vitl-fpc64-256` as a symlink to a local Hugging Face V-JEPA2 checkpoint, or pass an HF repository ID/path to `LatentWorldModel`. The checkpoint must include the Hugging Face model files and video processor config.

```bash
ln -s /absolute/path/to/vjepa2-vitl-fpc64-256 checkpoints/vjepa2-vitl-fpc64-256
python example.py
```

## API

```python
from latent_world_model import LatentWorldModel, LatentWorldModelConfig

model = LatentWorldModel(
    encoder_path="checkpoints/vjepa2-vitl-fpc64-256",
    config=LatentWorldModelConfig(
        num_video_frames=8,
        num_views=2,
        latent_action_dim=2048,  # choose this to match your action encoder
        num_action_tokens_per_timestep=8,
    ),
)

# uint8/raw RGB video: [batch, views, frames, channels, height, width]
# With 8 frames and V-JEPA tubelet=2, there are 4 latent time steps z0...z3.
# The predictor consumes z0...z2 and predicts z1...z3.
predicted, target = model(videos, latent_actions)
loss = (predicted - target).abs().mean()
```

`latent_actions` is the extension point:

```text
[B, context_steps * num_action_tokens_per_timestep, latent_action_dim]
```

For the default VLA-JEPA-compatible configuration it is `[B, 24, 2048]`: 3 context latent steps × 8 action tokens/step. Tokens at context step *i* condition the transition from `z_i` to `z_(i+1)`. They may come from a language model, a policy network, a learned action tokenizer, or any other module.

For the supplied VLA-JEPA checkpoint, the 24 world-model tokens are Qwen
hidden states at `<|action_i|>` slots.  They are the native learned
action-conditioning representation used by both source training and this
package's evaluation; no one-to-one reconstruction of the policy's stochastic
7-step action chunk is required.  The representation semantics and empirical
shuffle controls are documented in [`FINAL_REPORT.md`](FINAL_REPORT.md).

For projects that already produce V-JEPA latent tensors, skip video encoding:

```python
predicted_next_latents = model.predict_from_latents(context_latents, latent_actions)
```

`context_latents` has shape `[B, context_steps * patches_per_frame, num_views * encoder_hidden]`. The result has the same shape and represents the next latent block for every context step. For the supplied V-JEPA2 ViT-L encoder at 256px, both are `[B, 768, 2048]` (three 256-token blocks). During training this prediction is compared with encoded target blocks `z1...z3`.

## Evaluating predicted future latents

The complete frozen-checkpoint LIBERO protocol (strict-causal/original-joint,
C1/C3, H1/AR-H3, controls, bootstrap statistics, and resumable commands) is
documented in [`EVALUATION.md`](EVALUATION.md). A VLA-JEPA checkpoint can be
loaded without eagerly materializing unrelated Qwen/action-head tensors:

```python
model.load_predictor_checkpoint("../VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt")
```

Only the `vj_predictor.*` prefix is loaded; the V-JEPA2 encoder remains frozen.

The single authoritative result document is
[`FINAL_REPORT.md`](FINAL_REPORT.md). It is a standalone account of the model
interface, dataset collection and validation, experiment design, strict-causal
results, implementation audit, complete joint-C3 evaluation, limitations, and
final conclusions. No other report is required to interpret the work. The
analysis entry point is `latent_world_model.evaluation.deep_analysis`; large
generated artifacts remain outside Git and are indexed by
`reports/ARTIFACT_MANIFEST.json`.

For a result-free description of the collection logic, variables, formal
conditions, staged experiment funnel, and statistical rules, see
[`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md). It intentionally omits
implementation-specific commands and code.

V-JEPA2 has **no pixel decoder**. The checked upstream implementation (`vjepa2`, commit `204698b`) contains an encoder and a latent predictor only: its training target is the frozen teacher encoder's patch features, not RGB pixels. Consequently, neither `facebook/vjepa2-vitl-fpc64-256` nor the upstream `vitl.pt` checkpoint contains a compatible latent-to-pixel decoder. There is no official decoder checkpoint to download for this encoder.

Use representation-space evaluation first:

```python
from latent_world_model import evaluate_latent_prediction

predicted, target = model(videos, latent_actions)
metrics = evaluate_latent_prediction(predicted, target)
# l1, mse, mean_token_cosine, retrieval_accuracy
```

`retrieval_accuracy` measures whether each predicted future representation is nearest to its own ground-truth future within the batch. It helps detect representation collapse; use batches with at least two samples.

Pixel reconstruction requires a **newly trained** decoder. It must be trained on pairs `(frozen V-JEPA2 latent, original RGB frame)` from your target data and must exactly match this package's encoder setup: `facebook/vjepa2-vitl-fpc64-256`, 256px input, patch size 16, tubelet size 2, and the multi-view concatenation convention. Such a decoder has its own checkpoint (for example `checkpoints/pixel_decoder.pt`); it is not derivable from, nor included in, the V-JEPA2 encoder checkpoint. Train it before using pixel-level metrics such as PSNR, SSIM, or LPIPS.

## What is trainable

`model.encoder` is loaded frozen, kept in evaluation mode, and always run under `torch.no_grad()`. Train `model.predictor` only, unless your project explicitly changes this behavior. `model.loss(...)` uses the VLA-JEPA L1 latent prediction objective.

## Provenance

`predictor.py`, `vj2_modules.py`, and `vj2_tensors.py` are extracted from VLA-JEPA's action-conditioned V-JEPA predictor and retain their upstream Meta/V-JEPA licensing headers where applicable.
