#!/usr/bin/env bash
# Start the frozen VLA-JEPA pretrain checkpoint for zero-shot LIBERO collection.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(dirname "$repo_root")"
vla_root="$project_root/VLA-JEPA"
checkpoint="${CHECKPOINT_PATH:-$vla_root/checkpoints/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt}"
port="${1:-15193}"

if [[ ! -f "$checkpoint" ]]; then
  echo "Missing pretrained checkpoint: $checkpoint" >&2
  exit 1
fi

exec conda run --no-capture-output -n VLA_JEPA env \
  PYTHONPATH="$vla_root" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false \
  python "$vla_root/deployment/model_server/server_policy.py" \
    --ckpt_path "$checkpoint" \
    --port "$port" \
    --use_bf16 \
    --cuda 0
