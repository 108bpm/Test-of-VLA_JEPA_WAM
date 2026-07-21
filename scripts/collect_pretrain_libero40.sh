#!/usr/bin/env bash
# Collect disjoint zero-shot LIBERO task shards from the pretrain checkpoint.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 POLICY_PORT WORKER_ID NUM_WORKERS" >&2
  exit 2
fi

policy_port="$1"
worker_id="$2"
num_workers="$3"
if ! [[ "$worker_id" =~ ^[0-9]+$ && "$num_workers" =~ ^[1-9][0-9]*$ ]] || (( worker_id >= num_workers )); then
  echo "Require 0 <= WORKER_ID < NUM_WORKERS" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(dirname "$repo_root")"
vla_root="$project_root/VLA-JEPA"
libero_root="$project_root/LIBERO"
dataset_root="${DATASET_ROOT:-$repo_root/datasets/vla_jepa_pretrain_libero40_v1}"
checkpoint="${CHECKPOINT_PATH:-$vla_root/checkpoints/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt}"
max_restarts="${MAX_WORKER_RESTARTS:-8}"

mkdir -p "$dataset_root/videos" "$dataset_root/records" "$dataset_root/logs"

run_suite() {
  local suite="$1"
  shift
  local attempts=0
  local log_path="$dataset_root/logs/${suite}_worker${worker_id}.log"

  while true; do
    echo "[$(date -Is)] worker=$worker_id suite=$suite attempt=$((attempts + 1))" >> "$log_path"
    if conda run -n libero env \
      MALLOC_ARENA_MAX=2 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMBA_NUM_THREADS=1 \
      NUMBA_CACHE_DIR="/tmp/libero_pretrain_numba_worker${worker_id}" \
      MPLCONFIGDIR="/tmp/libero_pretrain_mpl_worker${worker_id}" \
      LIBERO_CONFIG_PATH=/tmp/libero_eval_config \
      MUJOCO_GL=egl \
      MUJOCO_EGL_DEVICE_ID=0 \
      TOKENIZERS_PARALLELISM=false \
      PYTHONPATH="$vla_root:$libero_root" \
      python "$vla_root/examples/LIBERO/eval_libero.py" \
        --args.host 127.0.0.1 \
        --args.port "$policy_port" \
        --args.pretrained-path "$checkpoint" \
        --args.task-suite-name "$suite" \
        --args.task-ids "$@" \
        --args.num-trials-per-task 10 \
        --args.video-out-path "$dataset_root/videos/$suite" \
        --args.rollout-data-path "$dataset_root/records" \
        --args.with-state false \
        --args.request-timeout 120 \
        --args.step-timeout 180 \
        --args.resume \
        >> "$log_path" 2>&1; then
      return 0
    fi

    attempts=$((attempts + 1))
    if (( attempts > max_restarts )); then
      echo "[$(date -Is)] worker=$worker_id suite=$suite exceeded $max_restarts restarts" >> "$log_path"
      return 1
    fi
    echo "[$(date -Is)] worker=$worker_id suite=$suite failed; retrying in 15s" >> "$log_path"
    sleep 15
  done
}

read -r -a suites <<< "${COLLECT_SUITES:-libero_spatial libero_object libero_goal libero_10}"
for suite in "${suites[@]}"; do
  case "$suite" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *)
      echo "Unsupported suite '$suite'; this collection intentionally excludes LIBERO-90" >&2
      exit 2
      ;;
  esac

  task_ids=$(seq "$worker_id" "$num_workers" 9)
  # shellcheck disable=SC2086 # Task ids intentionally expand into arguments.
  run_suite "$suite" $task_ids
done
