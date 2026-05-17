#!/usr/bin/env bash
# Runs inside a Beaker container. Invoked by launch_train.py / gantry.
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
TASK="${TASK:?Set TASK (e.g. libero_triple_2cam224_1e-4)}"
CODE_DIR="${CODE_DIR:-$(pwd)}"
PRECOMPUTE_TEXT="${PRECOMPUTE_TEXT:-0}"

cd "${CODE_DIR}"

WEKA_ROOT="${WEKA_ROOT:-/weka/oe-training-default/${USER_NAME:-yejink}}"
export FASTWAM_DATA_ROOT="${FASTWAM_DATA_ROOT:-${WEKA_ROOT}/data}"
export FASTWAM_CHECKPOINTS_ROOT="${FASTWAM_CHECKPOINTS_ROOT:-${WEKA_ROOT}/checkpoints}"
export FASTWAM_RUNS_ROOT="${FASTWAM_RUNS_ROOT:-${WEKA_ROOT}/runs}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_CHECKPOINTS_ROOT}}"
export HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-paths=weka paths.weka_user=${USER_NAME:-yejink}}"

echo "[beaker] data=${FASTWAM_DATA_ROOT} checkpoints=${FASTWAM_CHECKPOINTS_ROOT} runs=${FASTWAM_RUNS_ROOT}"

# Multi-node (Beaker leader selection + host networking).
if [[ -n "${BEAKER_REPLICA_COUNT:-}" && "${BEAKER_REPLICA_COUNT}" -gt 1 ]]; then
  export NNODES="${BEAKER_REPLICA_COUNT}"
  export NODE_RANK="${BEAKER_REPLICA_RANK}"
  export MASTER_ADDR="${BEAKER_LEADER_REPLICA_HOSTNAME}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  echo "[beaker] multi-node NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR}"
fi

mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}" "${FASTWAM_RUNS_ROOT}"

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  if ! python -c "import fastwam, torch" 2>/dev/null; then
    echo "[beaker] Installing fastwam (torch cu128 + editable install from pyproject.toml)..."
    pip install -U pip
    pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchcodec==0.5 \
      --extra-index-url https://download.pytorch.org/whl/cu128
    pip install -e .
  fi
fi

if [[ "${PRECOMPUTE_TEXT}" == "1" ]]; then
  echo "[beaker] Precomputing T5 text embeddings for task=${TASK}..."
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    scripts/precompute_text_embeds.py "task=${TASK}"
fi

echo "[beaker] Starting training: NUM_GPUS=${NUM_GPUS} TASK=${TASK}"
# shellcheck disable=SC2086
bash scripts/train_zero1.sh "${NUM_GPUS}" "task=${TASK}" ${HYDRA_OVERRIDES:-}
