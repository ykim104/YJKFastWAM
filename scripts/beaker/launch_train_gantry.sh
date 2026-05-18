#!/usr/bin/env bash
# Launch FastWAM training via Beaker Gantry (alternative to raw Beaker YAML).
#
# Usage:
#   ./scripts/beaker/launch_train_gantry.sh --user-name yejink --task libero_triple_2cam224_1e-4
#   ./scripts/beaker/launch_train_gantry.sh --user-name yejink --task libero_uncond_2cam224_1e-4 --gpus 8
#   ./scripts/beaker/launch_train_gantry.sh --user-name yejink --task libero_triple_2cam224_1e-4 --no-wandb
#   ./scripts/beaker/launch_train_gantry.sh --user-name yejink --task libero_triple_2cam224_1e-4 --fresh
#   ./scripts/beaker/launch_train_gantry.sh --user-name yejink --task libero_uncond_2cam224_1e-4 \
#     --resume /weka/oe-training/yejink/runs/libero_uncond_2cam224_1e-4/latest
# On restart after preemption, re-run the same command; auto-resumes from the latest Weka checkpoint.
#
# Gantry flags (--allow-dirty, etc.) are passed to gantry only, never to Hydra/train.py.
#
# Requires: pip install beaker-py beaker-gantry  (NOT the PyPI package named "beaker")
#           gantry config && beaker account whoami
# Gantry installs via --install (torch cu128 index + pip install -e . from pyproject.toml).
# Gantry clones this repo into the job workspace (--install + run_train.sh use that checkout).
# Weka (oe-training-default -> /weka/oe-training) is only for data, checkpoints, and runs.

set -euo pipefail

USER_NAME=""
TASK=""
NUM_GPUS=8
NUM_NODES=1
WORKSPACE="ai2/vida" # yejink-workspace"
BUDGET="ai2/robotics"
PRIORITY="high"
WEKA_BUCKET="oe-training-default"
WEKA_MOUNT="oe-training"
PRECOMPUTE_TEXT=0
WANDB=1
WANDB_SECRET="YEJINK_WANDB_API_KEY" #wandb-api-key"
BEAKER_AUTO_RESUME=1
FASTWAM_RUN_ID=""
RESUME_PATH=""
EXTRA=()
GANTRY_EXTRA=(--allow-dirty)

usage() {
  sed -n '2,12p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-name) USER_NAME="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --gpus) NUM_GPUS="$2"; shift 2 ;;
    --num-nodes) NUM_NODES="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --priority) PRIORITY="$2"; shift 2 ;;
    --weka-bucket) WEKA_BUCKET="$2"; shift 2 ;;
    --weka-mount) WEKA_MOUNT="$2"; shift 2 ;;
    --weka-volume) WEKA_BUCKET="$2"; shift 2 ;;  # alias for --weka-bucket
    --cluster) CLUSTER="$2"; shift 2 ;;
    --precompute-text) PRECOMPUTE_TEXT=1; shift ;;
    --low-vram) export BEAKER_LOW_VRAM=1; shift ;;
    --no-low-vram) export BEAKER_LOW_VRAM=0; shift ;;
    --allow-dirty) shift ;;  # default in GANTRY_EXTRA; do not add to Hydra overrides
    --wandb) WANDB=1; shift ;;
    --no-wandb) WANDB=0; shift ;;
    --wandb-secret) WANDB_SECRET="$2"; shift 2 ;;
    --fresh) BEAKER_AUTO_RESUME=0; shift ;;
    --run-id) FASTWAM_RUN_ID="$2"; shift 2 ;;
    --resume) RESUME_PATH="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "${USER_NAME}" && -n "${TASK}" ]] || usage

resolve_gantry() {
  if [[ -n "${GANTRY:-}" && -x "${GANTRY}" ]]; then
    echo "${GANTRY}"
    return 0
  fi
  if command -v gantry >/dev/null 2>&1; then
    command -v gantry
    return 0
  fi
  local repo_root="$1"
  if [[ -x "${repo_root}/.venv/bin/gantry" ]]; then
    echo "${repo_root}/.venv/bin/gantry"
    return 0
  fi
  if python -m gantry --help >/dev/null 2>&1; then
    echo "python -m gantry"
    return 0
  fi
  return 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEKA_ROOT="/weka/${WEKA_MOUNT}/${USER_NAME}"
DATA_ROOT="${WEKA_ROOT}/data"
CHECKPOINT_ROOT="${WEKA_ROOT}/checkpoints"
RUNS_ROOT="${WEKA_ROOT}/runs"

HYDRA_OVERRIDES="paths=weka paths.weka_user=${USER_NAME}"
if [[ "${WANDB}" == "1" ]]; then
  HYDRA_OVERRIDES="${HYDRA_OVERRIDES} wandb.enabled=true"
fi
if [[ -n "${RESUME_PATH}" ]]; then
  HYDRA_OVERRIDES="${HYDRA_OVERRIDES} resume=${RESUME_PATH}"
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  HYDRA_OVERRIDES="${HYDRA_OVERRIDES} ${EXTRA[*]}"
fi

GANTRY_ARGS=(
  run
  --yes
  "${GANTRY_EXTRA[@]}"
  --workspace "${WORKSPACE}"
  --budget "${BUDGET}"
  --priority "${PRIORITY}"
  --gpus "${NUM_GPUS}"
  --replicas "${NUM_NODES}"
  --shared-memory 64GiB
  --memory 200GiB
  --weka "${WEKA_BUCKET}:/weka/${WEKA_MOUNT}"
  --cluster "ai2/saturn"
  --cluster "ai2/jupiter"
#  --cluster "ai2/neptune"
#  --cluster "ai2/rhea"
#  --cluster "ai2/ceres"
  --env "USER_NAME=${USER_NAME}"
  --env "TASK=${TASK}"
  --env "NUM_GPUS=${NUM_GPUS}"
  --env "PRECOMPUTE_TEXT=${PRECOMPUTE_TEXT}"
  --env "FASTWAM_DATA_ROOT=${DATA_ROOT}"
  --env "FASTWAM_CHECKPOINTS_ROOT=${CHECKPOINT_ROOT}"
  --env "FASTWAM_RUNS_ROOT=${RUNS_ROOT}"
  --env "DIFFSYNTH_MODEL_BASE_PATH=${CHECKPOINT_ROOT}"
  --env "HYDRA_OVERRIDES=${HYDRA_OVERRIDES}"
  --env "BEAKER_LOW_VRAM=${BEAKER_LOW_VRAM:-auto}"
  --env "BEAKER_AUTO_RESUME=${BEAKER_AUTO_RESUME}"
  --env "FASTWAM_RUN_ID=${FASTWAM_RUN_ID}"
  --python-manager uv
  --uv-torch-backend cu128
  --default-python-version 3.10
  --install " (command -v apt-get >/dev/null && apt-get update -qq && apt-get install -y -qq python3.10-dev build-essential) || true; unset CUDA_HOME; uv pip install -e . --torch-backend cu128"
  --name "fastwam-${TASK}"
  --description "FastWAM ${TASK} (${USER_NAME})"
)

GANTRY_ARGS+=(--propagate-preemption)
if [[ "${NUM_NODES}" -gt 1 ]]; then
  GANTRY_ARGS+=(--leader-selection --host-networking --propagate-failure)
fi

if [[ "${WANDB}" == "1" ]]; then
  GANTRY_ARGS+=(
    --env-secret "WANDB_API_KEY=${WANDB_SECRET}"
    --env "WANDB_MODE=online"
  )
fi

cd "${REPO_ROOT}"
if ! GANTRY_CMD="$(resolve_gantry "${REPO_ROOT}")"; then
  echo "Error: 'gantry' not found. Install Beaker Gantry in this environment:" >&2
  echo "  pip install beaker-gantry" >&2
  echo "  gantry config   # one-time setup (Beaker token, default workspace, etc.)" >&2
  echo "Or submit without Gantry:" >&2
  echo "  ./scripts/beaker/launch_train.sh --user-name ${USER_NAME} --task ${TASK}" >&2
  exit 127
fi

echo "[paths] data=${DATA_ROOT} checkpoints=${CHECKPOINT_ROOT} runs=${RUNS_ROOT}"
echo "[wandb] enabled=$([[ "${WANDB}" == "1" ]] && echo true || echo false)"
echo "[resume] auto=$([[ "${BEAKER_AUTO_RESUME}" == "1" ]] && echo true || echo false) run_id=${FASTWAM_RUN_ID:-<auto>}"
echo ">>> ${GANTRY_CMD} ${GANTRY_ARGS[*]} -- bash scripts/beaker/run_train.sh"
# shellcheck disable=SC2086
exec ${GANTRY_CMD} "${GANTRY_ARGS[@]}" -- bash scripts/beaker/run_train.sh
