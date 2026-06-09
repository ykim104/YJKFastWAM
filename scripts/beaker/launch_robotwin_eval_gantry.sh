#!/usr/bin/env bash
# Launch RoboTwin policy evaluation on Beaker Gantry.
#
# RoboTwin's sim env (assets/, task_config/, envs/curobo) is gitignored and
# curobo/pytorch3d need a real CUDA toolchain, so the job consumes a PREBUILT
# RoboTwin staged on Weka (see --robotwin-env). Gantry still ships your latest
# code via the git clone.
#
# Usage:
#   # Single-task smoke test (recommended first):
#   ./scripts/beaker/launch_robotwin_eval_gantry.sh --user-name yejink \
#     --task robotwin_uncond_3cam_384_1e-4 --latest \
#     --task-name place_dual_shoes --gpus 1
#
#   # Released checkpoint, all tasks:
#   ./scripts/beaker/launch_robotwin_eval_gantry.sh --user-name yejink \
#     --task robotwin_uncond_3cam_384_1e-4 \
#     --ckpt /weka/oe-training/yejink/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
#     --dataset-stats /weka/oe-training/yejink/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
#     --gpus 8
#
# CKPT accepts: .pt | run dir | task dir | .../latest | --latest (-> ${WEKA}/runs/${TASK}/latest)
#
# Requires: pip install beaker-gantry && gantry config

set -euo pipefail

USER_NAME=""
TASK=""
CKPT=""
USE_LATEST=0
NUM_GPUS=8
WORKSPACE="ai2/vida"
BUDGET="ai2/robotics"
PRIORITY="high"
WEKA_BUCKET="oe-training-default"
WEKA_MOUNT="oe-training"
DATASET_STATS_PATH=""
EVAL_OUTPUT_DIR=""
EVAL_RUN_TAG=""
ROBOTWIN_ENV_DIR=""
ROBOTWIN_TASK_NAME=""
SKIP_ROBOTWIN_INSTALL=0
EXTRA=()
GANTRY_EXTRA=(--allow-dirty)

usage() {
  cat <<'EOF'
Usage:
  launch_robotwin_eval_gantry.sh --user-name NAME --task TASK (--ckpt PATH | --latest) [options]

Options:
  --user-name NAME          Weka user (paths under /weka/oe-training/NAME)
  --task TASK               Hydra task (e.g. robotwin_uncond_3cam_384_1e-4)
  --ckpt PATH               Checkpoint: .pt, run dir, task dir, or .../latest symlink
  --latest                  Use ${WEKA}/runs/${TASK}/latest
  --task-name NAME          Single RoboTwin task to eval (default: all tasks)
  --gpus N                  GPUs for parallel eval (default: 8)
  --robotwin-env PATH       Prebuilt RoboTwin on Weka (assets/ task_config/ envs/curobo/)
                            default: /weka/oe-training/NAME/robotwin/RoboTwin
  --dataset-stats PATH      dataset_stats.json (auto-detected from run dir if omitted)
  --eval-output-dir PATH    Results root (default: .../runs/eval/robotwin/${TASK}/<timestamp>)
  --eval-run-tag TAG        Subdir name instead of timestamp under eval output
  --skip-robotwin-install   Skip sim dependency install (image already has them)
  --workspace, --budget, --priority, --weka-bucket, --weka-mount
  --allow-dirty             Pass through to gantry (default on)
  Extra tokens are forwarded as Hydra overrides (e.g. EVALUATION.eval_num_episodes=20)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-name) USER_NAME="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --latest) USE_LATEST=1; shift ;;
    --task-name) ROBOTWIN_TASK_NAME="$2"; shift 2 ;;
    --gpus) NUM_GPUS="$2"; shift 2 ;;
    --robotwin-env) ROBOTWIN_ENV_DIR="$2"; shift 2 ;;
    --dataset-stats) DATASET_STATS_PATH="$2"; shift 2 ;;
    --eval-output-dir) EVAL_OUTPUT_DIR="$2"; shift 2 ;;
    --eval-run-tag) EVAL_RUN_TAG="$2"; shift 2 ;;
    --skip-robotwin-install) SKIP_ROBOTWIN_INSTALL=1; shift ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --priority) PRIORITY="$2"; shift 2 ;;
    --weka-bucket) WEKA_BUCKET="$2"; shift 2 ;;
    --weka-mount) WEKA_MOUNT="$2"; shift 2 ;;
    --weka-volume) WEKA_BUCKET="$2"; shift 2 ;;
    --allow-dirty) shift ;;
    -h|--help) usage ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "${USER_NAME}" && -n "${TASK}" ]] || usage
if [[ "${USE_LATEST}" == "1" && -n "${CKPT}" ]]; then
  echo "Error: use only one of --ckpt or --latest" >&2
  exit 1
fi
if [[ "${USE_LATEST}" == "1" ]]; then
  CKPT="latest"
elif [[ -z "${CKPT}" ]]; then
  usage
fi

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
if [[ -z "${ROBOTWIN_ENV_DIR}" ]]; then
  ROBOTWIN_ENV_DIR="${WEKA_ROOT}/robotwin/RoboTwin"
fi

EVAL_HYDRA_OVERRIDES=""
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  EVAL_HYDRA_OVERRIDES="${EXTRA[*]}"
fi

JOB_NAME="fastwam-rt-eval-${TASK}"
if [[ "${CKPT}" == "latest" ]]; then
  JOB_NAME="${JOB_NAME}-latest"
else
  JOB_NAME="${JOB_NAME}-$(basename "${CKPT}" .pt | tr '/' '-' | cut -c1-40)"
fi
if [[ -n "${ROBOTWIN_TASK_NAME}" ]]; then
  JOB_NAME="${JOB_NAME}-$(echo "${ROBOTWIN_TASK_NAME}" | tr '/' '-' | cut -c1-24)"
fi

GANTRY_ARGS=(
  run
  --yes
  "${GANTRY_EXTRA[@]}"
  --workspace "${WORKSPACE}"
  --budget "${BUDGET}"
  --priority "${PRIORITY}"
  --gpus "${NUM_GPUS}"
  --replicas 1
  --shared-memory 32GiB
  --memory 128GiB
  --weka "${WEKA_BUCKET}:/weka/${WEKA_MOUNT}"
  --cluster "ai2/saturn"
  --cluster "ai2/jupiter"
  --cluster "ai2/neptune"
  --env "USER_NAME=${USER_NAME}"
  --env "TASK=${TASK}"
  --env "CKPT=${CKPT}"
  --env "NUM_GPUS=${NUM_GPUS}"
  --env "FASTWAM_DATA_ROOT=${DATA_ROOT}"
  --env "FASTWAM_CHECKPOINTS_ROOT=${CHECKPOINT_ROOT}"
  --env "FASTWAM_RUNS_ROOT=${RUNS_ROOT}"
  --env "DIFFSYNTH_MODEL_BASE_PATH=${CHECKPOINT_ROOT}"
  --env "ROBOTWIN_ENV_DIR=${ROBOTWIN_ENV_DIR}"
  --env "EVAL_HYDRA_OVERRIDES=${EVAL_HYDRA_OVERRIDES}"
  --env "SKIP_ROBOTWIN_INSTALL=${SKIP_ROBOTWIN_INSTALL}"
  # SAPIEN renders via Vulkan. The NVIDIA container runtime only injects the
  # graphics/Vulkan driver stack when these capabilities are requested at container
  # start; the default (compute,utility) yields "failed to find a rendering device".
  --env "NVIDIA_DRIVER_CAPABILITIES=all"
  --env "NVIDIA_VISIBLE_DEVICES=all"
  --python-manager uv
  --uv-torch-backend cu128
  --default-python-version 3.10
  --install " (command -v apt-get >/dev/null && apt-get update -qq && apt-get install -y -qq python3.10-dev build-essential git) || true; unset CUDA_HOME; uv pip install -e . --torch-backend cu128"
  --name "${JOB_NAME}"
  --description "FastWAM RoboTwin eval ${TASK} (${USER_NAME}) ckpt=${CKPT}"
)

if [[ -n "${DATASET_STATS_PATH}" ]]; then
  GANTRY_ARGS+=(--env "DATASET_STATS_PATH=${DATASET_STATS_PATH}")
fi
if [[ -n "${EVAL_OUTPUT_DIR}" ]]; then
  GANTRY_ARGS+=(--env "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}")
fi
if [[ -n "${EVAL_RUN_TAG}" ]]; then
  GANTRY_ARGS+=(--env "EVAL_RUN_TAG=${EVAL_RUN_TAG}")
fi
if [[ -n "${ROBOTWIN_TASK_NAME}" ]]; then
  GANTRY_ARGS+=(--env "ROBOTWIN_TASK_NAME=${ROBOTWIN_TASK_NAME}")
fi

cd "${REPO_ROOT}"
if ! GANTRY_CMD="$(resolve_gantry "${REPO_ROOT}")"; then
  echo "Error: 'gantry' not found. Install: pip install beaker-gantry && gantry config" >&2
  exit 127
fi

chmod +x "${REPO_ROOT}/scripts/beaker/run_robotwin_eval.sh"

echo "[paths]    data=${DATA_ROOT} checkpoints=${CHECKPOINT_ROOT} runs=${RUNS_ROOT}"
echo "[robotwin] env=${ROBOTWIN_ENV_DIR}"
echo "[eval]     task=${TASK} ckpt=${CKPT} gpus=${NUM_GPUS} task_name=${ROBOTWIN_TASK_NAME:-<all>}"
echo ">>> ${GANTRY_CMD} ${GANTRY_ARGS[*]} -- bash scripts/beaker/run_robotwin_eval.sh"
# shellcheck disable=SC2086
exec ${GANTRY_CMD} "${GANTRY_ARGS[@]}" -- bash scripts/beaker/run_robotwin_eval.sh
