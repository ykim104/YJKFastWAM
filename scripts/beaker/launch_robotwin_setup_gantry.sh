#!/usr/bin/env bash
# One-shot: build curobo into a Weka-staged RoboTwin via Beaker Gantry.
#
# Run this ONCE (after staging assets/ + task_config/ on Weka). It builds curobo's
# CUDA extensions in the SAME image the eval job uses, caching them on Weka so eval
# jobs reuse them without rebuilding.
#
# Usage:
#   ./scripts/beaker/launch_robotwin_setup_gantry.sh --user-name yejink \
#     --robotwin-env /weka/oe-training/yejink/YJKFastWAM/third_party/RoboTwin
#
# Requires: pip install beaker-gantry && gantry config
set -euo pipefail

USER_NAME=""
NUM_GPUS=1
WORKSPACE="ai2/vida"
BUDGET="ai2/robotics"
PRIORITY="high"
WEKA_BUCKET="oe-training-default"
WEKA_MOUNT="oe-training"
ROBOTWIN_ENV_DIR=""
GANTRY_EXTRA=(--allow-dirty)

usage() {
  cat <<'EOF'
Usage:
  launch_robotwin_setup_gantry.sh --user-name NAME [--robotwin-env PATH] [options]

Options:
  --user-name NAME       Weka user (paths under /weka/oe-training/NAME)
  --robotwin-env PATH    Weka RoboTwin dir (must already have assets/ + task_config/)
                         default: /weka/oe-training/NAME/robotwin/RoboTwin
  --gpus N               GPUs (default: 1; only needed so curobo can verify on a GPU node)
  --workspace, --budget, --priority, --weka-bucket, --weka-mount
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-name) USER_NAME="$2"; shift 2 ;;
    --robotwin-env) ROBOTWIN_ENV_DIR="$2"; shift 2 ;;
    --gpus) NUM_GPUS="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --priority) PRIORITY="$2"; shift 2 ;;
    --weka-bucket) WEKA_BUCKET="$2"; shift 2 ;;
    --weka-mount) WEKA_MOUNT="$2"; shift 2 ;;
    --weka-volume) WEKA_BUCKET="$2"; shift 2 ;;
    --allow-dirty) shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "${USER_NAME}" ]] || usage

resolve_gantry() {
  if [[ -n "${GANTRY:-}" && -x "${GANTRY}" ]]; then echo "${GANTRY}"; return 0; fi
  if command -v gantry >/dev/null 2>&1; then command -v gantry; return 0; fi
  local repo_root="$1"
  if [[ -x "${repo_root}/.venv/bin/gantry" ]]; then echo "${repo_root}/.venv/bin/gantry"; return 0; fi
  if python -m gantry --help >/dev/null 2>&1; then echo "python -m gantry"; return 0; fi
  return 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEKA_ROOT="/weka/${WEKA_MOUNT}/${USER_NAME}"
if [[ -z "${ROBOTWIN_ENV_DIR}" ]]; then
  ROBOTWIN_ENV_DIR="${WEKA_ROOT}/robotwin/RoboTwin"
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
  --shared-memory 16GiB
  --memory 64GiB
  --weka "${WEKA_BUCKET}:/weka/${WEKA_MOUNT}"
  --cluster "ai2/saturn"
  --cluster "ai2/jupiter"
  --cluster "ai2/neptune"
  --env "USER_NAME=${USER_NAME}"
  --env "ROBOTWIN_ENV_DIR=${ROBOTWIN_ENV_DIR}"
  --python-manager uv
  --uv-torch-backend cu128
  --default-python-version 3.10
  --install " (command -v apt-get >/dev/null && apt-get update -qq && apt-get install -y -qq python3.10-dev build-essential git) || true; unset CUDA_HOME; uv pip install -e . --torch-backend cu128"
  --name "fastwam-rt-setup"
  --description "FastWAM RoboTwin env build (curobo) for ${USER_NAME} -> ${ROBOTWIN_ENV_DIR}"
)

cd "${REPO_ROOT}"
if ! GANTRY_CMD="$(resolve_gantry "${REPO_ROOT}")"; then
  echo "Error: 'gantry' not found. Install: pip install beaker-gantry && gantry config" >&2
  exit 127
fi

chmod +x "${REPO_ROOT}/scripts/beaker/run_robotwin_setup.sh"

echo "[robotwin] env=${ROBOTWIN_ENV_DIR}"
echo ">>> ${GANTRY_CMD} ${GANTRY_ARGS[*]} -- bash scripts/beaker/run_robotwin_setup.sh"
# shellcheck disable=SC2086
exec ${GANTRY_CMD} "${GANTRY_ARGS[@]}" -- bash scripts/beaker/run_robotwin_setup.sh
