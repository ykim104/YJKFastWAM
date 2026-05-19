#!/usr/bin/env bash
# Runs LIBERO full-benchmark eval inside a Beaker/Gantry container.
set -euo pipefail

beaker_on_err() {
  echo "[beaker-eval] ERROR: exit $? at ${BASH_SOURCE[1]}:${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
}
trap beaker_on_err ERR

TASK="${TASK:?Set TASK (e.g. libero_triple_2cam224_1e-4)}"
CKPT_INPUT="${CKPT:?Set CKPT (weights .pt, run dir, task dir, or .../latest)}"
NUM_GPUS="${NUM_GPUS:-8}"

if [[ -n "${CODE_DIR:-}" ]]; then
  if [[ ! -d "${CODE_DIR}" ]]; then
    echo "[beaker-eval] ERROR: CODE_DIR=${CODE_DIR} does not exist." >&2
    exit 1
  fi
  cd "${CODE_DIR}"
else
  echo "[beaker-eval] Using gantry/git checkout: $(pwd)"
fi

_BEAKER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup_job_env.sh
source "${_BEAKER_SCRIPT_DIR}/setup_job_env.sh"

WEKA_ROOT="${WEKA_ROOT:-/weka/oe-training/${USER_NAME:-yejink}}"
export FASTWAM_DATA_ROOT="${FASTWAM_DATA_ROOT:-${WEKA_ROOT}/data}"
export FASTWAM_CHECKPOINTS_ROOT="${FASTWAM_CHECKPOINTS_ROOT:-${WEKA_ROOT}/checkpoints}"
export FASTWAM_RUNS_ROOT="${FASTWAM_RUNS_ROOT:-${WEKA_ROOT}/runs}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_CHECKPOINTS_ROOT}}"

beaker_check_weka() {
  if [[ ! -d "/weka/oe-training" ]]; then
    echo "[beaker-eval] ERROR: Weka not mounted at /weka/oe-training" >&2
    ls -la /weka 2>&1 || true
    exit 1
  fi
  echo "[beaker-eval] Weka mount OK: /weka/oe-training"
}

beaker_resolve_ckpt_input() {
  local input="$1"
  if [[ "${input}" == "latest" ]]; then
    echo "${FASTWAM_RUNS_ROOT}/${TASK}/latest"
    return 0
  fi
  echo "${input}"
}

beaker_resolve_eval_paths() {
  local ckpt_input
  ckpt_input="$(beaker_resolve_ckpt_input "${CKPT_INPUT}")"
  local resolved
  resolved="$(
    PYTHON="${PYTHON:-python}"
    "${PYTHON}" - "${ckpt_input}" <<'PY'
import sys
from fastwam.utils.resume_paths import resolve_eval_ckpt

ckpt, stats = resolve_eval_ckpt(sys.argv[1])
print(ckpt)
print(stats or "")
PY
  )" || {
    echo "[beaker-eval] ERROR: failed to resolve CKPT=${ckpt_input}" >&2
    exit 1
  }
  export EVAL_CKPT="$(sed -n '1p' <<< "${resolved}")"
  local auto_stats
  auto_stats="$(sed -n '2p' <<< "${resolved}")"
  if [[ -z "${DATASET_STATS_PATH:-}" && -n "${auto_stats}" ]]; then
    export DATASET_STATS_PATH="${auto_stats}"
  fi
  echo "[beaker-eval] ckpt=${EVAL_CKPT}"
  if [[ -n "${DATASET_STATS_PATH:-}" ]]; then
    echo "[beaker-eval] dataset_stats=${DATASET_STATS_PATH}"
  else
    echo "[beaker-eval] WARNING: dataset_stats not found; pass DATASET_STATS_PATH or EVALUATION.dataset_stats_path"
  fi
}

beaker_install_eval_deps() {
  if [[ "${SKIP_LIBERO_INSTALL:-0}" == "1" ]]; then
    echo "[beaker-eval] SKIP_LIBERO_INSTALL=1"
    return 0
  fi

  echo "[beaker-eval] Installing eval dependencies (tmux, mujoco, LIBERO)..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux libglew-dev libosmesa6-dev patchelf >/dev/null || true
  fi

  # Pick installer: prefer uv (gantry venv has no `pip`), fall back to python -m pip.
  local _installer
  if command -v uv >/dev/null 2>&1; then
    _installer="uv_pip"
  elif "${PYTHON}" -m pip --version >/dev/null 2>&1; then
    _installer="py_pip"
  else
    echo "[beaker-eval] Bootstrapping pip via ensurepip..."
    "${PYTHON}" -m ensurepip --upgrade >/dev/null 2>&1 || true
    if "${PYTHON}" -m pip --version >/dev/null 2>&1; then
      _installer="py_pip"
    else
      echo "[beaker-eval] ERROR: neither uv nor pip available in venv ${VIRTUAL_ENV:-?}" >&2
      exit 1
    fi
  fi
  echo "[beaker-eval] Using installer: ${_installer}"

  pip_install() {
    if [[ "${_installer}" == "uv_pip" ]]; then
      uv pip install "$@"
    else
      "${PYTHON}" -m pip install --no-cache-dir "$@"
    fi
  }

  pip_install "mujoco==3.3.2"

  local libero_dir="${LIBERO_REPO:-/tmp/LIBERO}"
  if [[ ! -f "${libero_dir}/setup.py" ]]; then
    rm -rf "${libero_dir}"
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${libero_dir}"
  fi

  # Minimal runtime deps for OffScreenRenderEnv. Skip LIBERO's requirements.txt
  # because it pins old numpy/hydra/thop that conflict with the training venv.
  local libero_sim_deps=(
    "robosuite==1.4.0"
    "bddl==1.0.1"
    "gym==0.25.2"
    "cloudpickle==2.1.0"
    "easydict==1.9"
    "opencv-python-headless>=4.6.0.66"
    "matplotlib>=3.5.3"
  )
  echo "[beaker-eval] Installing LIBERO sim dependencies: ${libero_sim_deps[*]}"
  pip_install "${libero_sim_deps[@]}"

  echo "[beaker-eval] Installing LIBERO package (editable, no deps) from ${libero_dir}"
  pip_install --no-deps -e "${libero_dir}"

  "${PYTHON}" - <<'PY'
import libero
from libero.libero import benchmark
print("LIBERO import OK:", libero.__file__)
print("benchmark suites:", list(benchmark.get_benchmark_dict().keys())[:4], "...")
PY

  echo "[beaker-eval] LIBERO installed from ${libero_dir}"
}

beaker_check_weka
beaker_resolve_eval_paths
beaker_install_eval_deps

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/experiments/libero:${PYTHONPATH:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

RUN_TAG="${EVAL_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${FASTWAM_RUNS_ROOT}/eval/libero/${TASK}/${RUN_TAG}}"
mkdir -p "${EVAL_OUTPUT_DIR}"

HYDRA_ARGS=(
  "task=${TASK}"
  "ckpt=${EVAL_CKPT}"
  "paths=weka"
  "paths.weka_user=${USER_NAME:-yejink}"
  "MULTIRUN.num_gpus=${NUM_GPUS}"
  "EVALUATION.output_dir=${EVAL_OUTPUT_DIR}"
)

if [[ -n "${DATASET_STATS_PATH:-}" ]]; then
  HYDRA_ARGS+=("EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}")
fi

if [[ -n "${EVAL_HYDRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TOKENS=(${EVAL_HYDRA_OVERRIDES})
  HYDRA_ARGS+=("${EXTRA_TOKENS[@]}")
fi

echo "[beaker-eval] output_dir=${EVAL_OUTPUT_DIR}"
echo "[beaker-eval] >>> ${PYTHON} experiments/libero/run_libero_manager.py ${HYDRA_ARGS[*]}"
"${PYTHON}" experiments/libero/run_libero_manager.py "${HYDRA_ARGS[@]}"

echo "[beaker-eval] Done. Results: ${EVAL_OUTPUT_DIR}"
