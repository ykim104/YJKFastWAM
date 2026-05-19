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
    # libegl1 / libgles2 / libglvnd-dev: enable headless EGL (MUJOCO_GL=egl).
    # libosmesa6 + libglapi-mesa: enable software rendering fallback (MUJOCO_GL=osmesa).
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      tmux \
      libglew-dev libosmesa6 libosmesa6-dev libglapi-mesa \
      libegl1 libgles2 libglvnd-dev libglx-mesa0 libgl1 libglib2.0-0 \
      patchelf \
      >/dev/null || true
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
      # Pin uv to the gantry job venv so editable installs land on this PYTHON's sys.path.
      uv pip install --python "${PYTHON}" "$@"
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

  # PyTorch 2.6 changed torch.load default to weights_only=True. LIBERO's
  # init_states are pickled numpy arrays; force weights_only=False on those calls.
  local libero_benchmark_init="${libero_dir}/libero/libero/benchmark/__init__.py"
  if [[ -f "${libero_benchmark_init}" ]]; then
    if ! grep -q "weights_only=False" "${libero_benchmark_init}"; then
      echo "[beaker-eval] Patching LIBERO benchmark torch.load -> weights_only=False"
      sed -i \
        -e 's/torch\.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
        -e 's/torch\.load(\(self\.[a-zA-Z_]\+_path\))/torch.load(\1, weights_only=False)/g' \
        "${libero_benchmark_init}"
    fi
  fi

  # Minimal runtime deps for OffScreenRenderEnv. Skip LIBERO's requirements.txt
  # because it pins old numpy/hydra/thop that conflict with the training venv.
  local libero_sim_deps=(
    "robosuite==1.4.0"
    "bddl==1.0.1"
    "future==0.18.2"          # required by bddl
    "gym==0.25.2"
    "cloudpickle==2.1.0"
    "easydict==1.9"
    "opencv-python-headless>=4.6.0.66"
    "matplotlib>=3.5.3"
    "h5py"                    # LIBERO reads task init states from .hdf5
  )
  echo "[beaker-eval] Installing LIBERO sim dependencies: ${libero_sim_deps[*]}"
  pip_install "${libero_sim_deps[@]}"

  echo "[beaker-eval] Installing LIBERO package (editable, no deps) from ${libero_dir}"
  pip_install --no-deps -e "${libero_dir}"

  # Always add the LIBERO repo to PYTHONPATH: uv's PEP 660 editable install for
  # LIBERO's old setup.py doesn't reliably register the package on sys.path.
  export LIBERO_REPO_DIR="${libero_dir}"
  export PYTHONPATH="${libero_dir}:${PYTHONPATH:-}"

  # LIBERO's libero/libero/__init__.py asks for an interactive path on first
  # import. Pre-write the config to a writable LIBERO_CONFIG_PATH and skip it.
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/libero_config}"
  mkdir -p "${LIBERO_CONFIG_PATH}"
  "${PYTHON}" - <<PY
import os, yaml
libero_root = os.path.join(os.environ["LIBERO_REPO_DIR"], "libero", "libero")
cfg = {
    "benchmark_root": libero_root,
    "bddl_files":   os.path.join(libero_root, "bddl_files"),
    "init_states":  os.path.join(libero_root, "init_files"),
    "datasets":     os.path.normpath(os.path.join(libero_root, "..", "datasets")),
    "assets":       os.path.join(libero_root, "assets"),
}
cfg_dir = os.environ["LIBERO_CONFIG_PATH"]
os.makedirs(cfg_dir, exist_ok=True)
with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
    yaml.dump(cfg, f)
print("[beaker-eval] wrote LIBERO config:", os.path.join(cfg_dir, "config.yaml"))
PY

  echo "[beaker-eval] sys.path / VIRTUAL_ENV check:"
  "${PYTHON}" - <<PY
import sys, os
print("python:", sys.executable)
print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
print("LIBERO_CONFIG_PATH:", os.environ.get("LIBERO_CONFIG_PATH"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("site-packages entries:")
for p in sys.path:
    if "site-packages" in p:
        print(" ", p)
PY

  "${PYTHON}" -c "import libero; from libero.libero import benchmark; print('LIBERO OK:', libero.__file__); print('suites:', list(benchmark.get_benchmark_dict().keys())[:4])"

  echo "[beaker-eval] LIBERO installed from ${libero_dir}"
}

beaker_check_weka
beaker_resolve_eval_paths
beaker_install_eval_deps

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/experiments/libero:${PYTHONPATH:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Probe robosuite's GL context (the one LIBERO actually uses). robosuite's
# EGL path needs EGL_EXT_platform_device which Mesa's libEGL doesn't support;
# we fall back to OSMesa software rendering when that's the case.
beaker_probe_render_backend() {
  local probe_rc=0
  MUJOCO_GL="${MUJOCO_GL}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}" \
    "${PYTHON}" - <<'PY' || probe_rc=$?
import os, sys
backend = os.environ.get("MUJOCO_GL", "egl")
print(f"[beaker-eval] render probe: backend={backend}")
try:
    if backend == "egl":
        from robosuite.renderers.context.egl_context import EGLGLContext as Ctx
        ctx = Ctx(max_width=64, max_height=64, device_id=0)
    elif backend == "osmesa":
        from robosuite.renderers.context.osmesa_context import OSMesaGLContext as Ctx
        ctx = Ctx(max_width=64, max_height=64, device_id=0)
    else:
        raise ValueError(f"unsupported backend {backend}")
    ctx.free()
except Exception as e:
    print(f"[beaker-eval] render probe failed ({backend}): {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(2)
print(f"[beaker-eval] render probe OK: {backend}")
PY
  if [[ ${probe_rc} -ne 0 && "${MUJOCO_GL}" == "egl" ]]; then
    echo "[beaker-eval] Falling back to MUJOCO_GL=osmesa (software rendering)"
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa "${PYTHON}" - <<'PY'
from robosuite.renderers.context.osmesa_context import OSMesaGLContext
ctx = OSMesaGLContext(max_width=64, max_height=64, device_id=0); ctx.free()
print("[beaker-eval] render probe OK (osmesa)")
PY
  fi
}
beaker_probe_render_backend

# run_libero_parallel_test.sh launches each task in a tmux pane that `source ~/.bashrc`s.
# Beaker images don't have a useful ~/.bashrc, so the gantry venv + our PYTHONPATH /
# LIBERO_CONFIG_PATH are lost. Write a bashrc that re-establishes the env in every pane.
beaker_write_bashrc() {
  local bashrc="${HOME:-/root}/.bashrc"
  echo "[beaker-eval] Writing ${bashrc} for tmux panes (gantry venv + LIBERO env)"
  cat > "${bashrc}" <<EOF
# Auto-generated by FastWAM run_eval.sh
[[ \$- != *i* ]] || true
export VIRTUAL_ENV="${VIRTUAL_ENV:-/gantry-runtime/.venv}"
export PATH="\${VIRTUAL_ENV}/bin:\${PATH}"
export PYTHONPATH="${PYTHONPATH}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"
export LIBERO_REPO_DIR="${LIBERO_REPO_DIR}"
export MUJOCO_GL="${MUJOCO_GL}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH}"
export FASTWAM_DATA_ROOT="${FASTWAM_DATA_ROOT}"
export FASTWAM_CHECKPOINTS_ROOT="${FASTWAM_CHECKPOINTS_ROOT}"
export FASTWAM_RUNS_ROOT="${FASTWAM_RUNS_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FASTWAM_LEROBOT_VIDEO_BACKEND="${FASTWAM_LEROBOT_VIDEO_BACKEND:-pyav}"
EOF
  echo "[beaker-eval] bashrc head:"
  head -20 "${bashrc}" | sed 's/^/  /'
}
beaker_write_bashrc

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

beaker_dump_task_logs_on_fail() {
  local rc=$?
  local task_logs_dir="${EVAL_OUTPUT_DIR}/task_logs"
  if [[ ${rc} -ne 0 && -d "${task_logs_dir}" ]]; then
    echo "[beaker-eval] === Eval failed (rc=${rc}); dumping task logs ===" >&2
    for f in "${task_logs_dir}"/*.log; do
      [[ -f "$f" ]] || continue
      echo "[beaker-eval] ----- ${f} -----" >&2
      tail -n 200 "$f" >&2 || true
    done
    local failed_file="${EVAL_OUTPUT_DIR}/failed_tasks.txt"
    [[ -f "${failed_file}" ]] && { echo "[beaker-eval] failed_tasks:" >&2; cat "${failed_file}" >&2; }
  fi
  return ${rc}
}
trap beaker_dump_task_logs_on_fail EXIT

"${PYTHON}" experiments/libero/run_libero_manager.py "${HYDRA_ARGS[@]}"

echo "[beaker-eval] Done. Results: ${EVAL_OUTPUT_DIR}"
