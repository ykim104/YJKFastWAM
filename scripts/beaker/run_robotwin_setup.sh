#!/usr/bin/env bash
# One-shot RoboTwin environment build for Beaker eval (run inside gantry).
#
# Builds curobo *editable into the Weka-staged RoboTwin* using the SAME image the
# eval job uses, so the compiled CUDA extensions are ABI-compatible and cached on
# Weka. After this runs once, every eval job re-registers curobo without rebuild.
#
# It does NOT download assets/task_config — stage those plain files yourself
# (see README / launch_robotwin_setup_gantry.sh header).
#
# Required env (set by launch_robotwin_setup_gantry.sh):
#   ROBOTWIN_ENV_DIR     Weka RoboTwin dir (must already have assets/ + task_config/)
# Optional env:
#   CUROBO_GIT           curobo repo (default: https://github.com/NVlabs/curobo.git)
#   CUDA_TOOLKIT_VERSION apt cuda toolkit pkg suffix (default: 12-8, matches torch cu128)
set -euo pipefail

beaker_on_err() {
  echo "[rt-setup] ERROR: exit $? at ${BASH_SOURCE[1]}:${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
}
trap beaker_on_err ERR

if [[ -n "${CODE_DIR:-}" && -d "${CODE_DIR}" ]]; then
  cd "${CODE_DIR}"
fi

_BEAKER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup_job_env.sh
source "${_BEAKER_SCRIPT_DIR}/setup_job_env.sh"

WEKA_ROOT="${WEKA_ROOT:-/weka/oe-training/${USER_NAME:-yejink}}"
ROBOTWIN_ENV_DIR="${ROBOTWIN_ENV_DIR:-${WEKA_ROOT}/robotwin/RoboTwin}"
CUROBO_GIT="${CUROBO_GIT:-https://github.com/NVlabs/curobo.git}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12-8}"

if [[ ! -d "/weka/oe-training" ]]; then
  echo "[rt-setup] ERROR: Weka not mounted at /weka/oe-training" >&2
  exit 1
fi
echo "[rt-setup] ROBOTWIN_ENV_DIR=${ROBOTWIN_ENV_DIR}"
mkdir -p "${ROBOTWIN_ENV_DIR}/envs"

for sub in assets task_config; do
  if [[ ! -e "${ROBOTWIN_ENV_DIR}/${sub}" ]]; then
    echo "[rt-setup] WARNING: ${ROBOTWIN_ENV_DIR}/${sub} missing (stage it before eval)" >&2
  fi
done

# A real nvcc (matching torch's CUDA) is required to compile curobo's extensions;
# the train env only ships an nvcc *stub*. Install the CUDA toolkit from NVIDIA's apt repo.
rt_install_cuda_toolkit() {
  if command -v nvcc >/dev/null 2>&1 && nvcc -V 2>/dev/null | grep -q "release"; then
    if nvcc -V 2>/dev/null | grep -q "compilation tools"; then
      # Reject the fastwam stub (it can't compile).
      if [[ "$(command -v nvcc)" != *"fastwam-cuda-stub"* ]]; then
        echo "[rt-setup] Found real nvcc: $(command -v nvcc)"
        export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
        return 0
      fi
    fi
  fi
  echo "[rt-setup] Installing CUDA toolkit ${CUDA_TOOLKIT_VERSION} for curobo build..."
  local distro="ubuntu2204"
  if command -v lsb_release >/dev/null 2>&1; then
    case "$(lsb_release -rs 2>/dev/null)" in
      24.*) distro="ubuntu2404" ;;
      20.*) distro="ubuntu2004" ;;
      *)    distro="ubuntu2204" ;;
    esac
  fi
  apt-get update -qq || true
  apt-get install -y -qq wget ca-certificates gnupg || true
  local keyring="/tmp/cuda-keyring.deb"
  wget -qO "${keyring}" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i "${keyring}"
  apt-get update -qq
  apt-get install -y -qq "cuda-toolkit-${CUDA_TOOLKIT_VERSION}" || \
    apt-get install -y -qq "cuda-nvcc-${CUDA_TOOLKIT_VERSION}"
  local cuda_dir="/usr/local/cuda-${CUDA_TOOLKIT_VERSION/-/.}"
  if [[ -x "${cuda_dir}/bin/nvcc" ]]; then
    export CUDA_HOME="${cuda_dir}"
  elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
  export PATH="${CUDA_HOME}/bin:${PATH}"
  echo "[rt-setup] CUDA_HOME=${CUDA_HOME}"
  nvcc -V | tail -3
}

rt_pip() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON}" "$@"
  else
    "${PYTHON}" -m pip install --no-cache-dir "$@"
  fi
}

# Editable install in setuptools "compat" mode: unlike uv's default PEP 660
# editable (which hid curobo.wrap and skipped the in-place ext build), compat mode
# runs build_ext in-place (compiling the CUDA .so into the source tree, cached on
# Weka) and adds the real src/ dir to the path so all submodules import.
rt_pip_editable_compat() {
  local dir="$1"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON}" -e "${dir}" --no-build-isolation \
      --config-setting editable_mode=compat
  else
    "${PYTHON}" -m pip install --no-cache-dir -e "${dir}" --no-build-isolation \
      --config-settings editable_mode=compat
  fi
}

rt_build_curobo() {
  local curobo_dir="${ROBOTWIN_ENV_DIR}/envs/curobo"
  if [[ ! -f "${curobo_dir}/setup.py" && ! -f "${curobo_dir}/pyproject.toml" ]]; then
    echo "[rt-setup] Cloning curobo into ${curobo_dir}"
    rm -rf "${curobo_dir}"
    git clone "${CUROBO_GIT}" "${curobo_dir}"
  else
    echo "[rt-setup] Reusing existing curobo source at ${curobo_dir}"
  fi

  # Cover A100 (8.0), L40/L40S (8.9), H100 (9.0). Building for several arches is
  # slower but makes the cached .so usable across the eval clusters.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"

  echo "[rt-setup] Installing curobo (editable compat; compiles CUDA ext into Weka tree)..."
  # warp-lang is curobo's runtime dep; install it explicitly so eval jobs have it.
  rt_pip "warp-lang" || true
  rt_pip_editable_compat "${curobo_dir}"

  # Belt-and-suspenders: force the in-place extension build so the .so are present
  # in the source tree on Weka even if the editable install skipped build_ext.
  echo "[rt-setup] Compiling curobo CUDA extensions in-place (nvcc)..."
  ( cd "${curobo_dir}" && "${PYTHON}" setup.py build_ext --inplace ) || \
    echo "[rt-setup] WARNING: build_ext --inplace returned nonzero (check logs)" >&2

  echo "[rt-setup] Verifying curobo import..."
  "${PYTHON}" -c "import curobo; from curobo.wrap.reacher.motion_gen import MotionGen; print('curobo OK:', curobo.__file__)" || {
    echo "[rt-setup] editable import still failing; verifying via PYTHONPATH=src fallback..." >&2
    PYTHONPATH="${curobo_dir}/src:${PYTHONPATH:-}" "${PYTHON}" -c \
      "import curobo; from curobo.wrap.reacher.motion_gen import MotionGen; print('curobo OK (src):', curobo.__file__)"
  }
}

rt_install_cuda_toolkit
rt_build_curobo

echo "[rt-setup] Done. Staged RoboTwin env ready at ${ROBOTWIN_ENV_DIR}"
echo "[rt-setup] curobo built at ${ROBOTWIN_ENV_DIR}/envs/curobo (cached on Weka)"
