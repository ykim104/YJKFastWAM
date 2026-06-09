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
# RoboTwin uses curobo's classic API (curobo.wrap.*, pybind CUDA ext). curobo main
# was restructured to curobo/_src/* + cuda.core runtime compilation and dropped
# curobo.wrap, so we pin a compatible release tag.
CUROBO_REF="${CUROBO_REF:-v0.7.6}"
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

# Editable install via REAL pip in setuptools "compat" mode. uv's editable
# (strict PEP 660) only exposed the top-level `curobo` (hiding curobo.wrap) and
# its registered finder even shadows a PYTHONPATH=src fallback. pip's compat mode
# writes a .pth that adds src/ so every submodule resolves, and runs build_ext
# in-place (compiling the CUDA .so into the source tree, cached on Weka).
rt_ensure_pip() {
  if "${PYTHON}" -m pip --version >/dev/null 2>&1; then return 0; fi
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON}" pip setuptools wheel
  else
    "${PYTHON}" -m ensurepip --upgrade
  fi
}
rt_pip_editable_compat() {
  local dir="$1"
  rt_ensure_pip
  # Drop any broken prior editable registration (e.g. uv's) so its finder can't
  # shadow this install.
  "${PYTHON}" -m pip uninstall -y nvidia-curobo >/dev/null 2>&1 || true
  "${PYTHON}" -m pip install --no-cache-dir --no-build-isolation \
    --config-settings editable_mode=compat -e "${dir}"
}

rt_build_curobo() {
  local curobo_dir="${ROBOTWIN_ENV_DIR}/envs/curobo"
  # (Re)clone at the pinned tag if the expected v0.7.x src/ layout isn't present.
  # A prior run may have cloned `main` (root curobo/_src layout, no curobo.wrap).
  if [[ ! -d "${curobo_dir}/src/curobo/wrap" ]]; then
    echo "[rt-setup] Cloning curobo ${CUROBO_REF} into ${curobo_dir}"
    rm -rf "${curobo_dir}"
    git clone --depth 1 --branch "${CUROBO_REF}" "${CUROBO_GIT}" "${curobo_dir}"
  else
    echo "[rt-setup] Reusing existing curobo (${CUROBO_REF}) at ${curobo_dir}"
  fi

  # Cover A100 (8.0), L40/L40S (8.9), H100 (9.0). Building for several arches is
  # slower but makes the cached .so usable across the eval clusters.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"

  echo "[rt-setup] Installing curobo (editable compat; compiles CUDA ext into Weka tree)..."
  # warp-lang is curobo's runtime dep; install it explicitly so eval jobs have it.
  rt_pip "warp-lang" || true
  rt_pip_editable_compat "${curobo_dir}"

  # Belt-and-suspenders: force the in-place extension build so the .so land in the
  # src/ tree on Weka (cached) even if the editable install skipped build_ext.
  echo "[rt-setup] Compiling curobo CUDA extensions in-place (nvcc)..."
  ( cd "${curobo_dir}" && "${PYTHON}" setup.py build_ext --inplace ) || \
    echo "[rt-setup] WARNING: build_ext --inplace returned nonzero (check logs)" >&2

  echo "[rt-setup] Verifying curobo import..."
  "${PYTHON}" -c "import curobo; from curobo.wrap.reacher.motion_gen import MotionGen; print('curobo OK:', curobo.__file__)" || {
    echo "[rt-setup] editable import failing; verifying via PYTHONPATH=src fallback..." >&2
    PYTHONPATH="${curobo_dir}/src:${PYTHONPATH:-}" "${PYTHON}" -c \
      "import curobo; from curobo.wrap.reacher.motion_gen import MotionGen; print('curobo OK (src):', curobo.__file__)"
  }

  rt_write_curobo_requirements "${curobo_dir}"
}

# The eval image has no nvcc, so it can't `pip install -e` curobo (build_ext fails)
# and curobo's runtime deps (yourdfpy, trimesh[easy], ...) only live in this setup
# venv. Persist curobo's declared deps to Weka so eval installs them WITHOUT
# building, then imports curobo from the prebuilt src/ tree via PYTHONPATH.
rt_write_curobo_requirements() {
  local curobo_dir="$1"
  local out="${ROBOTWIN_ENV_DIR}/envs/curobo_requirements.txt"
  local egg_req
  egg_req="$(find "${curobo_dir}" -path "*nvidia_curobo.egg-info/requires.txt" 2>/dev/null | head -1)"
  {
    echo "warp-lang"
    if [[ -n "${egg_req}" && -f "${egg_req}" ]]; then
      # egg-info requires.txt lists base deps before the first [extras] section.
      # Drop torch (already in the cu128 venv; reinstalling would break it).
      awk 'BEGIN{p=1} /^\[/{p=0} p && NF{print}' "${egg_req}" | grep -viE "^torch([^A-Za-z0-9_]|$)"
    else
      echo "[rt-setup] WARNING: egg-info requires.txt not found; writing hardcoded deps" >&2
      printf '%s\n' yourdfpy "trimesh[easy]" numpy-quaternion networkx scipy pyyaml importlib_resources
    fi
  } | sort -u > "${out}"
  echo "[rt-setup] wrote curobo deps -> ${out}"
  sed 's/^/  /' "${out}"
}

rt_install_cuda_toolkit
rt_build_curobo

echo "[rt-setup] Done. Staged RoboTwin env ready at ${ROBOTWIN_ENV_DIR}"
echo "[rt-setup] curobo built at ${ROBOTWIN_ENV_DIR}/envs/curobo (cached on Weka)"
