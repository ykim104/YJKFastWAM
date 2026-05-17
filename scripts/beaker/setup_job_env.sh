# shellcheck shell=bash
# Shared Beaker/Gantry job environment: Python from gantry venv, CUDA for DeepSpeed.

# Prefer Gantry/job venv over image conda (often Python 3.13 + wrong accelerate).
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

export PYTHON="${PYTHON:-$(command -v python)}"
echo "[beaker] PYTHON=${PYTHON} ($("${PYTHON}" --version 2>&1))"

# uv/gantry venv Python headers are not under /usr/include; Triton/DeepSpeed need this for JIT.
_py_include="$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_path('include'))")"
export C_INCLUDE_PATH="${_py_include}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="${_py_include}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export CPATH="${_py_include}${CPATH:+:${CPATH}}"
echo "[beaker] PYTHON_INCLUDE=${_py_include}"

export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export DS_INFERENCE="${DS_INFERENCE:-0}"

# torchcodec is installed but often missing libnppicc on Beaker images; pyav avoids per-frame warnings.
export FASTWAM_LEROBOT_VIDEO_BACKEND="${FASTWAM_LEROBOT_VIDEO_BACKEND:-pyav}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Beaker images often export CUDA_HOME without a usable nvcc.
if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  unset CUDA_HOME
fi

# nvidia-cuda-nvcc-cu12 pip wheel ships ptxas/headers only — NOT bin/nvcc (NVIDIA forum #221307).
beaker_install_nvcc_stub() {
  local cuda_stub cuda_ver major minor patch
  cuda_stub="${FASTWAM_CUDA_STUB:-/tmp/fastwam-cuda-stub}"
  cuda_ver="$("${PYTHON}" -c "import torch; print(torch.version.cuda or '12.8')")"
  major="${cuda_ver%%.*}"
  minor_patch="${cuda_ver#*.}"
  minor="${minor_patch%%.*}"
  patch="${minor_patch#*.}"
  patch="${patch%%.*}"
  mkdir -p "${cuda_stub}/bin"
  cat >"${cuda_stub}/bin/nvcc" <<EOF
#!/bin/sh
# Stub for DeepSpeed import (nvcc -V). ZeRO training does not compile CUDA ops here.
case "\$1" in
  -V|--version)
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Copyright (c) 2005-2025 NVIDIA Corporation"
    echo "Cuda compilation tools, release ${major}.${minor}, V${major}.${minor}.${patch:-0}"
    exit 0
    ;;
esac
echo "fastwam: nvcc stub cannot compile (no system CUDA toolkit in image)" >&2
exit 1
EOF
  chmod +x "${cuda_stub}/bin/nvcc"
  export CUDA_HOME="${cuda_stub}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  echo "[beaker] CUDA_HOME=${CUDA_HOME} (nvcc stub for DeepSpeed import; torch cuda=${cuda_ver})"
}

beaker_try_apt_nvcc() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 1
  fi
  echo "[beaker] Trying apt to install system nvcc..."
  apt-get update -qq
  # Package names vary by image; try common CUDA 12.x compiler packages.
  if apt-get install -y -qq cuda-nvcc-12-8 2>/dev/null \
    || apt-get install -y -qq cuda-compiler-12-8 2>/dev/null \
    || apt-get install -y -qq nvidia-cuda-toolkit 2>/dev/null; then
    return 0
  fi
  return 1
}

beaker_setup_cuda() {
  local candidate nvcc_path

  if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "[beaker] Ignoring CUDA_HOME=${CUDA_HOME} (no bin/nvcc)"
    unset CUDA_HOME
  fi

  if nvcc_path="$(command -v nvcc 2>/dev/null)" && [[ -x "${nvcc_path}" ]]; then
    export CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "[beaker] CUDA_HOME=${CUDA_HOME} (PATH nvcc)"
    return 0
  fi

  for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      echo "[beaker] CUDA_HOME=${CUDA_HOME}"
      return 0
    fi
  done

  if [[ "${BEAKER_ALLOW_NVCC_STUB:-1}" == "1" ]]; then
    beaker_install_nvcc_stub
    return 0
  fi

  echo "[beaker] WARNING: nvcc not found." >&2
  echo "[beaker] Note: pip package nvidia-cuda-nvcc-cu12 does not ship bin/nvcc." >&2
  return 1
}
