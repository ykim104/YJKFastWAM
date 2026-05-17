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
# Training only (ZeRO); avoid pulling inference Triton kernels at import when possible.
export DS_INFERENCE="${DS_INFERENCE:-0}"

beaker_setup_cuda() {
  local candidate nvcc_path pkg_home cuda_home

  for candidate in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      echo "[beaker] CUDA_HOME=${CUDA_HOME}"
      return 0
    fi
  done

  if nvcc_path="$(command -v nvcc 2>/dev/null)"; then
    export CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "[beaker] CUDA_HOME=${CUDA_HOME}"
    return 0
  fi

  pkg_home="$("${PYTHON}" -c "
import sysconfig
from pathlib import Path
for nvcc in Path(sysconfig.get_paths()['purelib']).rglob('nvcc'):
    if nvcc.is_file():
        print(nvcc.resolve().parent.parent)
        break
" 2>/dev/null || true)"
  if [[ -n "${pkg_home}" && -x "${pkg_home}/bin/nvcc" ]]; then
    export CUDA_HOME="${pkg_home}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "[beaker] CUDA_HOME=${CUDA_HOME}"
    return 0
  fi

  echo "[beaker] WARNING: nvcc not found; DeepSpeed import may fail." >&2
  return 1
}

# Only probe CUDA after deps are installed (BEAKER_SETUP_CUDA=1 from run_train.sh).
if [[ "${BEAKER_SETUP_CUDA:-0}" == "1" ]]; then
  beaker_setup_cuda || true
fi
