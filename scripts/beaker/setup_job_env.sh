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

# Beaker images often export CUDA_HOME without a usable nvcc.
if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  unset CUDA_HOME
fi

beaker_setup_cuda() {
  local candidate nvcc_path pkg_home

  # Image often sets CUDA_HOME=/usr/local/cuda without nvcc; DeepSpeed trusts the env var.
  if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "[beaker] Ignoring CUDA_HOME=${CUDA_HOME} (no bin/nvcc)"
    unset CUDA_HOME
  fi

  pkg_home="$("${PYTHON}" -c "
from pathlib import Path
try:
    import nvidia.cuda_nvcc as m
    root = Path(m.__file__).resolve().parent
    if (root / 'bin' / 'nvcc').is_file():
        print(root)
        raise SystemExit(0)
except ImportError:
    pass
import sysconfig
for nvcc in Path(sysconfig.get_paths()['purelib']).rglob('bin/nvcc'):
    if nvcc.is_file():
        print(nvcc.resolve().parent.parent)
        break
" 2>/dev/null || true)"
  if [[ -n "${pkg_home}" && -x "${pkg_home}/bin/nvcc" ]]; then
    export CUDA_HOME="${pkg_home}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "[beaker] CUDA_HOME=${CUDA_HOME} (pip nvidia-cuda-nvcc)"
    return 0
  fi

  local venv_root="${VIRTUAL_ENV:-$(dirname "$(dirname "${PYTHON}")")}"
  local venv_nvcc="${venv_root}/bin/nvcc"
  if [[ -x "${venv_nvcc}" ]]; then
    export CUDA_HOME="$(cd "$(dirname "${venv_nvcc}")/.." && pwd)"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "[beaker] CUDA_HOME=${CUDA_HOME} (venv bin/nvcc)"
    return 0
  fi

  if nvcc_path="$(command -v nvcc 2>/dev/null)"; then
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

  echo "[beaker] WARNING: nvcc not found; DeepSpeed import may fail." >&2
  return 1
}
