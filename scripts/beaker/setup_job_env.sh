# shellcheck shell=bash
# Shared Beaker/Gantry job environment: Python from gantry venv, CUDA for DeepSpeed.

# Prefer Gantry/job venv over image conda (often Python 3.13 + wrong accelerate).
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

export PYTHON="${PYTHON:-$(command -v python)}"
echo "[beaker] PYTHON=${PYTHON} ($("${PYTHON}" --version 2>&1))"

_resolve_cuda_home() {
  local candidate nvcc_path pkg_home
  for candidate in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  if nvcc_path="$(command -v nvcc 2>/dev/null)"; then
    echo "$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
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
    echo "${pkg_home}"
    return 0
  fi
  return 1
}

if cuda_home="$(_resolve_cuda_home)"; then
  export CUDA_HOME="${cuda_home}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  echo "[beaker] CUDA_HOME=${CUDA_HOME}"
else
  echo "[beaker] WARNING: nvcc not found; DeepSpeed import may fail." >&2
fi

export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
