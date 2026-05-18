#!/usr/bin/env bash
# Resume FastWAM training via the task-level `latest` symlink or an explicit resume= path.
#
# Symlink (updated on each save in trainer.py):
#   {runs_root}/{task}/latest -> {run_id}/checkpoints/state/step_XXXXXX
#   e.g. /weka/oe-training/yejink/runs/libero_uncond_2cam224_1e-4/latest
#
# Usage (sourced from run_train.sh):
#   beaker_apply_auto_resume TASK_NAME RUNS_ROOT

beaker_resolve_latest_state() {
  local link_path="$1"
  if [[ ! -L "${link_path}" && ! -e "${link_path}" ]]; then
    return 1
  fi
  local resolved
  resolved="$(readlink -f "${link_path}")"
  if [[ -d "${resolved}" && -f "${resolved}/trainer_state.json" ]]; then
    echo "${resolved}"
    return 0
  fi
  return 1
}

beaker_run_id_from_state_dir() {
  local state_dir="$1"
  basename "$(dirname "$(dirname "$(dirname "${state_dir}")")")"
}

beaker_hydra_get_resume_path() {
  local token
  for token in ${HYDRA_OVERRIDES:-}; do
    if [[ "${token}" == resume=* ]]; then
      echo "${token#resume=}"
      return 0
    fi
  done
  return 1
}

beaker_apply_explicit_resume() {
  local resume_path
  resume_path="$(beaker_hydra_get_resume_path || true)"
  if [[ -z "${resume_path}" ]]; then
    return 0
  fi

  local resolved
  resolved="$(
    PYTHON="${PYTHON:-python}"
    "${PYTHON}" - "${resume_path}" <<'PY'
import sys
from fastwam.utils.resume_paths import resolve_resume_path

output_dir, resume = resolve_resume_path(sys.argv[1])
print(output_dir)
print(resume)
PY
  )" || {
    echo "[beaker] ERROR: failed to resolve resume path: ${resume_path}" >&2
    exit 1
  }

  local output_dir resume_state
  output_dir="$(sed -n '1p' <<< "${resolved}")"
  resume_state="$(sed -n '2p' <<< "${resolved}")"
  export RUN_ID="$(basename "${output_dir}")"

  local -a new_overrides=()
  local token replaced_resume=0 replaced_output=0
  for token in ${HYDRA_OVERRIDES:-}; do
    if [[ "${token}" == resume=* ]]; then
      new_overrides+=("resume=${resume_state}")
      replaced_resume=1
      continue
    fi
    if [[ "${token}" == output_dir=* ]]; then
      new_overrides+=("output_dir=${output_dir}")
      replaced_output=1
      continue
    fi
    new_overrides+=("${token}")
  done
  if (( replaced_resume == 0 )); then
    new_overrides+=("resume=${resume_state}")
  fi
  if (( replaced_output == 0 )); then
    new_overrides+=("output_dir=${output_dir}")
  fi
  HYDRA_OVERRIDES="${new_overrides[*]}"
  export HYDRA_OVERRIDES

  echo "[beaker] explicit resume: run_id=${RUN_ID} output_dir=${output_dir} state=${resume_state}"
}

beaker_apply_auto_resume() {
  local task_name="${1:?task name required}"
  local runs_root="${2:?runs root required}"

  if [[ "${BEAKER_AUTO_RESUME:-1}" == "0" ]]; then
    echo "[beaker] auto-resume disabled (BEAKER_AUTO_RESUME=0)"
    return 0
  fi

  if beaker_hydra_get_resume_path >/dev/null; then
    beaker_apply_explicit_resume
    return 0
  fi

  local task_runs="${runs_root}/${task_name}"
  if [[ ! -d "${task_runs}" ]]; then
    echo "[beaker] no prior runs at ${task_runs}; starting fresh"
    return 0
  fi

  local state_dir
  state_dir="$(beaker_resolve_latest_state "${task_runs}/latest" || true)"

  if [[ -z "${state_dir}" ]]; then
    echo "[beaker] no ${task_runs}/latest symlink; starting fresh"
    return 0
  fi

  export RUN_ID="$(beaker_run_id_from_state_dir "${state_dir}")"
  local output_dir="${task_runs}/${RUN_ID}"
  HYDRA_OVERRIDES="${HYDRA_OVERRIDES} resume=${state_dir} output_dir=${output_dir}"
  export HYDRA_OVERRIDES

  echo "[beaker] auto-resume: run_id=${RUN_ID} latest=${task_runs}/latest state=${state_dir}"
}
