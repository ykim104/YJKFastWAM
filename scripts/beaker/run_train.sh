#!/usr/bin/env bash
# Runs inside a Beaker container. Invoked by launch_train.py / gantry.
set -euo pipefail

beaker_on_err() {
  echo "[beaker] ERROR: exit $? at ${BASH_SOURCE[1]}:${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
}
trap beaker_on_err ERR

NUM_GPUS="${NUM_GPUS:-8}"
TASK="${TASK:?Set TASK (e.g. libero_triple_2cam224_1e-4)}"
PRECOMPUTE_TEXT="${PRECOMPUTE_TEXT:-0}"

# Gantry clones the repo into the job cwd; only cd when CODE_DIR is set and exists (raw Beaker on Weka).
if [[ -n "${CODE_DIR:-}" ]]; then
  if [[ ! -d "${CODE_DIR}" ]]; then
    echo "[beaker] ERROR: CODE_DIR=${CODE_DIR} does not exist." >&2
    echo "[beaker] For gantry jobs, omit CODE_DIR and use the cloned checkout at $(pwd)." >&2
    exit 1
  fi
  cd "${CODE_DIR}"
else
  echo "[beaker] Using gantry/git checkout: $(pwd)"
fi

# shellcheck source=setup_job_env.sh
_BEAKER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup_job_env.sh
source "${_BEAKER_SCRIPT_DIR}/setup_job_env.sh"
# shellcheck source=beaker_auto_resume.sh
source "${_BEAKER_SCRIPT_DIR}/beaker_auto_resume.sh"

WEKA_ROOT="${WEKA_ROOT:-/weka/oe-training/${USER_NAME:-yejink}}"
export FASTWAM_DATA_ROOT="${FASTWAM_DATA_ROOT:-${WEKA_ROOT}/data}"
export FASTWAM_CHECKPOINTS_ROOT="${FASTWAM_CHECKPOINTS_ROOT:-${WEKA_ROOT}/checkpoints}"
export FASTWAM_RUNS_ROOT="${FASTWAM_RUNS_ROOT:-${WEKA_ROOT}/runs}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_CHECKPOINTS_ROOT}}"
export HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-paths=weka paths.weka_user=${USER_NAME:-yejink}}"

# Gantry clones the repo (no CODE_DIR). Enable wandb by default for those jobs.
if [[ -z "${CODE_DIR:-}" ]] && [[ "${BEAKER_WANDB:-1}" != "0" ]]; then
  if [[ "${HYDRA_OVERRIDES}" != *"wandb.enabled"* ]]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES} wandb.enabled=true"
    export HYDRA_OVERRIDES
  fi
  export WANDB_MODE="${WANDB_MODE:-online}"
fi

echo "[beaker] data=${FASTWAM_DATA_ROOT} checkpoints=${FASTWAM_CHECKPOINTS_ROOT} runs=${FASTWAM_RUNS_ROOT}"

beaker_check_weka() {
  if [[ ! -d "/weka/oe-training" ]]; then
    echo "[beaker] ERROR: Weka not mounted at /weka/oe-training (is --weka oe-training-default:/weka/oe-training set?)" >&2
    ls -la /weka 2>&1 || true
    exit 1
  fi
  echo "[beaker] Weka mount OK: /weka/oe-training"
}

# ~48GB GPUs (L40/L40S) cannot run libero defaults (batch_size=16, no grad checkpointing).
# Triple co-denoising on ~80GB (H100/A100-80G) also needs smaller batch + grad checkpointing.
# BEAKER_LOW_VRAM=1 forces overrides; =0 disables; auto (default) detects from nvidia-smi.
beaker_is_triple_task() {
  [[ "${TASK}" == *triple* ]]
}

beaker_apply_memory_overrides() {
  local reason="${1:-}"
  local batch_size="${2:-8}"
  local grad_accum="${3:-2}"
  # batch_size + grad checkpointing are forced for memory safety (Hydra last-wins).
  # gradient_accumulation_steps is left to the user when set explicitly, so the
  # effective batch can be tuned (e.g. grad_accum=16 -> global batch 1024 on 8 GPUs).
  local ga_override="gradient_accumulation_steps=${grad_accum}"
  if [[ " ${HYDRA_OVERRIDES} " == *" gradient_accumulation_steps="* ]]; then
    ga_override=""
    echo "[beaker] keeping user gradient_accumulation_steps (memory default ${grad_accum} skipped)"
  fi
  HYDRA_OVERRIDES="${HYDRA_OVERRIDES} batch_size=${batch_size} ${ga_override} model.mot_checkpoint_mixed_attn=true"
  echo "[beaker] memory Hydra overrides (${reason}): batch_size=${batch_size} grad_accum=${ga_override:-<user>} mot_checkpoint=true"
}

beaker_query_gpu_mem_mib() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local mem_mib smi_rc=0
  # Do not use bare pipelines here: set -o pipefail + failed nvidia-smi exits the whole job silently.
  set +o pipefail
  mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ,')"
  smi_rc=$?
  set -o pipefail
  if (( smi_rc != 0 )); then
    return 1
  fi
  if [[ ! "${mem_mib}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  echo "${mem_mib}"
}

beaker_maybe_memory_overrides() {
  case "${BEAKER_LOW_VRAM:-auto}" in
    0|false|no)
      if beaker_is_triple_task; then
        local mem_mib=""
        if mem_mib="$(beaker_query_gpu_mem_mib)"; then
          # 75000–90000 MiB ≈ 80GB class (H100-80G, A100-80G); skip 96GB+ where batch 16 may fit.
          if (( mem_mib > 49152 && mem_mib < 90000 )); then
            beaker_apply_memory_overrides "triple task on ${mem_mib} MiB GPU" 8 2
          fi
        fi
      fi
      return 0
      ;;
    1|true|yes)
      if beaker_is_triple_task; then
        beaker_apply_memory_overrides "BEAKER_LOW_VRAM forced (triple)" 4 4
      else
        beaker_apply_memory_overrides "BEAKER_LOW_VRAM forced" 8 2
      fi
      return 0
      ;;
  esac

  local mem_mib=""
  if ! mem_mib="$(beaker_query_gpu_mem_mib)"; then
    echo "[beaker] nvidia-smi unavailable; skipping memory overrides"
    return 0
  fi

  # 49152 MiB ≈ 48GB class (L40/L40S ~45–48 GiB). Triple needs a smaller batch than two-mod on these GPUs.
  if (( mem_mib <= 49152 )); then
    if beaker_is_triple_task; then
      beaker_apply_memory_overrides "triple on ${mem_mib} MiB GPU (<=48GB)" 4 4
    else
      beaker_apply_memory_overrides "GPU ${mem_mib} MiB (<=48GB)" 8 2
    fi
  elif beaker_is_triple_task && (( mem_mib < 90000 )); then
    beaker_apply_memory_overrides "triple task on ${mem_mib} MiB GPU" 8 2
  else
    echo "[beaker] GPU ${mem_mib} MiB — using task defaults (batch_size from Hydra)"
  fi
}

echo "[beaker] Detecting GPU memory for overrides (BEAKER_LOW_VRAM=${BEAKER_LOW_VRAM:-auto}, TASK=${TASK})..."
beaker_maybe_memory_overrides

# Gantry CLI flags (e.g. --allow-dirty) must not reach train.py / Hydra.
beaker_sanitize_hydra_overrides() {
  local -a gantry_only_flags=(
    --allow-dirty
    --no-priority
    --no-venv
    --no-python-cache
  )
  local -a cleaned=()
  local token
  for token in ${HYDRA_OVERRIDES:-}; do
    local drop=0
    for flag in "${gantry_only_flags[@]}"; do
      if [[ "${token}" == "${flag}" ]]; then
        echo "[beaker] Dropping gantry-only flag from HYDRA_OVERRIDES: ${token}" >&2
        drop=1
        break
      fi
    done
    if (( drop == 0 )); then
      cleaned+=("${token}")
    fi
  done
  HYDRA_OVERRIDES="${cleaned[*]}"
  export HYDRA_OVERRIDES
}

beaker_sanitize_hydra_overrides
echo "[beaker] HYDRA_OVERRIDES=${HYDRA_OVERRIDES}"

# Multi-node (Beaker leader selection + host networking).
if [[ -n "${BEAKER_REPLICA_COUNT:-}" && "${BEAKER_REPLICA_COUNT}" -gt 1 ]]; then
  export NNODES="${BEAKER_REPLICA_COUNT}"
  export NODE_RANK="${BEAKER_REPLICA_RANK}"
  export MASTER_ADDR="${BEAKER_LEADER_REPLICA_HOSTNAME}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  echo "[beaker] multi-node NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR}"
fi

beaker_check_weka
echo "[beaker] Creating output dirs..."
mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}" "${FASTWAM_RUNS_ROOT}"

_beaker_ensure_pip() {
  if "${PYTHON}" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  echo "[beaker] Bootstrapping pip in gantry venv..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON}" pip setuptools wheel
    return 0
  fi
  local get_pip="/tmp/get-pip.py"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${get_pip}"
  "${PYTHON}" "${get_pip}"
}

_beaker_install_deps() {
  echo "[beaker] Installing fastwam + training deps into ${PYTHON}..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON}" -e . --torch-backend cu128
    return 0
  fi
  _beaker_ensure_pip
  "${PYTHON}" -m pip install -U pip
  "${PYTHON}" -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
}

_beaker_ensure_nvcc() {
  if beaker_setup_cuda; then
    return 0
  fi
  beaker_try_apt_nvcc || true
  beaker_setup_cuda
}

echo "[beaker] Checking Python deps..."
if ! "${PYTHON}" -c "import accelerate, fastwam, torch" 2>/dev/null; then
  if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
    echo "[beaker] WARNING: SKIP_PIP_INSTALL=1 but deps missing; installing anyway." >&2
  fi
  _beaker_install_deps
fi

echo "[beaker] Configuring CUDA for DeepSpeed..."
# DeepSpeed calls nvcc -V at import; pip nvidia-cuda-nvcc-cu12 has no nvcc binary.
_beaker_ensure_nvcc || {
  echo "[beaker] ERROR: could not configure CUDA_HOME for DeepSpeed." >&2
  exit 1
}
"${PYTHON}" -c "import accelerate, fastwam, torch; print('[beaker] deps OK')"
if ! "${PYTHON}" -c "import deepspeed; print('[beaker] deepspeed OK:', deepspeed.__version__)"; then
  echo "[beaker] ERROR: deepspeed import failed." >&2
  exit 1
fi

if [[ "${PRECOMPUTE_TEXT}" == "1" ]]; then
  echo "[beaker] Precomputing T5 text embeddings for task=${TASK}..."
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
    scripts/precompute_text_embeds.py "task=${TASK}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Reuse the latest Weka checkpoint for this task (gantry / Beaker preemption restarts).
beaker_apply_auto_resume "${TASK}" "${FASTWAM_RUNS_ROOT}"

if [[ -n "${FASTWAM_RUN_ID:-}" && -z "${RUN_ID:-}" ]]; then
  export RUN_ID="${FASTWAM_RUN_ID}"
  echo "[beaker] Using pinned run id: RUN_ID=${RUN_ID}"
fi

echo "[beaker] Starting training: NUM_GPUS=${NUM_GPUS} TASK=${TASK} RUN_ID=${RUN_ID:-<new>}"
# shellcheck disable=SC2086
bash scripts/train_zero1.sh "${NUM_GPUS}" "task=${TASK}" ${HYDRA_OVERRIDES:-}
