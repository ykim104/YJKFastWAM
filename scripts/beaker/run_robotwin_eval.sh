#!/usr/bin/env bash
# Runs RoboTwin full-benchmark eval inside a Beaker/Gantry container.
#
# Unlike LIBERO, the RoboTwin simulator environment is NOT in the git tree
# (task_config/, assets/, envs/curobo are .gitignore'd) and curobo/pytorch3d are
# CUDA-from-source builds that the training image's nvcc *stub* cannot compile.
# So this runner consumes a PREBUILT RoboTwin staged on Weka (ROBOTWIN_ENV_DIR)
# and only installs the light pure-python sim deps into the gantry venv.
#
# Required env (set by launch_robotwin_eval_gantry.sh):
#   TASK                 Hydra task (e.g. robotwin_uncond_3cam_384_1e-4)
#   CKPT                 weights .pt / run dir / task dir / .../latest / "latest"
#   ROBOTWIN_ENV_DIR     Weka dir with a fully-installed RoboTwin, must contain:
#                          assets/  task_config/  envs/curobo/
#   NUM_GPUS             GPUs for the parallel eval manager (default 8)
# Optional env:
#   ROBOTWIN_TASK_NAME   single task to eval (default: all tasks in _eval_step_limit.yml)
#   DATASET_STATS_PATH   dataset_stats.json (auto-detected from run dir if omitted)
#   EVAL_OUTPUT_DIR      results root (default: .../runs/eval/robotwin/${TASK}/<ts>)
#   EVAL_RUN_TAG         subdir name instead of timestamp
#   EVAL_HYDRA_OVERRIDES extra Hydra tokens forwarded to the manager
#   ROBOTWIN_PIP_REQS    requirements file inside ROBOTWIN_ENV_DIR to install
set -euo pipefail

beaker_on_err() {
  echo "[beaker-rt] ERROR: exit $? at ${BASH_SOURCE[1]}:${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
}
trap beaker_on_err ERR

TASK="${TASK:?Set TASK (e.g. robotwin_uncond_3cam_384_1e-4)}"
CKPT_INPUT="${CKPT:?Set CKPT (weights .pt, run dir, task dir, or .../latest)}"
NUM_GPUS="${NUM_GPUS:-8}"

if [[ -n "${CODE_DIR:-}" ]]; then
  if [[ ! -d "${CODE_DIR}" ]]; then
    echo "[beaker-rt] ERROR: CODE_DIR=${CODE_DIR} does not exist." >&2
    exit 1
  fi
  cd "${CODE_DIR}"
else
  echo "[beaker-rt] Using gantry/git checkout: $(pwd)"
fi

_BEAKER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup_job_env.sh
source "${_BEAKER_SCRIPT_DIR}/setup_job_env.sh"

REPO_ROOT="$(pwd)"
WEKA_ROOT="${WEKA_ROOT:-/weka/oe-training/${USER_NAME:-yejink}}"
export FASTWAM_DATA_ROOT="${FASTWAM_DATA_ROOT:-${WEKA_ROOT}/data}"
export FASTWAM_CHECKPOINTS_ROOT="${FASTWAM_CHECKPOINTS_ROOT:-${WEKA_ROOT}/checkpoints}"
export FASTWAM_RUNS_ROOT="${FASTWAM_RUNS_ROOT:-${WEKA_ROOT}/runs}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_CHECKPOINTS_ROOT}}"

# Default the staged RoboTwin env location on Weka.
ROBOTWIN_ENV_DIR="${ROBOTWIN_ENV_DIR:-${WEKA_ROOT}/robotwin/RoboTwin}"

beaker_check_weka() {
  if [[ ! -d "/weka/oe-training" ]]; then
    echo "[beaker-rt] ERROR: Weka not mounted at /weka/oe-training" >&2
    ls -la /weka 2>&1 || true
    exit 1
  fi
  echo "[beaker-rt] Weka mount OK: /weka/oe-training"
}

beaker_check_robotwin_env() {
  echo "[beaker-rt] ROBOTWIN_ENV_DIR=${ROBOTWIN_ENV_DIR}"
  if [[ ! -d "${ROBOTWIN_ENV_DIR}" ]]; then
    echo "[beaker-rt] ERROR: staged RoboTwin env not found at ROBOTWIN_ENV_DIR." >&2
    echo "[beaker-rt] Stage a fully-installed RoboTwin (assets/, task_config/, envs/curobo/) there." >&2
    exit 1
  fi
  local missing=0
  for sub in assets task_config envs/curobo; do
    if [[ ! -e "${ROBOTWIN_ENV_DIR}/${sub}" ]]; then
      echo "[beaker-rt] WARNING: missing ${ROBOTWIN_ENV_DIR}/${sub}" >&2
      missing=1
    fi
  done
  if (( missing )); then
    echo "[beaker-rt] WARNING: staged RoboTwin env is incomplete; eval may fail." >&2
  fi
}

# Symlink the gitignored sim env (assets, task_config, embodiment configs, prebuilt
# curobo) from the Weka staging dir into the gantry-cloned third_party/RoboTwin.
beaker_link_robotwin_env() {
  local rt="${REPO_ROOT}/third_party/RoboTwin"
  if [[ ! -d "${rt}" ]]; then
    echo "[beaker-rt] ERROR: ${rt} not found in checkout." >&2
    exit 1
  fi
  local sub
  for sub in assets task_config envs/curobo; do
    local src="${ROBOTWIN_ENV_DIR}/${sub}"
    local dst="${rt}/${sub}"
    [[ -e "${src}" ]] || { echo "[beaker-rt] skip link (no src): ${src}"; continue; }
    mkdir -p "$(dirname "${dst}")"
    # Replace any existing path/symlink with a link to the staged copy.
    rm -rf "${dst}"
    ln -sfn "${src}" "${dst}"
    echo "[beaker-rt] linked ${dst} -> ${src}"
  done
}

# Policy symlink RoboTwin's eval harness expects (policy/fastwam_policy).
beaker_link_policy() {
  local rt="${REPO_ROOT}/third_party/RoboTwin"
  ln -sfn "${REPO_ROOT}/experiments/robotwin/fastwam_policy" "${rt}/policy/fastwam_policy"
  echo "[beaker-rt] policy symlink: ${rt}/policy/fastwam_policy -> ${REPO_ROOT}/experiments/robotwin/fastwam_policy"
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
    "${PYTHON}" - "${ckpt_input}" <<'PY'
import sys
from fastwam.utils.resume_paths import resolve_eval_ckpt

ckpt, stats = resolve_eval_ckpt(sys.argv[1])
print(ckpt)
print(stats or "")
PY
  )" || {
    echo "[beaker-rt] ERROR: failed to resolve CKPT=${ckpt_input}" >&2
    exit 1
  }
  export EVAL_CKPT="$(sed -n '1p' <<< "${resolved}")"
  local auto_stats
  auto_stats="$(sed -n '2p' <<< "${resolved}")"
  if [[ -z "${DATASET_STATS_PATH:-}" && -n "${auto_stats}" ]]; then
    export DATASET_STATS_PATH="${auto_stats}"
  fi
  echo "[beaker-rt] ckpt=${EVAL_CKPT}"
  if [[ -n "${DATASET_STATS_PATH:-}" ]]; then
    echo "[beaker-rt] dataset_stats=${DATASET_STATS_PATH}"
  else
    echo "[beaker-rt] WARNING: dataset_stats not found; pass DATASET_STATS_PATH or EVALUATION.dataset_stats_path"
  fi
}

beaker_install_sim_deps() {
  if [[ "${SKIP_ROBOTWIN_INSTALL:-0}" == "1" ]]; then
    echo "[beaker-rt] SKIP_ROBOTWIN_INSTALL=1"
    return 0
  fi
  echo "[beaker-rt] Installing RoboTwin sim system + python deps..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    # SAPIEN renders via Vulkan (not GL); ffmpeg encodes eval videos.
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      libvulkan1 vulkan-tools libvulkan-dev \
      libegl1 libgles2 libglvnd-dev libgl1 libglib2.0-0 \
      libosmesa6 libgl1-mesa-glx libglew-dev \
      ffmpeg git \
      >/dev/null || true
  fi

  local pip_install
  if command -v uv >/dev/null 2>&1; then
    pip_install=(uv pip install --python "${PYTHON}")
  else
    "${PYTHON}" -m ensurepip --upgrade >/dev/null 2>&1 || true
    pip_install=("${PYTHON}" -m pip install --no-cache-dir)
  fi

  # sapien's __init__ imports pkg_resources, which setuptools REMOVED in v81.
  # The minimal uv venv has no setuptools, so install a pre-81 release that still
  # ships pkg_resources, and verify (fall back to a known-good pin if needed).
  "${pip_install[@]}" "setuptools<81" wheel || true
  if ! "${PYTHON}" -c "import pkg_resources" >/dev/null 2>&1; then
    echo "[beaker-rt] pkg_resources missing; pinning setuptools==70.2.0" >&2
    "${pip_install[@]}" "setuptools==70.2.0" || true
  fi

  # Prefer an exact requirements file staged on Weka so versions match the
  # prebuilt curobo / sapien used when collecting the assets.
  local reqs="${ROBOTWIN_PIP_REQS:-}"
  if [[ -z "${reqs}" && -f "${ROBOTWIN_ENV_DIR}/requirements.eval.txt" ]]; then
    reqs="${ROBOTWIN_ENV_DIR}/requirements.eval.txt"
  fi
  if [[ -n "${reqs}" && -f "${reqs}" ]]; then
    echo "[beaker-rt] Installing RoboTwin python deps from ${reqs}"
    "${pip_install[@]}" -r "${reqs}"
  else
    echo "[beaker-rt] Installing default RoboTwin python deps (no staged requirements file found)"
    # Pure-python / wheel deps only. curobo + pytorch3d come from the staged env
    # because they need a real CUDA toolkit to build (not present in this image).
    "${pip_install[@]}" \
      "sapien==3.0.0b1" \
      "mplib==0.2.1" \
      "toppra" \
      "transforms3d" \
      "trimesh" \
      "open3d" \
      "gymnasium" \
      "h5py" \
      "imageio[ffmpeg]" \
      "pydantic" \
      "opencv-python-headless>=4.6.0.66" \
      "matplotlib"
  fi

  # Register the prebuilt curobo from the staged env. Its CUDA ext were compiled
  # by run_robotwin_setup.sh into the Weka source tree, so this editable install
  # reuses the .so (no recompile, no nvcc needed) and pulls curobo's runtime deps
  # (warp-lang, etc.) into this fresh venv.
  local curobo_dir="${ROBOTWIN_ENV_DIR}/envs/curobo"
  if ! "${PYTHON}" -c "import curobo.wrap" >/dev/null 2>&1; then
    if [[ ! -d "${curobo_dir}/src/curobo/wrap" ]]; then
      echo "[beaker-rt] ERROR: prebuilt curobo (v0.7.x src/ layout) not found at ${curobo_dir}." >&2
      echo "[beaker-rt] Build it first: scripts/beaker/launch_robotwin_setup_gantry.sh" >&2
      exit 1
    fi
    # This image has no real nvcc, so we must NOT `pip install -e` curobo (build_ext
    # would fail). The CUDA .so were prebuilt on Weka by the setup job; we only
    # install curobo's runtime deps and import it from src/ via PYTHONPATH.
    echo "[beaker-rt] Installing curobo runtime deps (no build) + src/ on PYTHONPATH"
    local curobo_reqs="${ROBOTWIN_ENV_DIR}/envs/curobo_requirements.txt"
    if [[ -f "${curobo_reqs}" ]]; then
      "${pip_install[@]}" -r "${curobo_reqs}" || true
    else
      echo "[beaker-rt] WARNING: ${curobo_reqs} missing; installing hardcoded curobo deps" >&2
      "${pip_install[@]}" "warp-lang" yourdfpy "trimesh[easy]" numpy-quaternion networkx scipy pyyaml importlib_resources || true
    fi
    # curobo v0.7.6 pins warp-lang>=0.9.0 (open), but warp>=1.x no longer auto-exposes
    # the `warp.torch` submodule that curobo accesses as `wp.torch.device_from_torch`.
    # Pin a compatible warp release.
    "${pip_install[@]}" "warp-lang==${CUROBO_WARP_VERSION:-1.0.2}" || true
    # Drop any stale editable finder so PYTHONPATH=src wins, then expose the prebuilt tree.
    "${PYTHON}" -m pip uninstall -y nvidia-curobo >/dev/null 2>&1 || true
    export PYTHONPATH="${curobo_dir}/src:${PYTHONPATH:-}"
  fi

  "${PYTHON}" -c "import curobo.wrap; from curobo.wrap.reacher.motion_gen import MotionGen; print('[beaker-rt] curobo OK:', curobo.__file__)" || {
    echo "[beaker-rt] ERROR: curobo.wrap unimportable after dep install; re-run launch_robotwin_setup_gantry.sh." >&2
    exit 1
  }

  beaker_patch_sim_pkgs
}

# RoboTwin's _install.sh patches the installed sapien + mplib site-packages. Those
# packages are reinstalled fresh in this gantry venv, so re-apply the same edits.
beaker_patch_sim_pkgs() {
  local sapien_loc mplib_loc
  sapien_loc="$("${PYTHON}" -c "import os,sapien;print(os.path.dirname(sapien.__file__))" 2>/dev/null || true)"
  if [[ -n "${sapien_loc}" && -f "${sapien_loc}/wrapper/urdf_loader.py" ]]; then
    if ! grep -q 'encoding="utf-8"' "${sapien_loc}/wrapper/urdf_loader.py"; then
      echo "[beaker-rt] Patching sapien urdf_loader.py (utf-8 open)"
      sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${sapien_loc}/wrapper/urdf_loader.py" || true
    fi
  fi
  mplib_loc="$("${PYTHON}" -c "import os,mplib;print(os.path.dirname(mplib.__file__))" 2>/dev/null || true)"
  if [[ -n "${mplib_loc}" && -f "${mplib_loc}/planner.py" ]]; then
    echo "[beaker-rt] Patching mplib planner.py (drop 'or collide' screw-plan guard)"
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
      "${mplib_loc}/planner.py" || true
  fi
}

# SAPIEN uses Vulkan; Beaker NVIDIA containers mount the driver but may lack the
# Vulkan ICD manifest, so register NVIDIA's ICD like run_eval.sh does for EGL.
beaker_register_nvidia_vulkan() {
  # Prefer an existing NVIDIA ICD manifest (NVIDIA container runtime usually injects one).
  local icd existing
  for existing in \
    /usr/share/vulkan/icd.d/nvidia_icd.json \
    /etc/vulkan/icd.d/nvidia_icd.json; do
    if [[ -f "${existing}" ]]; then
      export VK_ICD_FILENAMES="${existing}"
      echo "[beaker-rt] Using existing Vulkan ICD: ${existing}"
      return 0
    fi
  done

  # Refresh the linker cache, then locate the NVIDIA GLVND/Vulkan driver lib.
  ldconfig 2>/dev/null || true
  local libnvvk
  libnvvk="$(ldconfig -p 2>/dev/null | awk '/libGLX_nvidia\.so\.0/ {print $NF; exit}')"
  if [[ -z "${libnvvk}" ]]; then
    local cand
    for cand in \
      /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 \
      /usr/lib64/libGLX_nvidia.so.0 \
      /usr/lib/libGLX_nvidia.so.0 \
      /usr/local/nvidia/lib64/libGLX_nvidia.so.0; do
      [[ -e "${cand}" ]] && { libnvvk="${cand}"; break; }
    done
  fi

  icd="/usr/share/vulkan/icd.d/nvidia_icd.json"
  mkdir -p /usr/share/vulkan/icd.d || true
  if [[ -n "${libnvvk}" ]]; then
    cat > "${icd}" <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD" : { "library_path" : "${libnvvk}", "api_version" : "1.3.0" }
}
EOF
    export VK_ICD_FILENAMES="${icd}"
    echo "[beaker-rt] Registered NVIDIA Vulkan ICD: ${icd} -> ${libnvvk}"
  else
    echo "[beaker-rt] WARNING: NVIDIA Vulkan lib not found; SAPIEN rendering may fail." >&2
    echo "[beaker-rt] (ldconfig has no libGLX_nvidia.so.0; check the node's GPU/driver mount)" >&2
  fi
}

# RoboTwin's embodiment/curobo .yml configs embed ABSOLUTE asset paths, expanded
# from ${ASSETS_PATH} by update_embodiment_config_path.py at download time (on the
# staging box that was '/root/RoboTwin'). Regenerate them from the *_tmp.yml
# templates so the paths point at the staged Weka assets (stable, always mounted).
beaker_fix_embodiment_paths() {
  local rt="${ROBOTWIN_ENV_DIR}"
  if [[ ! -d "${rt}/assets/embodiments" ]]; then
    echo "[beaker-rt] WARNING: ${rt}/assets/embodiments missing; skipping embodiment path fixup" >&2
    return 0
  fi
  if ! ls "${rt}"/assets/embodiments/**/*_tmp.yml >/dev/null 2>&1 && \
     [[ -z "$(find "${rt}/assets/embodiments" -name '*_tmp.yml' -print -quit 2>/dev/null)" ]]; then
    echo "[beaker-rt] WARNING: no *_tmp.yml templates under ${rt}/assets/embodiments;" >&2
    echo "[beaker-rt]          existing .yml may keep stale absolute paths." >&2
    return 0
  fi
  echo "[beaker-rt] Regenerating embodiment configs with ASSETS_PATH=${rt}"
  ( cd "${rt}" && "${PYTHON}" script/update_embodiment_config_path.py < /dev/null ) \
    | sed 's/^/[beaker-rt]   /' || echo "[beaker-rt] WARNING: embodiment path fixup returned nonzero" >&2
}

beaker_check_weka
beaker_check_robotwin_env
beaker_resolve_eval_paths
beaker_install_sim_deps
beaker_link_robotwin_env
beaker_link_policy
beaker_fix_embodiment_paths
beaker_register_nvidia_vulkan

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# RoboTwin's render setup (envs/_base_task.py and script/test_render.py) swallows
# render errors with a bare `except: print("Render Error"); exit()`, which exits 0
# and leaves no result file. Reproduce that exact setup here and print the real
# traceback so failures are diagnosable (non-fatal; the manager runs regardless).
beaker_probe_sapien_render() {
  echo "[beaker-rt] --- render diagnostics ---"
  echo "[beaker-rt] NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
  echo "[beaker-rt] VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
  echo "[beaker-rt] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi -L 2>&1 | head -8 | sed 's/^/[beaker-rt]   /' || true
  ls -1 /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null | sed 's/^/[beaker-rt]   icd: /' || true
  ldconfig -p 2>/dev/null | grep -iE "libGLX_nvidia|libvulkan|libnvidia-glvkspirv|libnvidia-rtcore" | sed 's/^/[beaker-rt]   lib: /' || true
  if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>&1 | grep -iE "deviceName|driverName|apiVersion|GPU id|deviceType" | head -12 | sed 's/^/[beaker-rt]   vk: /' || true
  fi
  echo "[beaker-rt] Probing SAPIEN ray-tracing render setup (real traceback if it fails)..."
  "${PYTHON}" - <<'PY' || echo "[beaker-rt] WARNING: SAPIEN render probe FAILED (see traceback above)"
import traceback
import sapien.core as sapien

def setup(denoiser):
    eng = sapien.Engine()
    from sapien.render import set_global_config
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    r = sapien.SapienRenderer()
    eng.set_renderer(r)
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    if denoiser is not None:
        sapien.render.set_ray_tracing_denoiser(denoiser)
    eng.create_scene(sapien.SceneConfig())
    return r

# Probe the exact RoboTwin config (oidn), then weaker denoisers to localize the cause.
for d in ("oidn", "optix", None):
    try:
        setup(d)
        print(f"[probe] RENDER OK (rt shader, denoiser={d})")
        break
    except Exception:
        print(f"[probe] RENDER FAIL (rt shader, denoiser={d}):")
        traceback.print_exc()
PY
}
beaker_probe_sapien_render

RUN_TAG="${EVAL_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${FASTWAM_RUNS_ROOT}/eval/robotwin/${TASK}/${RUN_TAG}}"
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
if [[ -n "${ROBOTWIN_TASK_NAME:-}" ]]; then
  HYDRA_ARGS+=("EVALUATION.task_name=${ROBOTWIN_TASK_NAME}")
fi
if [[ -n "${EVAL_HYDRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TOKENS=(${EVAL_HYDRA_OVERRIDES})
  HYDRA_ARGS+=("${EXTRA_TOKENS[@]}")
fi

echo "[beaker-rt] output_dir=${EVAL_OUTPUT_DIR}"
echo "[beaker-rt] >>> ${PYTHON} experiments/robotwin/run_robotwin_manager.py ${HYDRA_ARGS[*]}"

"${PYTHON}" experiments/robotwin/run_robotwin_manager.py "${HYDRA_ARGS[@]}"

echo "[beaker-rt] Done. Results: ${EVAL_OUTPUT_DIR}"
