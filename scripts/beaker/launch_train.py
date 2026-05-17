#!/usr/bin/env python3
"""Generate and submit a FastWAM training job on AI2 Beaker.

Usage:
  python scripts/beaker/launch_train.py --user-name yejink --task libero_triple_2cam224_1e-4
  python scripts/beaker/launch_train.py --user-name yejink --task libero_uncond_2cam224_1e-4 --num-gpus 8 --dry-run
  python scripts/beaker/launch_train.py --user-name yejink --task robotwin_uncond_3cam_384_1e-4 --num-nodes 8 --num-gpus 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TRAIN_SH = REPO_ROOT / "scripts" / "beaker" / "run_train.sh"

DEFAULT_CLUSTER = "ai2/jupiter"


def _yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _weka_paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    weka_root = f"/weka/{args.weka_mount}/{args.user_name}"
    data_root = args.data_root or f"{weka_root}/data"
    checkpoints_root = args.checkpoints_root or args.model_base_path or f"{weka_root}/checkpoints"
    runs_root = args.runs_root or f"{weka_root}/runs"
    return weka_root, data_root, checkpoints_root, runs_root


def build_yaml(args: argparse.Namespace) -> str:
    code_dir = args.code_dir or f"/weka/{args.weka_mount}/{args.user_name}/YJKFastWam"
    run_script = args.run_script or f"{code_dir}/scripts/beaker/run_train.sh"
    weka_root, data_root, checkpoints_root, runs_root = _weka_paths(args)

    hydra_parts = [f"paths=weka", f"paths.weka_user={args.user_name}"]
    hydra_parts.extend(args.hydra_override)
    if args.wandb:
        hydra_parts.append("wandb.enabled=true")
    hydra = " ".join(hydra_parts).strip()

    env_lines = [
        f"  - name: USER_NAME",
        f'    value: "{args.user_name}"',
        f"  - name: CODE_DIR",
        f'    value: "{code_dir}"',
        f"  - name: TASK",
        f'    value: "{args.task}"',
        f"  - name: NUM_GPUS",
        f'    value: "{args.num_gpus}"',
        f"  - name: PRECOMPUTE_TEXT",
        f'    value: "{1 if args.precompute_text else 0}"',
        f"  - name: WEKA_ROOT",
        f'    value: "{weka_root}"',
        f"  - name: FASTWAM_DATA_ROOT",
        f'    value: "{data_root}"',
        f"  - name: FASTWAM_CHECKPOINTS_ROOT",
        f'    value: "{checkpoints_root}"',
        f"  - name: FASTWAM_RUNS_ROOT",
        f'    value: "{runs_root}"',
        f"  - name: DIFFSYNTH_MODEL_BASE_PATH",
        f'    value: "{checkpoints_root}"',
        f"  - name: HYDRA_OVERRIDES",
        f"    value: {_yaml_quote(hydra)}",
        f"  - name: NCCL_SOCKET_IFNAME",
        f'    value: "ib"',
        f"  - name: NCCL_TIMEOUT",
        f'    value: "36000000"',
        f"  - name: OMP_NUM_THREADS",
        f'    value: "{args.omp_threads}"',
        f"  - name: TERM",
        f'    value: "xterm"',
    ]
    if args.wandb:
        env_lines.extend(
            [
                f"  - name: WANDB_API_KEY",
                f"    secret: {args.wandb_secret or 'wandb-api-key'}",
                f"  - name: WANDB_MODE",
                f'    value: "online"',
            ]
        )

    env_block = "\n".join(env_lines) + "\n"

    cluster_list = args.cluster
    if args.beaker_image:
        image_block = f"    beaker: {args.beaker_image}\n"
    else:
        image_block = f"    docker: {args.docker_image}\n"

    lines = [
        "version: v2",
        f"description: {_yaml_quote(args.description)}",
        f"budget: {args.budget}",
        "tasks:",
        "- name: fastwam-train",
        f"  replicas: {args.num_nodes}",
        "  image:",
        image_block.rstrip(),
        "  command: ['/bin/bash', '-c']",
        "  arguments:",
        "  - >-",
        f"    bash {run_script}",
        "  datasets:",
        f"  - mountPath: /weka/{args.weka_mount}",
        "    source:",
        f"      weka: {args.weka_bucket}",
        "  result:",
        "    path: /data/results",
        "  envVars:",
    ]
    lines.extend(env_lines)
    lines.extend(
        [
            "  resources:",
            f"    gpuCount: {args.num_gpus}",
            f"    memory: {args.memory}",
            "  context:",
            f"    priority: {args.priority}",
            f"    preemptible: {str(args.preemptible).lower()}",
            "  constraints:",
            f"    cluster: [{cluster_list}]",
        ]
    )
    if args.num_nodes > 1:
        lines.extend(
            [
                "  hostNetworking: true",
                "  leaderSelection: true",
            ]
        )
    return "\n".join(lines) + "\n"


def print_summary(args: argparse.Namespace) -> None:
    code_dir = args.code_dir or f"/weka/{args.weka_mount}/{args.user_name}/YJKFastWam"
    weka_root, data_root, checkpoints_root, runs_root = _weka_paths(args)
    print("=" * 50)
    print(f" FastWAM training — {args.task}")
    print("=" * 50)
    print(f"  User:          {args.user_name}")
    print(f"  Code dir:      {code_dir}")
    print(f"  Data:          {data_root}")
    print(f"  Checkpoints:   {checkpoints_root}")
    print(f"  Runs:          {runs_root}")
    print(f"  Task:          {args.task}")
    print(f"  GPUs / node:   {args.num_gpus}")
    print(f"  Nodes:         {args.num_nodes}")
    print(f"  Total GPUs:    {args.num_gpus * args.num_nodes}")
    print(f"  Workspace:     {args.workspace}")
    print(f"  Budget:        {args.budget}")
    print(f"  Priority:      {args.priority}")
    print(f"  Cluster:       {args.cluster}")
    print(f"  Precompute:    {args.precompute_text}")
    if args.docker_image:
        print(f"  Image:         docker:{args.docker_image}")
    else:
        print(f"  Image:         beaker:{args.beaker_image}")
    print("=" * 50)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch FastWAM training on Beaker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              %(prog)s --user-name yejink --task libero_triple_2cam224_1e-4
              %(prog)s --user-name yejink --task libero_uncond_2cam224_1e-4 --wandb
              %(prog)s --user-name yejink --task robotwin_uncond_3cam_384_1e-4 --num-nodes 8 --num-gpus 8
            """
        ),
    )
    parser.add_argument("--user-name", required=True, help="Weka username (e.g. yejink)")
    parser.add_argument("--task", required=True, help="Hydra task name (configs/task/<name>.yaml)")
    parser.add_argument("--experiment-id", default=None, help="Short label for Beaker description")
    parser.add_argument("--num-gpus", type=int, default=8, help="GPUs per node (default: 8)")
    parser.add_argument("--num-nodes", type=int, default=1, help="Beaker replicas / nodes (default: 1)")
    parser.add_argument("--workspace", default="ai2/yejink-workspace", help="Beaker workspace")
    parser.add_argument("--budget", default="ai2/robots", help="Beaker budget account")
    parser.add_argument("--priority", default="normal", choices=["low", "normal", "high", "urgent", "immediate"])
    parser.add_argument(
        "--weka-bucket",
        default="oe-training-default",
        help="Beaker Weka bucket name (default: oe-training-default)",
    )
    parser.add_argument(
        "--weka-mount",
        default="oe-training",
        help="Mount path suffix: files at /weka/<mount>/... (default: oe-training)",
    )
    parser.add_argument(
        "--weka-volume",
        default=None,
        help="Alias for --weka-bucket (deprecated)",
    )
    parser.add_argument("--code-dir", default=None, help="Repo on weka (default: /weka/<mount>/<user>/YJKFastWam)")
    parser.add_argument("--data-root", default=None, help="Dataset root (default: /weka/<mount>/<user>/data)")
    parser.add_argument("--checkpoints-root", default=None, help="Checkpoints root (default: /weka/<mount>/<user>/checkpoints)")
    parser.add_argument("--runs-root", default=None, help="Training outputs (default: /weka/<mount>/<user>/runs)")
    parser.add_argument("--model-base-path", default=None, help="Alias for --checkpoints-root (DIFFSYNTH_MODEL_BASE_PATH)")
    parser.add_argument("--beaker-image", default=None, help="Beaker image (user/name)")
    parser.add_argument(
        "--docker-image",
        default="nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04",
        help="Docker image if --beaker-image is not set (default: CUDA 12.8 runtime)",
    )
    parser.add_argument("--memory", default="200GiB", help="System memory per replica")
    parser.add_argument("--omp-threads", default="16", help="OMP_NUM_THREADS")
    parser.add_argument("--precompute-text", action="store_true", help="Run T5 embed precompute before train")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb (wandb.enabled=true)")
    parser.add_argument(
        "--wandb-secret",
        default="wandb-api-key",
        help="Beaker secret name (env var in job is still WANDB_API_KEY)",
    )
    parser.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cluster",
        default=DEFAULT_CLUSTER,
        help="Beaker cluster to run on (default: ai2/jupiter)",
    )
    parser.add_argument("--hydra-override", nargs="*", default=[], help="Extra Hydra overrides")
    parser.add_argument("--run-script", default=None, help="Path to run_train.sh on the container")
    parser.add_argument("--dry-run", action="store_true", help="Print YAML only, do not submit")
    parser.add_argument("--save-yaml", default=None, help="Write generated YAML to this path")
    args = parser.parse_args()
    if args.weka_volume is not None:
        args.weka_bucket = args.weka_volume

    exp_label = args.experiment_id or args.task
    args.description = f"FastWAM train {exp_label} ({args.user_name}, {args.num_nodes}x{args.num_gpus} GPU)"

    if not RUN_TRAIN_SH.is_file():
        print(f"ERROR: missing {RUN_TRAIN_SH}", file=sys.stderr)
        return 1

    print_summary(args)
    yaml_content = build_yaml(args)
    print()

    if args.save_yaml:
        save_path = Path(args.save_yaml)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(yaml_content)
        print(f"Wrote {save_path}")

    if args.dry_run:
        print(yaml_content)
        return 0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    cmd = ["beaker", "experiment", "create", yaml_path, "-w", args.workspace]
    print(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
