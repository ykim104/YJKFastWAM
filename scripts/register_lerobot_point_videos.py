#!/usr/bin/env python3
"""Register AllTracker ``observation.points.*`` videos in LeRobot ``meta/info.json``.

After running AllTracker ``inference_dataset.py``, MP4s live under::

    videos/chunk-*/observation.points.image/episode_*.mp4
    videos/chunk-*/observation.points.wrist_image/episode_*.mp4

This script copies the ``observation.images.*`` feature entries to ``observation.points.*``
and updates ``total_videos``. RGB image keys are unchanged.

Example (Weka)::

    python scripts/register_lerobot_point_videos.py \\
        --data-root /weka/oe-training/yejink/data/libero_mujoco3.3.2

Example (single dataset)::

    python scripts/register_lerobot_point_videos.py \\
        --dataset /weka/oe-training/yejink/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

IMAGE_TO_POINTS = (
    ("observation.images.image", "observation.points.image"),
    ("observation.images.wrist_image", "observation.points.wrist_image"),
)

DEFAULT_LIBERO_DATASETS = (
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)


def _load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def _collect_mp4s(dataset_root: Path, video_key: str) -> set[Path]:
    found: set[Path] = set()
    for pattern in (f"videos/**/{video_key}/*.mp4", f"videos/chunk-*/{video_key}/*.mp4"):
        found.update(dataset_root.glob(pattern))
    return found


def register_points_in_info(
    dataset_root: Path,
    *,
    dry_run: bool = False,
    points_codec: str = "h264",
) -> dict:
    dataset_root = dataset_root.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")

    info = _load_info(dataset_root)
    features = info.setdefault("features", {})
    added: list[str] = []
    warnings: list[str] = []

    for image_key, points_key in IMAGE_TO_POINTS:
        if image_key not in features:
            warnings.append(f"skip {points_key}: source feature {image_key!r} not in info.json")
            continue
        if features[image_key].get("dtype") != "video":
            warnings.append(
                f"skip {points_key}: {image_key!r} has dtype={features[image_key].get('dtype')!r}, expected 'video'"
            )
            continue

        mp4s = _collect_mp4s(dataset_root, points_key)
        if not mp4s:
            raise FileNotFoundError(
                f"No MP4 files for {points_key} under {dataset_root / 'videos'}. "
                f"Run AllTracker inference_dataset.py on this dataset first."
            )

        points_ft = copy.deepcopy(features[image_key])
        if points_ft.get("info"):
            points_ft["info"] = dict(points_ft["info"])
            points_ft["info"]["video.codec"] = points_codec

        features[points_key] = points_ft
        added.append(points_key)

    if not added:
        raise RuntimeError(f"No observation.points.* features registered for {dataset_root}")

    # Recompute total_videos from all video features present on disk.
    total_videos = 0
    for key, ft in features.items():
        if ft.get("dtype") != "video":
            continue
        n = len(_collect_mp4s(dataset_root, key))
        if n == 0:
            warnings.append(f"video feature {key!r} has dtype=video but 0 mp4 files on disk")
        total_videos += n

    info["total_videos"] = total_videos

    result = {
        "dataset": str(dataset_root),
        "added_keys": added,
        "total_videos": total_videos,
        "warnings": warnings,
    }

    if dry_run:
        result["dry_run"] = True
        return result

    backup = info_path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(info_path, backup)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
        f.write("\n")

    return result


def discover_datasets(data_root: Path) -> list[Path]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)

    named = [data_root / name for name in DEFAULT_LIBERO_DATASETS]
    if all((p / "meta" / "info.json").is_file() for p in named):
        return named

    return sorted(p for p in data_root.iterdir() if (p / "meta" / "info.json").is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Parent directory containing libero_*_lerobot datasets (registers all found).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Single LeRobot dataset directory (meta/info.json inside).",
    )
    parser.add_argument(
        "--points-codec",
        type=str,
        default="h264",
        help="video.codec written into observation.points.* feature info (AllTracker ffmpeg output).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write info.json.")
    args = parser.parse_args()

    if args.dataset is None and args.data_root is None:
        parser.error("Provide --dataset or --data-root.")

    if args.dataset is not None:
        datasets = [args.dataset]
    else:
        datasets = discover_datasets(args.data_root)

    print(f"Registering observation.points.* in {len(datasets)} dataset(s)...")
    for ds in datasets:
        print(f"\n=== {ds} ===")
        try:
            out = register_points_in_info(
                ds, dry_run=args.dry_run, points_codec=args.points_codec
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        for w in out.get("warnings", []):
            print(f"  WARNING: {w}")
        print(f"  added/updated keys: {out['added_keys']}")
        print(f"  total_videos: {out['total_videos']}")
        if args.dry_run:
            print("  (dry run — info.json not modified)")


if __name__ == "__main__":
    main()
