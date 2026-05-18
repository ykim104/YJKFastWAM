import os
import shutil
from pathlib import Path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def update_latest_symlink(link_path: str | Path, target_path: str | Path) -> Path:
    """Atomically point ``link_path`` at ``target_path`` (absolute dir target).

    Replaces an existing file, symlink, or directory at ``link_path``.
    """
    link = Path(link_path)
    target = Path(target_path).resolve()
    if not target.is_dir():
        raise NotADirectoryError(f"Latest symlink target must be a directory: {target}")

    link.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = link.with_name(f"{link.name}.tmp.{os.getpid()}")

    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(target, target_is_directory=True)

    if link.is_dir() and not link.is_symlink():
        shutil.rmtree(link)
    elif link.exists() or link.is_symlink():
        link.unlink()

    os.replace(tmp_link, link)
    return link
