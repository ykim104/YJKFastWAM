"""Resolve resume checkpoint paths and align them with run output directories."""

from __future__ import annotations

from pathlib import Path


def run_dir_from_state_dir(state_dir: Path) -> Path:
    # .../{run_id}/checkpoints/state/{step_tag}
    checkpoints_dir = state_dir.parent.parent
    if checkpoints_dir.name != "checkpoints":
        raise ValueError(f"Expected checkpoints/ parent for state dir, got: {state_dir}")
    return checkpoints_dir.parent


def run_dir_from_weights_file(weights_path: Path) -> Path:
    # .../{run_id}/checkpoints/weights/{step_tag}.pt
    weights_dir = weights_path.parent
    if weights_dir.name != "weights" or weights_dir.parent.name != "checkpoints":
        raise ValueError(f"Expected checkpoints/weights/ layout, got: {weights_path}")
    return weights_dir.parent.parent


def _resolve_latest_state(latest_link: Path) -> Path:
    if not latest_link.exists() and not latest_link.is_symlink():
        raise FileNotFoundError(f"Checkpoint symlink not found: {latest_link}")
    resolved = latest_link.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Resolved checkpoint is not a directory: {resolved}")
    if not (resolved / "trainer_state.json").is_file():
        raise FileNotFoundError(
            f"Resolved checkpoint missing trainer_state.json (not a full training state): {resolved}"
        )
    return resolved


def resolve_resume_path(resume: str | Path, output_dir: str | Path | None = None) -> tuple[str, str]:
    """Map a user-provided resume path to (output_dir, resume_path_for_trainer).

    Accepts:
      - full state dir: .../checkpoints/state/step_XXXXXX
      - latest symlink: .../checkpoints/state/latest or .../{task}/latest
      - run directory:  .../{run_id}  (uses checkpoints/state/latest)
      - weights only:   .../checkpoints/weights/step_XXXXXX.pt

    When ``output_dir`` is omitted or does not contain the checkpoint, the run
    directory inferred from the resume path is used so ``latest`` symlinks are
    updated in the correct place on save.
    """
    path = Path(str(resume)).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    explicit_output = Path(output_dir).expanduser().resolve() if output_dir else None

    # weights-only checkpoint
    if path.is_file() and path.suffix == ".pt":
        run_dir = run_dir_from_weights_file(path)
        out = str(explicit_output) if explicit_output is not None else str(run_dir)
        return out, str(path)

    if path.name == "latest":
        state_dir = _resolve_latest_state(path)
        run_dir = run_dir_from_state_dir(state_dir)
        out = str(explicit_output) if explicit_output is not None else str(run_dir)
        return out, str(state_dir)

    if path.is_dir() and (path / "trainer_state.json").is_file():
        run_dir = run_dir_from_state_dir(path)
        out = str(explicit_output) if explicit_output is not None else str(run_dir)
        return out, str(path)

    if path.is_dir():
        state_latest = path / "checkpoints" / "state" / "latest"
        if state_latest.exists() or state_latest.is_symlink():
            state_dir = _resolve_latest_state(state_latest)
            run_dir = run_dir_from_state_dir(state_dir)
            out = str(explicit_output) if explicit_output is not None else str(run_dir)
            return out, str(state_dir)

        state_dir = path / "checkpoints" / "state"
        if state_dir.is_dir():
            nested_latest = state_dir / "latest"
            if nested_latest.exists() or nested_latest.is_symlink():
                resolved = _resolve_latest_state(nested_latest)
                run_dir = run_dir_from_state_dir(resolved)
                out = str(explicit_output) if explicit_output is not None else str(run_dir)
                return out, str(resolved)

    raise FileNotFoundError(
        f"Could not resolve resume path to a checkpoint: {resume}. "
        "Expected a state dir, state/latest symlink, run dir with checkpoints/state/latest, "
        "or checkpoints/weights/*.pt file."
    )
