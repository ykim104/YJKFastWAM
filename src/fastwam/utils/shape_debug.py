"""Tiny shape-debug helper for tracing tensor shapes through the model.

Enable by setting `FASTWAM_DEBUG_SHAPES=1` in the environment. Print output is
prefixed with `[shape]` so it is easy to grep / strip in production runs.

The helper is intentionally cheap when disabled: a single env-var lookup at
import time, and an early-return boolean check per `dprint` call.
"""

from __future__ import annotations

import os
from typing import Any

import torch

_ENABLED = os.environ.get("FASTWAM_DEBUG_SHAPES", "0") not in ("", "0", "false", "False")


def shape_debug_enabled() -> bool:
    return _ENABLED


def _fmt(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, (list, tuple)):
        return type(value).__name__ + "(" + ", ".join(_fmt(v) for v in value) + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={_fmt(v)}" for k, v in value.items()) + "}"
    return repr(value)


def dprint(scope: str, **named_tensors: Any) -> None:
    """Print a labelled set of tensors / values when shape debug is enabled."""
    if not _ENABLED:
        return
    parts = [f"{name}={_fmt(value)}" for name, value in named_tensors.items()]
    print(f"[shape] {scope}: " + " | ".join(parts), flush=True)


def dsection(title: str) -> None:
    if not _ENABLED:
        return
    bar = "=" * max(8, 80 - len(title) - 2)
    print(f"[shape] {title} {bar}", flush=True)
