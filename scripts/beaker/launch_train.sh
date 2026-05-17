#!/usr/bin/env bash
# Submit FastWAM training to AI2 Beaker.
#
# Usage:
#   ./scripts/beaker/launch_train.sh --user-name yejink --task libero_triple_2cam224_1e-4
#   ./scripts/beaker/launch_train.sh --user-name yejink --task libero_uncond_2cam224_1e-4 --wandb
#   ./scripts/beaker/launch_train.sh --user-name yejink --task libero_triple_2cam224_1e-4 --precompute-text --dry-run
#
# Requires: beaker CLI authenticated (`beaker account whoami`)
# Gantry: code comes from git clone in the job (no Weka copy required).
# Weka (oe-training-default -> /weka/oe-training): data, checkpoints, runs only.
# launch_train.py (raw Beaker): optional --code-dir on Weka if not using gantry.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "${REPO_ROOT}/scripts/beaker/launch_train.py" "$@"
