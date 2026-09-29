#!/usr/bin/env bash
# Sync, then run a command in the project dir on the Nano. Usage: scripts/remote.sh make
set -euo pipefail

JETSON="${JETSON:-nano}"
REMOTE_DIR="${REMOTE_DIR:-jetson_nano_journey}"

"$(dirname "$0")/sync.sh"
ssh -t "$JETSON" "export PATH=/usr/local/cuda/bin:\$PATH; cd $REMOTE_DIR && $*"
