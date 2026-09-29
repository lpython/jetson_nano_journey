#!/usr/bin/env bash
# Copy results back from the Nano. Usage: scripts/pull.sh results/   (path relative to project)
set -euo pipefail

JETSON="${JETSON:-nano}"
REMOTE_DIR="${REMOTE_DIR:-jetson_nano_journey}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB="${1:?usage: pull.sh <remote-subpath>}"

mkdir -p "$ROOT/$(dirname "$SUB")"
rsync -az "$JETSON:$REMOTE_DIR/$SUB" "$ROOT/$SUB"
