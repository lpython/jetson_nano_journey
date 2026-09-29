#!/usr/bin/env bash
# One-way mirror: Mac (git repo) -> Nano (build dir). Remote edits are overwritten.
set -euox pipefail

JETSON="${JETSON:-nano}"
REMOTE_DIR="${REMOTE_DIR:-jetson_nano_journey}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
  --filter=':- .gitignore' \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  "$ROOT/" "$JETSON:$REMOTE_DIR/"
