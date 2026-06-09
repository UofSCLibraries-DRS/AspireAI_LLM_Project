#!/usr/bin/env bash
set -euo pipefail

# Change this to choose where the downloaded results should live locally.
DEST_DIR="${DEST_DIR:-$HOME/Research/library/data/results}"

# Change this if you want to pull a different group of remote model folders.
MODEL_DIR_PATTERN="${MODEL_DIR_PATTERN:-M*}"

REMOTE_USER="jaaydin"
REMOTE_HOST="login-theia.rc.sc.edu"
REMOTE_PORT="222"
REMOTE_MODELS_DIR="/work/jaaydin/models"

SSH_CMD="ssh -p ${REMOTE_PORT}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MODELS_DIR}/"

mkdir -p "$DEST_DIR"

echo "Pulling model results from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MODELS_DIR}"
echo "Writing to ${DEST_DIR}"

rsync -av --prune-empty-dirs \
  -e "$SSH_CMD" \
  --include="${MODEL_DIR_PATTERN}/" \
  --include="${MODEL_DIR_PATTERN}/results/***" \
  --exclude='*' \
  "$REMOTE" \
  "${DEST_DIR}/"
