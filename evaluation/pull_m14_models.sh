#!/usr/bin/env bash
set -euo pipefail

readonly REMOTE="jaaydin@login-theia.rc.sc.edu"
readonly REMOTE_DIR="/work/jaaydin/models"
readonly DEST_DIR="/home/john/Research/library/models"

mkdir -p "$DEST_DIR"

rsync \
  --archive \
  --partial \
  --info=progress2 \
  --include='M14*/***' \
  --include='M14*' \
  --exclude='*' \
  --rsh='ssh -p 222' \
  "$REMOTE:$REMOTE_DIR/" \
  "$DEST_DIR/"
