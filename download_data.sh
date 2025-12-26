# Copyright (c) Meta Platforms, Inc. and affiliates.

#!/usr/bin/env bash
set -e

PROJECT_ID=np8b2
DEST_DIR=data

# Create destination directory
mkdir -p "${DEST_DIR}"

# Check osfclient
if ! command -v osf >/dev/null 2>&1; then
  echo "osfclient not found. Installing..."
  pip install --user osfclient
fi

echo "Downloading OSF project ${PROJECT_ID} -> ${DEST_DIR}"

# This fetches EVERYTHING recursively under osfstorage
# osf -p "${PROJECT_ID}" fetch --force osfstorage/ "${DEST_DIR}/"

osf --project "$PROJECT_ID" clone "$DEST_DIR"

echo "Download complete."
