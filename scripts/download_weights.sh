#!/bin/bash
# Download LigandMPNN model weights

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/../graphrelax/LigandMPNN"
WEIGHTS_DIR="$PACKAGE_DIR/model_params"

mkdir -p "$WEIGHTS_DIR"

echo "Downloading LigandMPNN model weights..."

# Base URL for LigandMPNN weights
BASE_URL="https://files.ipd.uw.edu/pub/ligandmpnn"

# Download main model weights
wget -q --show-progress -O "$WEIGHTS_DIR/proteinmpnn_v_48_020.pt" \
    "$BASE_URL/proteinmpnn_v_48_020.pt"

wget -q --show-progress -O "$WEIGHTS_DIR/ligandmpnn_v_32_010_25.pt" \
    "$BASE_URL/ligandmpnn_v_32_010_25.pt"

wget -q --show-progress -O "$WEIGHTS_DIR/solublempnn_v_48_020.pt" \
    "$BASE_URL/solublempnn_v_48_020.pt"

# Download side chain packer weights
wget -q --show-progress -O "$WEIGHTS_DIR/ligandmpnn_sc_v_32_002_16.pt" \
    "$BASE_URL/ligandmpnn_sc_v_32_002_16.pt"

echo "Done! Model weights downloaded to $WEIGHTS_DIR"
