#!/bin/bash
# Download LigandMPNN model weights
#
# By default, downloads to ~/.graphrelax/weights/
# Override with GRAPHRELAX_WEIGHTS_DIR environment variable

set -e

# Determine weights directory
if [ -n "$GRAPHRELAX_WEIGHTS_DIR" ]; then
    WEIGHTS_DIR="$GRAPHRELAX_WEIGHTS_DIR"
else
    WEIGHTS_DIR="$HOME/.graphrelax/weights"
fi

mkdir -p "$WEIGHTS_DIR"

echo "Downloading LigandMPNN model weights to $WEIGHTS_DIR..."

# Base URL for LigandMPNN weights
BASE_URL="https://files.ipd.uw.edu/pub/ligandmpnn"

# Download main model weights
echo "Downloading proteinmpnn_v_48_020.pt..."
curl -# -o "$WEIGHTS_DIR/proteinmpnn_v_48_020.pt" \
    "$BASE_URL/proteinmpnn_v_48_020.pt"

echo "Downloading ligandmpnn_v_32_010_25.pt..."
curl -# -o "$WEIGHTS_DIR/ligandmpnn_v_32_010_25.pt" \
    "$BASE_URL/ligandmpnn_v_32_010_25.pt"

echo "Downloading solublempnn_v_48_020.pt..."
curl -# -o "$WEIGHTS_DIR/solublempnn_v_48_020.pt" \
    "$BASE_URL/solublempnn_v_48_020.pt"

# Download side chain packer weights
echo "Downloading ligandmpnn_sc_v_32_002_16.pt..."
curl -# -o "$WEIGHTS_DIR/ligandmpnn_sc_v_32_002_16.pt" \
    "$BASE_URL/ligandmpnn_sc_v_32_002_16.pt"

echo "Done! Model weights downloaded to $WEIGHTS_DIR"
