#!/bin/bash

usage() {
    echo "Usage: $0 <log|linear> <tulips|default> [output.mp4] [headless]"
    echo ""
    echo "Environment:"
    echo "  HDF5   input simulation file (default: examples/BSE_Detailed_Output_0.h5)"
    echo "  FRAMES preprocessed frames file (default: frames_data.npz)"
    exit 1
}

# Require at least 2 args
if [ "$#" -lt 2 ]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

SCALING=$1
IMAGES=$2
MP4_OUT=$3
HEADLESS=$4

# Validate scaling
case "$SCALING" in
    log|linear) ;;
    *)
        echo "Error: Invalid scaling value '$SCALING'. Must be 'log' or 'linear'."
        usage
        ;;
esac

# Validate images
case "$IMAGES" in
    tulips|default) ;;
    *)
        echo "Error: Invalid images value '$IMAGES'. Must be 'tulips' or 'default'."
        usage
        ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HDF5="${HDF5:-$REPO_DIR/examples/BSE_Detailed_Output_0.h5}"
FRAMES="${FRAMES:-frames_data.npz}"

# Preprocess if needed
if [ ! -f "$FRAMES" ]; then
    echo "$FRAMES not found. Running preprocess on $HDF5..."
    vimes-preprocess "$HDF5" "$FRAMES" || exit 1
else
    echo "$FRAMES found. Skipping preprocess."
fi

# Build animation command
CMD=(vimes "$FRAMES" --scaling "$SCALING" --images "$IMAGES")

if [ -n "$MP4_OUT" ]; then
    CMD+=(--save-mp4 "$MP4_OUT")
fi

if [ "$HEADLESS" = "headless" ]; then
    CMD+=(--no-display)
fi

echo "Running vimes:"
echo "${CMD[@]}"

"${CMD[@]}"
