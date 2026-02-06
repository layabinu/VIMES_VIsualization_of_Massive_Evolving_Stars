#!/bin/bash

usage() {
    echo "Usage: $0 <log|linear> <tulips|default> [output.mp4] [headless]"
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

FRAMES="/Users/ronin/animation/Final Animation/frames_data.npz"
PREPROCESS="/Users/ronin/animation/Final Animation/preprocess.py"
ANIMATE="/Users/ronin/animation/Final Animation/animate.py"
COLOR="/Users/ronin/animation/Final Animation/temp_to_color.py"

# Preprocess if needed
if [ ! -f "$FRAMES" ]; then
    echo "frames_data.npz not found. Running preprocess..."
    python3 "$PREPROCESS"
    python3 "$COLOR"
else
    echo "frames_data.npz found. Skipping preprocess."
fi

# Build animation command
CMD=(python3 "$ANIMATE" --scaling "$SCALING" --images "$IMAGES")

if [ -n "$MP4_OUT" ]; then
    CMD+=(--save-mp4 "$MP4_OUT")
fi

if [ "$HEADLESS" = "headless" ]; then
    CMD+=(--no-display)
fi

echo "Running animate.py:"
echo "${CMD[@]}"

"${CMD[@]}"
