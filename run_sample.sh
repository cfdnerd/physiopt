#!/bin/bash

# ==========================================
# Configuration Section
# ==========================================

# Text prompt for generation
PROMPT="A lightweight structural beam designed to carry heavy loads."

# Attention backend (options: xformers, flash_attn, sdpa)
BACKEND="xformers"

# Output folder
OUT_FOLDER="out/sampleOut"

# Number of samples
N_SAMPLES=5

# ==========================================
# Execution Logic
# ==========================================

# Allow command line argument to override the config prompt
if [ "$#" -ge 1 ]; then
    PROMPT="$1"
fi

if [ -z "$PROMPT" ]; then
    echo "Error: Prompt cannot be empty. Set it in the script or pass as argument."
    exit 1
fi

echo "Running sample with prompt: '$PROMPT'"
echo "Output folder: $OUT_FOLDER"
echo "Backend: $BACKEND"

ATTN_BACKEND=$BACKEND python scripts/sample.py \
    --text "$PROMPT" \
    --out_folder "$OUT_FOLDER" \
    --n_samples "$N_SAMPLES" \
    --mesh
