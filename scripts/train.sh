#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CKPT_FILE="$SCRIPT_DIR/../checkpoints/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt"

mkdir -p $SCRIPT_DIR/../checkpoints/stable-diffusion-v1/

if [ -f "$CKPT_FILE" ]; then
    echo "SD1.5 checkpoint file already exists. Skipping download."
else
    echo "SD1.5 checkpoint file not found. Downloading..."
    curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -o "$CKPT_FILE"
fi

if [ -z "$1" ]; then
  echo "Usage: $0 <GPU_NUMS>"
  exit 1
fi

GPU_NUMS=$1

torchrun --nproc_per_node=$GPU_NUMS $SCRIPT_DIR/../main.py --name promptfix --base configs/promptfix.yaml --train --logdir train_logs