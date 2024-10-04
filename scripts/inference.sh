#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir $SCRIPT_DIR/../checkpoints
wget -P $SCRIPT_DIR/../checkpoints/ -N https://huggingface.co/yeates/PromptFix/resolve/main/promptfix.ckpt

echo "Pre-trained model downloaded to checkpoints/promptfix.ckpt"
echo "Begin inference..."
python $SCRIPT_DIR/../process_images_json.py