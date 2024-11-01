#!/bin/bash

# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROMPTFIX_HOME=$(pwd)
echo $PROMPTFIX_HOME 

mkdir $PROMPTFIX_HOME/checkpoints
wget -P $PROMPTFIX_HOME/checkpoints/ -N https://huggingface.co/yeates/PromptFix/resolve/main/promptfix.ckpt

echo "Pre-trained model downloaded to checkpoints/promptfix.ckpt"
echo "Begin inference..."
python $PROMPTFIX_HOME/process_images_json.py