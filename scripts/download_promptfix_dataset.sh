#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Datasets will be downloaded and extracted to ./dataset/PromptfixData/data/"
echo "You are going to need ~416GB of free disk space."

read -p "[y] to confirm, others to exit: " reply
if [ "$reply" != "y" ]; then
    echo "Exiting..."
    exit 1
fi

mkdir -p $SCRIPT_DIR/../dataset/
cd $SCRIPT_DIR/../dataset/

pip install -U huggingface_hub
huggingface-cli download --repo-type dataset --resume-download yeates/PromptfixData --local-dir PromptfixData --local-dir-use-symlinks False

echo "Download completed"