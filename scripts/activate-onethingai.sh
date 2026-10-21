#!/usr/bin/env bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/loopscape-py311
export HF_HOME=/root/onethingai-fs/cache/huggingface
export TORCH_HOME=/root/onethingai-fs/cache/torch
export XDG_CACHE_HOME=/root/onethingai-fs/cache/xdg
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"
cd /root/onethingai-fs/projects/loopscape-tangoflux
