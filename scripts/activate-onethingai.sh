#!/usr/bin/env bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/loopscape-py311
export HF_HOME=/root/onethingai-fs/cache/huggingface
cd /root/onethingai-fs/projects/loopscape-tangoflux
