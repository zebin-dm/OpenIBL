#!/bin/sh
root_path="/data/zebin"
export TORCH_HOME="${root_path}/pretrain"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/data/zebin/OpenIBL:$PYTHONPATH
PYTHON=${PYTHON:-"python"}
GPUS=1

python examples/feature_pca.py
