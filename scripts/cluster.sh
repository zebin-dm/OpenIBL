#!/bin/sh
root_path="/data/zebin"
export TORCH_HOME="${root_path}/pretrain"

export PYTHONPATH=/data/zebin/OpenIBL:$PYTHONPATH
ARCH=prnet
BB_NAME=vgg16
conv_dim=256

# if [ $# -ne 1 ]
#   then
#     echo "Arguments error: <ARCH>"
#     exit 1
# fi

python -u examples/cluster.py -d pitts -a ${ARCH} -b 64 --width 640 --height 480 \
     --bb_name ${BB_NAME}  --conv_dim ${conv_dim}
