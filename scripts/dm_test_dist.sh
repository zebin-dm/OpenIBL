#!/bin/sh
root_path="/data/zebin"
export TORCH_HOME="${root_path}/pretrain"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export PYTHONPATH=/data/zebin/OpenIBL:$PYTHONPATH
PYTHON=${PYTHON:-"python"}
GPUS=7

# RESUME=$1
ARCH=vgg16

DATASET=pitts
SCALE=250k

# if [ $# -lt 1 ]
#   then
#     echo "Arguments error: <MODEL PATH>"
#     echo "Optional arguments: <DATASET (default:pitts)> <SCALE (default:250k)>"
#     exit 1
# fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/dm_test_pitts_best.py --launcher pytorch \
    --scale ${SCALE} \
    --test-batch-size 32 -j 2 