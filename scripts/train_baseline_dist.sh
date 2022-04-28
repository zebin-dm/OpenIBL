#!/bin/sh
root_path="/data/zebin"
export TORCH_HOME="${root_path}/pretrain"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export PYTHONPATH=/data/zebin/OpenIBL:$PYTHONPATH
PYTHON=${PYTHON:-"python"}
GPUS=7

DATASET=pitts
SCALE=250k
# ARCH=prnet
BB_NAME=vgg16
ARCH=vgg16
conv_dim=512
ReduceDim=4096
LAYERS=conv5
LOSS=$1
LR=0.001

if [ $# -ne 1 ]
  then
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    exit 1
fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn \
  --width 640 --height 480 --tuple-size 1 -j 2 --neg-num 10 --test-batch-size 16 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 100 --step-size 5 --cache-size 1000 \
  --resume "" \
  --logs-dir logs/netVLADBaseline/${DATASET}${SCALE}-${ARCH}/${BB_NAME}-${LOSS}-lr${LR}-tuple${GPUS}-cd${conv_dim}-rd${ReduceDim} \
  --bb_name ${BB_NAME} --conv_dim ${conv_dim} --reduce_dim ${ReduceDim}\
  >./nohug.log  2>&1 &
