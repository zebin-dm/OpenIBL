#!/bin/sh
root_path="/data/zebin"
export TORCH_HOME="${root_path}/pretrain"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/data/zebin/OpenIBL:$PYTHONPATH
PYTHON=${PYTHON:-"python"}
GPUS=8

DATASET=pitts
# SCALE=30k
SCALE=250k
ARCH=prnet
BB_NAME=vgg16
conv_dim=256
ReduceDim=4096
LAYERS=conv5
LOSS=sare_ind
LR=0.001

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img_sfrs.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --syncbn \
  --width 640 --height 480 --tuple-size 1 -j 2 --test-batch-size 16 \
  --neg-num 10  --pos-pool 20 --neg-pool 1000 --pos-num 10 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} --soft-weight 0.5 \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 --generations 20 --temperature 0.07 0.07 0.06 0.05 \
  --logs-dir logs/netVLAD/${DATASET}${SCALE}-${ARCH}/${BB_NAME}-${LOSS}-lr${LR}-tuple${GPUS}-cd${conv_dim}-rd${ReduceDim}-SFRS \
  --bb_name ${BB_NAME} --conv_dim ${conv_dim} --reduce_dim ${ReduceDim}\
  >./nohug.log  2>&1 &
#   --sync-gather
