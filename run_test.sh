root_path="/data/zebin"
export TORCH_HOME="${root_path}/pretrain"

export PYTHONPATH=/data/zebin/OpenIBL:$PYTHONPATH
# python ibl/models/model_hub.py
# python ibl/models/prnet.py
# python ibl/models/save_onnx.py
# python ibl/datasets/pitts.py
python common_test.py