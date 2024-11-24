set -ex
cd /workspace/pangyunhe/project/M-IND

pip install -r requirements.txt
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online   
wandb enabled
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
# wandb offline
# wandb disabled

# NUM_GPUS=8

# torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  pipeline_for_multimodal.py \
# deepspeed --include localhost:3  pipeline_for_multimodal.py \
deepspeed --include localhost:1 predict_for_multimodal_old.py \
    configs/last/last_predict.json 2>&1 | tee output/last/last_predict.log