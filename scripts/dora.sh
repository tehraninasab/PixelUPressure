#!/bin/bash

#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:4
#SBATCH --job-name=dora
#SBATCH --output=/home/a/amarkr1/workshop/sdfs/logs/slurm-%j.out
#SBATCH --time=3:00:00

module load python3
module load cuda
module load httpproxy

source /scratch/a/amarkr1/venvs/pixelperfect-env/bin/activate

accelerate launch scripts/finetune_chexpert.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --train_data_path dataset/chexpert/chexpert_train_20k.csv \
    --image_root_path /scratch/a/amarkr1/data/ \
    --output_dir /scratch/a/amarkr1/sdfs/saved_models \
    --resolution 512 \
    --train_batch_size 8 \
    --num_train_epochs 101 \
    --gradient_accumulation_steps 4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_weight_decay 1e-2 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 1.0 \
    --num_workers 6 \
    --save_steps 2500 \
    --save_epochs 5 \
    --use_dora \
    --lora_rank 8 \
    --unet_lr 1e-4 \
