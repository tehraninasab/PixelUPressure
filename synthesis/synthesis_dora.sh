

MODEL_VERSION=dora8
CHECKPOINT=45
START=0
STOP=-1

CUDA_VISIBLE_DEVICES=5,6,7 python3 synthesis/synthesize_images.py \
    --dataset_type chexpert \
    --combination_csv synthesized_metadata/balanced_combdist_5k.csv \
    --finetuned_model_path saved_models/sd15_${MODEL_VERSION}/checkpoint-${CHECKPOINT}-0 \
    --output_csv_path synthesized_metadata/sd15_${MODEL_VERSION}_5k.csv \
    --save_dir synthesized_images/sd15_${MODEL_VERSION}_5k \
    --path synthesized_images/sd15_${MODEL_VERSION}_5k \
    --batch_size 16 \
    --start_index ${START} \
    --stop_index ${STOP} \
    --model_version ${MODEL_VERSION} \
    --seed 42 
