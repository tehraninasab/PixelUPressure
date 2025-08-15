

MODEL_VERSION=unet
CHECKPOINT=45
START=0
STOP=-1

python3 synthesis/synthesize_images.py \
    --dataset_type chexpert \
    --combination_csv synthesized_metadata/balanced_combdist_5k.csv \
    --finetuned_model_path saved_models/gcp/sd15_${MODEL_VERSION}/checkpoint-${CHECKPOINT}-0 \
    --output_csv_path synthesized_metadata/sd15_${MODEL_VERSION}_5k.csv \
    --save_dir output/sd15_${MODEL_VERSION}_5k \
    --path output/sd15_${MODEL_VERSION}_5k \
    --batch_size 16 \
    --start_index ${START} \
    --stop_index ${STOP} \
    --model_version ${MODEL_VERSION} \
    --seed 42 

