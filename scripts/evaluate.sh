#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Bash script to run the CheXpert evaluation pipeline
# -----------------------------------------------------------------------------

# Path to the Python evaluation script
SCRIPT="scripts/evaluate.py"  # <-- Update with your actual script filename
MODEL_VERSION='bitfit'
CUDA_VISIBLE_DEVICES=4  # <-- Set this to the appropriate GPU ID if needed
# Default arguments (customize as needed)
CHEXPERT_CSV="dataset/chexpert/chexpert_test.csv"
CHEXPERT_ROOT="/cim/data"
GEN_CSV="synthesized_metadata/v1/sd15_${MODEL_VERSION}_5k.csv"
GEN_ROOT="/cim/amarkr/sdfs"
OUT_DIR="eval_out/${MODEL_VERSION}_5k"
BATCH=64
DEVICE="cuda"


# -----------------------------------------------------------------------------
# Usage message
# -----------------------------------------------------------------------------
function usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"  
    echo "  -s SCRIPT           Path to the evaluation Python script (default: $SCRIPT)"
    echo "  -r CHEXPERT_CSV     CSV of real CheXpert metadata (default: $CHEXPERT_CSV)"
    echo "  -R CHEXPERT_ROOT    Root dir of CheXpert images (default: $CHEXPERT_ROOT)"
    echo "  -g GEN_CSV          CSV of generated images metadata (default: $GEN_CSV)"
    echo "  -G GEN_ROOT         Root dir of generated images (default: $GEN_ROOT)"
    echo "  -o OUT_DIR          Output directory (default: $OUT_DIR)"
    echo "  -b BATCH            Batch size (default: $BATCH)"
    echo "  -d DEVICE           Device (cpu or cuda) (default: $DEVICE)"
    echo
    exit 1
}

# -----------------------------------------------------------------------------
# Parse command-line options
# -----------------------------------------------------------------------------
while getopts ":s:r:R:g:G:o:b:d:n:k:" opt; do
    case $opt in
        s) SCRIPT="$OPTARG" ;;      
        r) CHEXPERT_CSV="$OPTARG" ;;  
        R) CHEXPERT_ROOT="$OPTARG" ;;  
        g) GEN_CSV="$OPTARG" ;;       
        G) GEN_ROOT="$OPTARG" ;;      
        o) OUT_DIR="$OPTARG" ;;      
        b) BATCH="$OPTARG" ;;       
        d) DEVICE="$OPTARG" ;;      
        *) usage ;;                   
    esac
done

# -----------------------------------------------------------------------------
# Run the evaluation
# -----------------------------------------------------------------------------
echo "Running evaluation script with the following parameters:"
echo "  SCRIPT         = $SCRIPT"
echo "  CHEXPERT_CSV   = $CHEXPERT_CSV"
echo "  CHEXPERT_ROOT  = $CHEXPERT_ROOT"
echo "  GEN_CSV        = $GEN_CSV"
echo "  GEN_ROOT       = $GEN_ROOT"
echo "  OUT_DIR        = $OUT_DIR"
echo "  BATCH          = $BATCH"
echo "  DEVICE         = $DEVICE"

time CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "$SCRIPT" \
    --chexpert_csv "$CHEXPERT_CSV" \
    --chexpert_root "$CHEXPERT_ROOT" \
    --gen_csv "$GEN_CSV" \
    --gen_root "$GEN_ROOT" \
    --out_dir "$OUT_DIR" \
    --batch $BATCH \
    --device "$DEVICE"


echo "\nEvaluation complete!"