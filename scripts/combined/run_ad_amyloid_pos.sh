#!/bin/bash
#
#SBATCH --job-name=ad-am-pos
#SBATCH --error=ad-am-pos.err
#SBATCH --output=ad-am-pos.out
#
#SBATCH --time=2-00:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu_mem:20000

# This assumes that the conda environment has been created using the requirements.txt file:
# 1. conda create --name connected_speech_classification python=3.11
# 2. pip install -r requirements.txt
source activate connected-speech-classification

REPO_DIR="../.."
LARGE_DATA_DIR="/data/u0150403"

# Define all interview parts
allQs=("q1_short_subject_wise" "q5_subject_wise" "q4_subject_wise" "q3_subject_wise" "q2_subject_wise" "q1_subject_wise" "subject_wise")
EPOCHS=10
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
MLFLOW_URI="${LARGE_DATA_DIR}/mlflow/mlruns"
MODEL_BASE="helena-balabin/robbert-2023-dutch-base-ft-nlp-xxl"

# 1. Data processing
python3 $REPO_DIR/connected_speech_classification/data/prepare_disease_status_datasets.py prepare-ad-vs-am-pos-datasets

# 2. Classification of AD versus amyloid positive CU
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--batch_size $BATCH_SIZE \
	--epochs $EPOCHS \
	--n_data_splits 1 \
	--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/ad_am_pos_$q" \
	--mlflow_tracking_uri $MLFLOW_URI \
	--classification_model_base $MODEL_BASE
done

# 3. Mlflow results to LaTeX tables
# 3.1 Export the mlflow experiments to csv files
mlflow experiments csv -x 685451340341447252 -o "${REPO_DIR}/data/output/mlflow-results/ad_am_pos_cls.csv"

# 3.2 Convert the csv files to LaTeX tables
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/ad_am_pos_cls.csv"
