#!/bin/bash
#
#SBATCH --job-name=debug-all-steps-combined
#SBATCH --error=debug-all-steps-combined.err
#SBATCH --output=debug-all-steps-combined.out
#
#SBATCH --time=72:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu_mem:20000

# This assumes that the conda environment has been created using the requirements.txt file:
# 1. conda create --name connected_speech_classification python=3.11
# 2. pip install -r requirements.txt
source activate connected-speech-classification

# Define all interview parts
allQs=("q1_short_subject_wise")
EPOCHS=1
FOLDS=3

# Note that the interactive input is not working in the SLURM environment
# Ask the user for the repository directory with a default value
echo "Please provide the path to the repository directory (default: ./):"
read REPO_DIR
REPO_DIR=${REPO_DIR:-../../}
REPO_DIR=$(realpath "$REPO_DIR")
echo "Using repository directory: $REPO_DIR"

# Ask the user for the large data directory with a default value
echo "Please provide the path to the large data directory (default: ./data):"
read LARGE_DATA_DIR
LARGE_DATA_DIR=${LARGE_DATA_DIR:-../../data}
LARGE_DATA_DIR=$(realpath "$LARGE_DATA_DIR")
echo "Using large data directory: $LARGE_DATA_DIR"

# 1. Data processing
python3 $REPO_DIR/connected_speech_classification/data/prepare_disease_status_datasets.py prepare-ad-hc-amyloid-pos-neg-datasets \
	--combined_dataset_dir "${REPO_DIR}/data/input/combined-cohort-e2e-v4"

# 2. Experiment 1: Independent classification
# 2.1 Classification on AD vs half of amyloid negative HC
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--batch_size 4 \
	--k_folds $FOLDS \
	--freeze_most_layers \
	--gradient_accumulation_steps 1 \
	--epochs $EPOCHS \
	--n_data_splits 2 \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/ad_half_neg_hc_$q"
done
# 2.2 Classification of amyloid positive vs other half of amyloid negative HC
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--classify_amyloid \
	--batch_size 4 \
	--k_folds $FOLDS \
	--freeze_most_layers \
	--gradient_accumulation_steps 1 \
	--epochs $EPOCHS \
	--n_data_splits 2 \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/amyloid_pos_half_neg_cls_$q"
done

# 3. Experiment 2: Sequential classification
# Previously: Get the checkpoint, use the last checkpoint (last changed folder)
# first_checkpoint_folder=$(find "${LARGE_DATA_DIR}/huggingface/transformers/ad_half_neg_hc_subject_wise_ad_hc_ft_subject_wise_helena-balabin-robbert-2023-dutch-base-ft-nlp-xxl/" -type d -name 'checkpoint*' -print | tail -n 1)
# Now: Get the saved model (no checkpoints)
first_checkpoint_folder="${LARGE_DATA_DIR}/huggingface/transformers/ad_half_neg_hc_subject_wise_ad_hc_ft_subject_wise_helena-balabin-robbert-2023-dutch-base-ft-nlp-xxl/"
echo "Checkpoint folder for exp 2: $first_checkpoint_folder"

# Use the classification model that has been trained on AD vs half of amyloid negative (combined interview) to classify amyloid positive versus other half of amyloid negative
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--classification_model_base "$first_checkpoint_folder" \
	--classify_amyloid \
	--batch_size 4 \
	--k_folds $FOLDS \
	--freeze_most_layers \
	--gradient_accumulation_steps 1 \
	--epochs $EPOCHS \
	--n_data_splits 2 \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/amyloid_pos_half_neg_cls_$q"
done

# 4. Experiment 3: Joint multi-task classification
# Train a classification model jointly on AD vs HC and amyloid positive versus amyloid negative
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--classify_jointly \
	--batch_size 4 \
	--k_folds $FOLDS \
	--freeze_most_layers \
	--gradient_accumulation_steps 1 \
	--epochs $EPOCHS \
	--n_data_splits 2 \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/ad_half_neg_hc_$q" --dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/amyloid_pos_half_neg_cls_$q"
done

# 5. Mlflow results to LaTeX tables
# 5.1 Export the mlflow experiments to csv files
mlflow experiments csv -x 685451340341447252 -o "${REPO_DIR}/data/output/mlflow-results/ad_hc_cls.csv"
mlflow experiments csv -x 523382240786442755 -o "${REPO_DIR}/data/output/mlflow-results/amyloid_cls.csv"
mlflow experiments csv -x 136153611125199525 -o "${REPO_DIR}/data/output/mlflow-results/joint_ad_amyloid_cls.csv"
# 5.2 Convert the csv files to LaTeX tables
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/ad_hc_cls.csv" \
	--output_dir "${REPO_DIR}/data/output/mlflow-results-debug"
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/amyloid_cls.csv" \
	--output_dir "${REPO_DIR}/data/output/mlflow-results-debug"
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/joint_ad_amyloid_cls.csv" \
	--output_dir "${REPO_DIR}/data/output/mlflow-results-debug"
