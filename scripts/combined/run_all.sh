#!/bin/bash
#
#SBATCH --job-name=all-steps-combined
#SBATCH --error=all-steps-combined.err
#SBATCH --output=all-steps-combined.out
#
#SBATCH --time=3-00:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu_mem:20000

# This assumes that the conda environment has been created using the requirements.txt file:
# 1. conda create --name connected_speech_classification python=3.11
# 2. pip install -r requirements.txt
source activate connected_speech_classification

# Ask the user for the repository directory with a default value
echo "Please provide the path to the repository directory (default: ./):"
read REPO_DIR
REPO_DIR=${REPO_DIR:-./}
# Ask the user for the large data directory with a default value
echo "Please provide the path to the large data directory (default: ./data/):"
read LARGE_DATA_DIR
LARGE_DATA_DIR=${LARGE_DATA_DIR:-./data/}

# Define all interview parts
allQs=("q1_short_subject_wise" "q5_subject_wise" "q4_subject_wise" "q3_subject_wise" "q2_subject_wise" "q1_subject_wise" "subject_wise")
EPOCHS=10
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
MLFLOW_URI="${LARGE_DATA_DIR}/mlflow/mlruns"
MODEL_BASE="helena-balabin/robbert-2023-dutch-base-ft-nlp-xxl"

# 1. Data processing
python3 $REPO_DIR/connected_speech_classification/data/prepare_disease_status_datasets.py prepare-ad-hc-amyloid-pos-neg-datasets \
	--combined_dataset_dir "${REPO_DIR}/data/input/combined-cohort-e2e-v4"

# 2. Experiment 1: Independent classification
# 2.1 Classification on AD vs half of amyloid negative HC
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
 	--config "$q" \
	--batch_size $BATCH_SIZE \
	--epochs $EPOCHS \
	--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/ad_half_neg_hc_$q" \
	--mlflow_tracking_uri $MLFLOW_URI \
	--classification_model_base $MODEL_BASE
done
# 2.2 Classification of amyloid positive vs other half of amyloid negative HC
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--classify_amyloid \
	--batch_size $BATCH_SIZE \
	--epochs $EPOCHS \
	--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/amyloid_pos_half_neg_cls_$q" \
	--mlflow_tracking_uri $MLFLOW_URI \
	--classification_model_base $MODEL_BASE
done
# 2.3 Classification of amyloid negative group 1 versus amyloid negative group 2
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--classify_baseline \
	--batch_size $BATCH_SIZE \
	--epochs $EPOCHS \
	--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/amyloid_neg_baseline_$q" \
	--mlflow_tracking_uri $MLFLOW_URI \
	--classification_model_base $MODEL_BASE
done

# 3. Experiment 2: Sequential classification
# Get the saved model (no checkpoints)
saved_model_folder="${LARGE_DATA_DIR}/huggingface/transformers/ad_half_neg_hc_subject_wise_ad_hc_ft_subject_wise_helena-balabin-robbert-2023-dutch-base-ft-nlp-xxl"
# Use the classification model that has been trained on AD vs half of amyloid negative (combined interview) to classify amyloid positive versus other half of amyloid negative
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--classification_model_base "$saved_model_folder" \
	--classify_amyloid \
	--batch_size $BATCH_SIZE \
	--epochs $EPOCHS \
	--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/amyloid_pos_half_neg_cls_$q" \
	--mlflow_tracking_uri $MLFLOW_URI \
	--classification_model_base $MODEL_BASE
done

# 4. Experiment 3: Joint multi-task classification
# Train a classification model jointly on AD vs amyloid negative HC and amyloid positive versus amyloid negative
for q in "${allQs[@]}"; do
python3 $REPO_DIR/connected_speech_classification/models/disease_status_classifier.py classify-disease-label \
	--config "$q" \
	--classify_jointly \
	--batch_size $BATCH_SIZE \
	--epochs $EPOCHS \
	--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
	--dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/ad_half_neg_hc_$q" --dataset_dir "${LARGE_DATA_DIR}/huggingface/datasets/amyloid_pos_half_neg_cls_$q" \
	--mlflow_tracking_uri $MLFLOW_URI \
	--classification_model_base $MODEL_BASE
done

# 5. Mlflow results to LaTeX tables
# 5.1 Export the mlflow experiments to csv files
mlflow experiments csv -x 685451340341447252 -o "${REPO_DIR}/data/output/mlflow-results/ad_hc_cls.csv"
mlflow experiments csv -x 523382240786442755 -o "${REPO_DIR}/data/output/mlflow-results/amyloid_cls.csv"
mlflow experiments csv -x 136153611125199525 -o "${REPO_DIR}/data/output/mlflow-results/joint_ad_amyloid_cls.csv"
mlflow experiments csv -x 861177041110502316 -o "${REPO_DIR}/data/output/mlflow-results/baseline_cls.csv"

# 5.2 Convert the csv files to LaTeX tables
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/ad_hc_cls.csv"
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/amyloid_cls.csv"
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/joint_ad_amyloid_cls.csv"
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/baseline_cls.csv"

# 5.3 Also for the first data_split
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/ad_hc_cls.csv" --data_split 0
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/amyloid_cls.csv" --data_split 0
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/joint_ad_amyloid_cls.csv" --data_split 0
python3 $REPO_DIR/connected_speech_classification/evaluation/format_mlflow_results.py convert-mlflow-tables \
	--result_file "${REPO_DIR}/data/output/mlflow-results/baseline_cls.csv" --data_split 0