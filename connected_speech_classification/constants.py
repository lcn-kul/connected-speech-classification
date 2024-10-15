# -*- coding: utf-8 -*-

"""Constants."""

import os

from dotenv import load_dotenv

# Use dotenv to properly load device-dependent constants
load_dotenv()

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

# Directory for data, logs, models, notebooks
DATA_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "data")
LOGS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "logs")
MODELS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "models")
NOTEBOOKS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "notebooks")
# See if a separate large storage directory should be used for large project-specific datasets and models
LARGE_NLP_HBT_STORAGE_DIR = os.getenv("LARGE_NLP_HBT_STORAGE_PATH") or DATA_DIR

# Sub-directories of data
RAW_DIR = os.path.join(DATA_DIR, "raw")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MISC_DIR = os.path.join(DATA_DIR, "misc")

# Dir for mlflow results
MLFLOW_RESULTS_DIR = os.path.join(OUTPUT_DIR, "mlflow-results")

# Dir for combined data from ARCK + MC + F-PACK
COMBINED_INPUT_DIR = os.path.join(RAW_DIR, "nelf_asr_ft_lcn", "combined")
COMBINED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "combined-cohort")
# Dirs/paths for ARCK data
ARCK_INPUT_DIR = os.path.join(INPUT_DIR, "arck")
# Dirs/paths for MC data
MC_INPUT_DIR = os.path.join(INPUT_DIR, "mc")
# Dirs/paths for  FPACK data
FPACK_INPUT_DIR = os.path.join(INPUT_DIR, "fpack")
FPACK_RAW_DIR = os.path.join(RAW_DIR, "fpack")
FPACK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "fpack")

# Overview of possible configs
FPACK_CONFIGS = [
    "q1_short_subject_wise",
    "q1_subject_wise",
    "q2_subject_wise",
    "q3_subject_wise",
    "q4_subject_wise",
    "q5_subject_wise",
    "subject_wise",
    # "question_wise", Leave out question_wise analyses
]
# Mappings between questions of the interviews and indicators in the respective file names
Q_FILE_MAPPINGS = {
    "q1_short_subject_wise": "bio_part1",
    "q1_subject_wise": "bio",
    "q2_subject_wise": "day",
    "q3_subject_wise": "actua",
    "q4_subject_wise": "object",
    "q5_subject_wise": "picture",
}
Q_FILE_MAPPING_NAMES = {
    "q1_short_subject_wise": "bio short",
    "q1_subject_wise": "bio",
    "q2_subject_wise": "day",
    "q3_subject_wise": "news",
    "q4_subject_wise": "nouns",
    "q5_subject_wise": "scene",
    "subject_wise": "all",
}

# Filepath for the custom huggingface data loading scripts
PACKAGE_DIR = os.path.join(PROJECT_DIR, "nlphumanbraintext")
PACKAGE_DATA_DIR = os.path.join(PACKAGE_DIR, "data")
LCN_PREPROCESSING_SCRIPT_PATH = os.path.join(
    PACKAGE_DATA_DIR,
    "preprocess_lcn_transcripts.py",
)

# Load constants from environment variables (set in the .env file) or set a default one
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or os.path.join(
    MODELS_DIR, "mlruns"
)
# Load the huggingface token env variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# Load a constant for distinguishing between local and cluster execution (default = True)
LOCAL_EXECUTION = (
    os.getenv("LOCAL_EXECUTION") if os.getenv("LOCAL_EXECUTION") is not None else "True"
)
# Constant for the maximum batch size that should be used during inference
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE")) or 4
# Huggingface access token for the Pred-based sentence embedding models
HF_TOKEN_PRED_BERT = os.getenv("HF_TOKEN_PRED_BERT")
# Directory for saving large datasets or language models from huggingface
LARGE_DATASET_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_HF_STORAGE_PATH") or "~/.cache/huggingface", "datasets"
)
LARGE_MODELS_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_HF_STORAGE_PATH") or "~/.cache/huggingface", "transformers"
)
LARGE_HUB_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_HF_STORAGE_PATH") or "~/.cache/huggingface", "hub"
)
HUB_MODEL_ID = os.getenv("HUB_MODEL_ID") or "nlp-xxl-fine-tuned"
LARGE_WHISPER_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_HF_STORAGE_PATH") or "~/.cache/huggingface", "whisper"
)

# Path to a default preprocessed AD vs amyloid negative control dataset
AD_AM_NEG_HC_DATASET_DIR = os.path.join(
    LARGE_DATASET_STORAGE_PATH,
    "ad_half_neg_hc_subject_wise_0"
)
# Path to the directory with the saved predictions
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Create directories and files
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FPACK_RAW_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(FPACK_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FPACK_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# NLP model types
FINE_TUNED_NLP_XXL_MODEL = "helena-balabin/robbert-2023-dutch-base-ft-nlp-xxl"
DUTCH_DEFAULT_MODEL = "DTAI-KULeuven/robbert-2023-dutch-large"
DUTCH_NER_DEFAULT = "wietsedv/bert-base-dutch-cased-finetuned-udlassy-ner"
DUTCH_SENTIMENT_ANALYSIS_DEFAULT = "wietsedv/bert-base-dutch-cased-finetuned-sentiment"
DUTCH_STOP_WORD_FILE = os.path.join(INPUT_DIR, "stopwords/stop_words_dutch.txt")
DUTCH_PUNCTUATION_MODEL = "oliverguhr/fullstop-dutch-punctuation-prediction"
LONGFORMER_BASE_MODEL = "markussagen/xlm-roberta-longformer-base-4096"
LONGFORMER_NLP_XXL_MODEL = "helena-balabin/nlp-xxl-xlm-r-longformer"
AD_HC_FT_SENT_EMBED_MODEL = os.path.join(
    LARGE_MODELS_STORAGE_PATH,
    "ad_hc_ft_q1_subject_wise_helena-balabin-nlp-xxl-stsb-xlm-r-multilingual",
    "checkpoint-920",
)
MODEL_LIST_AD_HC = [
    FINE_TUNED_NLP_XXL_MODEL,
    DUTCH_DEFAULT_MODEL,
    LONGFORMER_BASE_MODEL,
]
LABEL_DIR = os.path.join(RAW_DIR, "labels")
LABEL_METADATA_FILE = os.path.join(LABEL_DIR, "labels_meta.csv")
LABEL_METADATA_NORMS_FILE = os.path.join(LABEL_DIR, "labels_meta_npo_norms.csv")
FPACK_METADATA_FILE = os.path.join(LABEL_DIR, "Masterfile_FPACK_MR_2312.xlsx")

# Create a custom color palette
CUSTOM_COLOR_PALETTE = [
    "#FFC9B5",
    "#B2CEDE",
    "#8CDFD6",
    "#6DC0D5",
    "#416788",
    "#DBAD6A",
    "#C44536",
    "#A24B64",
    "#15616D",
    "#993955",
]

CUSTOM_HIST_COLOR_PALETTE = [
    "#7fcf99ff",
    "#ffdb58ff",
    "#9678b7ff",
    "#ff8166ff",
    "#6d8dbdff",
    "#c95f5fff",
]

# Brain characteristics name to scalar/vector mapping
BRAIN_CHARACTERISTICS_TYPES = {
    "amyloid_load": "scalar",
    "volumetry": "vector",
}

# Metrics used for classification
CLS_METRICS = [
    "accuracy", "f1_macro", "f1", "precision_macro", "precision", "npv", "recall_macro", "f1_micro",
    "precision_micro", "recall_micro", "recall", "specificity", "roc_auc", "sensitivity", "npv", "ppv",
    "sens_at_90_spec", "spec_at_90_sens",
]
MEDICAL_METRICS = [
    "auc", "lower", "upper", "sensitivity", "specificity", "ppv", "npv",
]
# lower and upper represent the confidence interval for the auc
ML_METRICS = [
    "accuracy", "f1", "precision", "recall",
]
