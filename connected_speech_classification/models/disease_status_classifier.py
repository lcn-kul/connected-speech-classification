"""Fine-tuned embedding model for classification."""

# Imports
import os
from typing import Dict, List, Optional

import click
import mlflow
import numpy as np
import pandas as pd
from datasets import (
    load_from_disk,
    Value,
)
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback

from connected_speech_classification.constants import (
    AD_AM_NEG_HC_DATASET_DIR,
    CLS_METRICS,
    FINE_TUNED_NLP_XXL_MODEL,
    LABEL_METADATA_NORMS_FILE,
    LARGE_MODELS_STORAGE_PATH,
    LOGS_DIR,
    MLFLOW_TRACKING_URI,
)
from connected_speech_classification.evaluation.evaluate_disease_status_classifier import compute_metrics
from connected_speech_classification.models.multi_task_classification_model import (
    MultiTaskSequenceClassificationModel,
)
from connected_speech_classification.utils import (
    describe_dataset_labels,
    log_data_split_specific_auc_metrics,
    get_train_test_datasets,
)
from connected_speech_classification.visualization.visualize_classification_metrics import plot_confusion_matrix


@click.command()
@click.option("--classification_model_base", type=str, default=FINE_TUNED_NLP_XXL_MODEL)
@click.option("--dataset_dir", multiple=True, default=[AD_AM_NEG_HC_DATASET_DIR])
@click.option("--config", type=str, default="q1_subject_wise")
@click.option("--classify_amyloid", is_flag=True, default=False)
@click.option("--classify_jointly", is_flag=True, default=False)
@click.option("--classify_baseline", is_flag=True, default=False)
@click.option("--k_folds", type=int, default=5)
@click.option("--max_seq_len", type=int, default=512)
@click.option("--batch_size", type=int, default=16)
@click.option("--gradient_accumulation_steps", type=int, default=1)
@click.option("--early_stopping", is_flag=True, default=False)
@click.option("--early_stopping_patience", type=int, default=10)
@click.option("--early_stopping_metric", type=str, default="loss")
@click.option("--epochs", type=int, default=100)
@click.option("--mlflow_tracking_uri", type=str, default=MLFLOW_TRACKING_URI)
@click.option("--logging_path", type=str, default=LOGS_DIR)
@click.option("--model_output_dir", type=str, default=LARGE_MODELS_STORAGE_PATH)
@click.option("--n_data_splits", type=int, default=5)
@click.option("--freeze_most_layers", is_flag=True, default=False)
@click.option("--labels_meta_npo_file", type=str, default=LABEL_METADATA_NORMS_FILE)
@click.option("--debug", is_flag=True, default=False)
def classify_disease_label(
    classification_model_base: str = FINE_TUNED_NLP_XXL_MODEL,
    dataset_dir: Optional[List[str]] = None,
    config: str = "q1_subject_wise",
    classify_amyloid: bool = False,
    classify_jointly: bool = False,
    classify_baseline: bool = False,
    k_folds: int = 5,
    max_seq_len: int = 512,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = "recall_sensitivity",
    epochs: int = 100,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    logging_path: str = LOGS_DIR,
    model_output_dir: str = LARGE_MODELS_STORAGE_PATH,
    n_data_splits: int = 5,
    freeze_most_layers: bool = False,
    labels_meta_npo_file: str = LABEL_METADATA_NORMS_FILE,
    debug: bool = False,
) -> None:
    """Fine-tune a given embedding model for AD vs HC classification.

    :param classification_model_base: Name/huggingface identifier of the model that should be fine-tuned,
        defaults to LONGFORMER_NLP_XXL_MODEL
    :type classification_model_base: str
    :param dataset_dir: Path to the directories where the preprocessed datasets are stored
    :type dataset_dir: Optional[List[str]]
    :param config: Name of the train configuration that should be used to preprocess the data,
        defaults to "q1_subject_wise"
    :type config: str
    :param classify_amyloid: Whether to classify the amyloid status instead of of AD vs HC, defaults to False
    :type classify_amyloid: bool
    :param classify_jointly: Whether to classify AD vs HC and amyloid status jointly, defaults to False
    :type classify_jointly: bool
    :param classify_baseline: Whether to classify between two groups of amyloid negative datasets as a baseline,
        defaults to False
    :type classify_baseline: bool
    :param k_folds: Number of folds that should be used for cross validation, defaults to 5
    :type k_folds: int
    :param max_seq_len: Maximum sequence length that should be used for the model/tokenizer, defaults to 4096
    :type max_seq_len: int
    :param batch_size: Batch size that should be used for training/testing, defaults to 16
    :type batch_size: int
    :param gradient_accumulation_steps: Number of gradient accumulation steps that should be used for training,
        defaults to 1
    :type gradient_accumulation_steps: int
    :param early_stopping: Whether early stopping should be used, defaults to False
    :type early_stopping: bool
    :param early_stopping_patience: Number of steps used for early stopping, defaults to 10
    :type early_stopping_patience: int
    :param early_stopping_metric: Metric that should be used for early stopping, defaults to "recall_sensitivity"
    :type early_stopping_metric: str
    :param epochs: Number of maximum steps that should be used for training, defaults to 100
    :type epochs: int
    :param mlflow_tracking_uri: URI of the MLflow tracking server, defaults to MLFLOW_TRACKING_URI
    :type mlflow_tracking_uri: str
    :param logging_path: Path to the directory where the logs should be stored, defaults to LOGS_DIR
    :type logging_path: str
    :param model_output_dir: Path to the directory where the model should be stored,
        defaults to LARGE_MODELS_STORAGE_PATH
    :type model_output_dir: str
    :param n_data_splits: Number of data splits of the amyloid negative group that should be evaluated in the experiment,
        not equal to the cv splits, defaults to 5
    :type n_data_splits: int
    :param freeze_most_layers: Whether to freeze most layers of the model (everything apart from last Transformer layer),
        and the classification head(s), defaults to False
    :type freeze_most_layers: bool
    :param labels_meta_npo_file: Path to the labels metadata file with the NPO norms, defaults to LABEL_METADATA_NORMS_FILE
    :type labels_meta_npo_file: str
    :param debug: Whether to run the script in debug mode, defaults to False
    :type debug: bool
    """
    # Check whether the model is a longformer model
    if dataset_dir is None:
        dataset_dir = [AD_AM_NEG_HC_DATASET_DIR]
    longformer = "longformer" in classification_model_base
    # Check if amyloid status should be classified, or if joint classification should be performed
    disease_status = "ad_hc"
    if classify_amyloid:
        disease_status = "amyloid"
    if classify_jointly:
        disease_status = "joint_ad_amyloid"
    if classify_baseline:
        disease_status = "baseline"

    # Split by tasks for multi-task learning
    tasks = ["ad_", "amyloid_"] if classify_jointly else [""]

    # If the dataset is a directory (i.e., a previously trained
    # model rather than a pretrained one from the huggingface model hub),
    # use a random data split to get the tokenizer
    tokenizer_base = None
    if os.path.isdir(classification_model_base):
        tokenizer_base = classification_model_base + "_0"
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_base if tokenizer_base else classification_model_base,
    )

    # Initialize the cross-validation folds
    folds = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Setup MLflow experiment tracking for tracking the metrics
    debug_flag = "debug_" if debug else ""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=f"{debug_flag}{disease_status}_cls")

    # Create an output folder for the predictions in the model output directory
    top_level_predictions_dir = os.path.join(
        model_output_dir,
        f"{config}_{classification_model_base.replace('/', '-')}_predictions",
    )
    os.makedirs(top_level_predictions_dir, exist_ok=True)

    # Load the labels metadata needed for the evaluation later on
    labels_meta_npo = pd.read_csv(labels_meta_npo_file)

    # In total, there are 3 levels of nested runs: top-level configuration/interview part specific,
    # data split specific, and fold specific
    # 1) Start the top-level configuration-specific experiment run
    with mlflow.start_run(run_name=f"{config}_{classification_model_base}"):
        # Record top-level metrics (averaged across data splits)
        data_split_metrics: Dict = {}
        for metric in CLS_METRICS:
            for task in tasks:
                for level in ["subject", "item"]:
                    data_split_metrics[f"{task}{level}_{metric}"] = []

        # Iterate over different splits of the amyloid negative group
        for i in range(n_data_splits):
            # Create output directories if they don't exist
            os.makedirs(
                os.path.join(
                    model_output_dir,
                    f"{disease_status}_ft_{config}_{classification_model_base.replace('/', '-')}_{i}",
                ),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(
                    logging_path,
                    f"{disease_status}_ft_{config}_{classification_model_base.replace('/', '-')}_{i}",
                ),
                exist_ok=True,
            )

            # Define some helper variables
            combined_datasets = {}
            unique_subjects_all_datasets = {}
            splits: List = []

            for data_path in dataset_dir:
                # Modify the dataset path to include the data split
                data_path = data_path + f"_{i}"
                combined_dataset = load_from_disk(data_path)
                # Get the correct label depending on whether amyloid status should be classified or AD vs HC or
                # baseline classification (amyloid neg group 1 vs amyloid neg group 2)
                if "baseline" in data_path or "imb" in data_path:
                    label_key = "group_label"
                elif "amyloid" in data_path:
                    label_key = "amyloid_label"
                else:
                    label_key = "ad_label"

                # Rename the label key
                combined_dataset = combined_dataset.rename_column(label_key, "label")
                # Cast the label to int
                combined_dataset = combined_dataset.cast_column("label", Value(dtype="int32"))
                combined_dataset = combined_dataset.class_encode_column("label")
                # Remove the other label
                combined_dataset = combined_dataset.remove_columns("ad_label" if "amyloid" in data_path else "amyloid_label")

                # Add a task id to the dataset
                combined_dataset = combined_dataset.map(
                    lambda x: {"task_ids": 0 if "ad" in label_key else 1, **x},
                )

                # Get the sentence key
                sent_key = [k for k in combined_dataset.features.keys() if "subject_id" not in k][0]
                # Tokenize the dataset
                combined_dataset = combined_dataset.map(
                    lambda x: tokenizer(
                        x[sent_key],
                        padding="max_length",
                        max_length=max_seq_len,
                        truncation=True,
                        return_tensors="pt",
                    ),
                    batched=True,
                )
                # Shuffle the dataset
                combined_dataset = combined_dataset.shuffle(seed=42)
                # Append the dataset to the list
                combined_datasets.update({label_key.replace(r"_label", ""): combined_dataset})

                # Get the unique subjects and their labels, there are multiple instances per subject
                unique_subjects = combined_dataset.unique("subject_id")
                unique_subjects_all_datasets[label_key.replace(r"_label", "")] = unique_subjects
                # Get the labels for these subjects
                subject_labels = [
                    combined_dataset.filter(lambda x: x["subject_id"] == subject)[0]["label"] for subject in unique_subjects
                ]
                if len(splits) == 0:
                    splits = list(folds.split(np.zeros(len(unique_subjects)), subject_labels))
                # If there are multiple datasets, we need to create the splits for each dataset and put them together
                else:
                    additional_splits = list(folds.split(np.zeros(len(unique_subjects)), subject_labels))
                    # Zip the splits and the additional splits, so that the result is a list of
                    # ((train_idx_sp, train_idx_add), (test_idx_sp, test_idx_add))
                    splits = list(zip(splits, additional_splits, strict=False))
                    splits = [((split[0][0], split[1][0]), (split[0][1], split[1][1])) for split in splits]

            # Record metrics averaged across folds for each data split configuration
            fold_metrics: Dict = {}
            for metric in CLS_METRICS + ["predictions", "optimal_predictions", "labels", "subject_ids"]:
                for task in tasks:
                    for level in ["subject", "item"]:
                        fold_metrics[f"{task}{level}_{metric}"] = []
                        if "auc" in metric:
                            # Also store the fpr and tpr for the ROC curve plot
                            fold_metrics[f"{task}{level}_all_fpr"] = [None] * k_folds
                            fold_metrics[f"{task}{level}_all_tpr"] = [None] * k_folds
                            # Also store the predictions for the confusion matrix
                            fold_metrics[f"{task}{level}_conf_matrix"] = np.zeros((2, 2))
                            fold_metrics[f"{task}{level}_optimal_conf_matrix"] = np.zeros((2, 2))

            # 2) Start the data split-specific experiment run
            with mlflow.start_run(run_name=f"data_split_{i}", nested=True):
                # Create a data-split specific output directory for the predictions
                data_split_predictions_dir = os.path.join(top_level_predictions_dir, f"data_split_{i}")
                os.makedirs(data_split_predictions_dir, exist_ok=True)

                mlflow.log_param("data_split", i)
                # Log the datasets
                for idx, combined_dataset in enumerate(combined_datasets.values()):
                    mlflow.log_artifacts(f"{dataset_dir[idx]}_{i}")
                    dataset_description = describe_dataset_labels(combined_dataset, "label")
                    mlflow.log_text(dataset_description, f"dataset_description_{idx}_{i}.txt")

                # Iterate over folds
                for fold, (train_idx, test_idx) in enumerate(splits):
                    # 3) Start the fold-specific experiment run
                    with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                        # Split the dataset, filter by the subjects defined in train/test
                        train_dataset, test_dataset = get_train_test_datasets(
                            combined_datasets=combined_datasets,
                            unique_subjects=unique_subjects,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            batch_size=batch_size,
                            unique_subjects_all_datasets=unique_subjects_all_datasets,
                        )

                        # Define the callback(s)
                        callbacks = [MLflowCallback()]
                        if early_stopping:
                            # Set up an early stopping callback (optional)
                            early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
                            callbacks.append(early_stopping_callback)

                        # If the dataset is a directory (i.e., a previously trained model
                        # rather than a pretrained one from the huggingface model hub),
                        # append the data split index to the model output directory
                        classification_model_base_with_index = None
                        if os.path.isdir(classification_model_base):
                            classification_model_base_with_index = classification_model_base + f"_{i}"

                        # Initialize a custom model with two (shared) classification heads for joint classification
                        if classify_jointly:
                            model_config = AutoConfig.from_pretrained(
                                classification_model_base_with_index
                                if classification_model_base_with_index
                                else classification_model_base
                            )
                            model_config.num_labels = 2
                            model_config.add_pooling_layer = False

                            model = MultiTaskSequenceClassificationModel(
                                classification_model_base_with_index
                                if classification_model_base_with_index
                                else classification_model_base,
                                config=model_config,
                                cache_dir=model_output_dir,
                            )
                        # Or initialize a model with a single head for classification
                        else:
                            model = AutoModelForSequenceClassification.from_pretrained(
                                classification_model_base_with_index
                                if classification_model_base_with_index
                                else classification_model_base,
                                num_labels=2,
                                cache_dir=model_output_dir,
                            )

                        # Freeze most layers of the model (except for last transformer + cls layers), if specified
                        if freeze_most_layers:
                            # Get the number of transformer layers
                            layers = model.config.num_hidden_layers
                            for name, param in model.named_parameters():
                                if "classifier" in name:
                                    param.requires_grad = True
                                elif str(layers - 1) in name:
                                    param.requires_grad = True
                                else:
                                    param.requires_grad = False

                        # Initialize the trainer
                        trainer = Trainer(
                            model=model,
                            args=TrainingArguments(
                                output_dir=os.path.join(
                                    model_output_dir,
                                    f"{os.path.basename(dataset_dir[0])}_{disease_status}_ft_{config}_"
                                    f"{classification_model_base.replace('/', '-')}_{i}",
                                ),
                                overwrite_output_dir=True,
                                num_train_epochs=epochs,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                eval_accumulation_steps=len(test_dataset) // batch_size + 1,
                                # This is to make sure the evaluation is done on the entire set at once
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                warmup_ratio=0.1,
                                logging_dir=os.path.join(
                                    logging_path,
                                    f"{os.path.basename(dataset_dir[0])}_{disease_status}_ft_{config}_"
                                    f"{classification_model_base.replace('/', '-')}_{i}",
                                ),
                                logging_steps=5,
                                evaluation_strategy="steps",
                                eval_steps=20,
                                save_strategy="no",
                                fp16=True if longformer else False,  # Use mixed precision training for longformer models
                                load_best_model_at_end=True if early_stopping else False,
                                metric_for_best_model="eval_amyloid_loss" if classify_jointly else early_stopping_metric,
                                group_by_length=True if classify_jointly else False,
                                # Improvised way to group the tasks into the batches
                                length_column_name="task_ids" if classify_jointly else "length",
                                # Improvised way to group the tasks into the batches
                                report_to="mlflow",
                            ),
                            train_dataset=train_dataset,
                            eval_dataset=test_dataset,
                            tokenizer=tokenizer,
                            compute_metrics=lambda x: compute_metrics(x, subject_labels=test_dataset),
                            callbacks=callbacks,
                        )
                        # Train the model
                        trainer.train()
                        # Log fold-specific metrics
                        metrics = trainer.evaluate()
                        # Log all metrics on each fold
                        for metric in CLS_METRICS + ["predictions", "optimal_predictions", "labels", "subject_ids"]:
                            for task in tasks:
                                for level in ["subject", "item"]:
                                    fold_metrics[f"{task}{level}_{metric}"].append(metrics[f"eval_{task}{level}_{metric}"])
                                    if "auc" in metric:
                                        fold_metrics[f"{task}{level}_all_fpr"][fold] = metrics[f"eval_{task}{level}_all_fpr"]
                                        fold_metrics[f"{task}{level}_all_tpr"][fold] = metrics[f"eval_{task}{level}_all_tpr"]
                                        fold_metrics[f"{task}{level}_conf_matrix"] += metrics[f"eval_{task}{level}_conf_matrix"]
                                        fold_metrics[f"{task}{level}_optimal_conf_matrix"] += metrics[
                                            f"eval_{task}{level}_optimal_conf_matrix"
                                        ]

                        # Save the model once at the end
                        trainer.save_model()

                # Log the mean metrics across all folds (for a given data split)
                for metric in CLS_METRICS:
                    for task in tasks:
                        for level in ["subject", "item"]:
                            # Make sure to also log the log-specific metrics to allow for a t-test later on
                            mlflow.log_param(f"all_{task}{level}_{metric}_{i}", fold_metrics[f"{task}{level}_{metric}"])
                            # Log the mean
                            mlflow.log_metric(f"mean_{task}{level}_{metric}", np.mean(fold_metrics[f"{task}{level}_{metric}"]))
                            if "auc" in metric:
                                # Log data-split specific AUC-based metrics (only once per task/level)
                                log_data_split_specific_auc_metrics(
                                    task=task,
                                    level=level,
                                    fold_metrics=fold_metrics,
                                    split=i,
                                    disease_status=disease_status,
                                )
                            # Log them for the data split (for the average later)
                            data_split_metrics[f"{task}{level}_{metric}"].append(fold_metrics[f"{task}{level}_{metric}"])

                # Create a data frame with the predictions, optional predictions, subject_ids and labels for each fold
                # based on fold_metrics (only on subject-level)
                for task in tasks:
                    task_data = {}
                    for metric in ["predictions", "optimal_predictions", "labels", "subject_ids"]:
                        # Get the data
                        data = [fold_metrics[f"{task}subject_{metric}"][i] for i in range(k_folds)]
                        # Flatten the data if it is a list of lists
                        if isinstance(data[0], list):
                            data = [item for sublist in data for item in sublist]
                        # Also flatten a list of np arrays
                        if isinstance(data[0], np.ndarray):
                            data = np.concatenate(data).tolist()
                        task_data[metric] = data
                    # Add the cross-validation split
                    task_data["cv_split"] = [[i] * len(fold_metrics[f"{task}subject_predictions"][i]) for i in range(k_folds)]
                    # Flatten the cv_split
                    task_data["cv_split"] = [item for sublist in task_data["cv_split"] for item in sublist]  # type: ignore
                    # Create a data frame
                    task_df = pd.DataFrame(task_data)
                    # Rename subject_ids to subject_id
                    task_df = task_df.rename(columns={"subject_ids": "subject_id"})
                    # Merge the data frame with the labels metadata
                    task_df = task_df.merge(labels_meta_npo, on="subject_id")

                    # Create a confusion matrix: First get the labels for the confusion matrix
                    if disease_status == "baseline":
                        task_label = "Amyloid- CU"
                    elif disease_status == "joint_ad_amyloid":
                        task_label = "Amyloid+ CU" if "amyloid" in task else "Prodromal AD"
                    elif disease_status == "amyloid":
                        task_label = "Amyloid+ CU"
                    else:
                        task_label = "Prodromal AD"

                    # Create a confusion matrix for correct predictions vs the one/two/three_norm_below_min_1 column
                    for norm_col in ["one_norm_below_min_1", "two_norms_below_min_1", "three_norms_below_min_1"]:
                        for preds_col in ["predictions", "optimal_predictions"]:
                            confusion_matrix = pd.crosstab(
                                task_df[preds_col],
                                task_df[norm_col],
                                rownames=[f"Classified as {task_label}"],
                                colnames=["Low Language Scores"],
                            )
                            # If the confusion matrix is not 2x2 use reindex to add the missing columns/rows
                            if confusion_matrix.shape != (2, 2):
                                confusion_matrix = confusion_matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                            # Plot the confusion matrix
                            conf_plot = plot_confusion_matrix(
                                matrix=confusion_matrix.values,
                                labels=[
                                    "False",
                                    "True",
                                ],
                                xlabel="Low Language Scores",
                                ylabel=f"Classified as {task_label}",
                            )
                            # Log the confusion matrix
                            mlflow.log_figure(conf_plot, f"{task}{norm_col}_{preds_col}_confusion_matrix.png")

                        # Also create a confusion matrix for label vs
                        # the one/two/three_norm_below_min_1 column
                        # Create a confusion matrix
                        confusion_matrix_ground_truth = pd.crosstab(
                            task_df["labels"],
                            task_df[norm_col],
                            rownames=[task_label],
                            colnames=["Language Deficit"],
                        )
                        # If the confusion matrix is not 2x2 use reindex to add the missing columns/rows
                        if confusion_matrix_ground_truth.shape != (2, 2):
                            confusion_matrix_ground_truth = confusion_matrix_ground_truth.reindex(
                                index=[0, 1],
                                columns=[0, 1],
                                fill_value=0,
                            )
                        # Plot the confusion matrix
                        conf_plot_ground_truth = plot_confusion_matrix(
                            matrix=confusion_matrix_ground_truth.values,
                            labels=[
                                "False",
                                "True",
                            ],
                            xlabel="Language Deficit",
                            ylabel=task_label,
                        )
                        # Log the confusion matrix
                        mlflow.log_figure(conf_plot_ground_truth, f"{task}{norm_col}_ground_truth_confusion_matrix.png")

                    # Log the data frame as an artifact
                    predictions_path = os.path.join(data_split_predictions_dir, f"{task}predictions.csv")
                    task_df.to_csv(predictions_path, index=False)
                    mlflow.log_artifact(predictions_path)

        # Log the mean metrics across all data splits (for a given configuration)
        for metric in CLS_METRICS:
            for task in tasks:
                for level in ["subject", "item"]:
                    # Log the mean
                    mlflow.log_metric(f"mean_{task}{level}_{metric}", np.mean(data_split_metrics[f"{task}{level}_{metric}"]))
                    if "auc" in metric:
                        # "Bootstrap SEM": Derive the SEM of each data split and then take the mean SEM
                        # Take the SEM of the AUCs in each data split
                        sems = [np.std(aucs) / np.sqrt(len(aucs)) for aucs in data_split_metrics[f"{task}{level}_{metric}"]]
                        # Report the mean SEM
                        mlflow.log_metric(f"mean_{task}{level}_{metric}_sem", np.mean(sems))
                        # Report upper/lower 95% CI based on the SEM
                        mlflow.log_metric(
                            f"mean_{task}{level}_{metric}_ci_upper",
                            np.mean(data_split_metrics[f"{task}{level}_{metric}"]) + 1.96 * np.mean(sems),
                        )
                        mlflow.log_metric(
                            f"mean_{task}{level}_{metric}_ci_lower",
                            np.mean(data_split_metrics[f"{task}{level}_{metric}"]) - 1.96 * np.mean(sems),
                        )

        # Log some parameters
        mlflow.log_param("disease_status", disease_status)
        mlflow.log_param("classification_model_base", classification_model_base)
        mlflow.log_param("config", config)
        mlflow.log_param("k_folds", k_folds)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("gradient_accumulation_steps", gradient_accumulation_steps)


@click.group()
def cli() -> None:
    """Fine-tune a specified embedding model for classifying the disease status."""


if __name__ == "__main__":
    cli.add_command(classify_disease_label)
    cli()
