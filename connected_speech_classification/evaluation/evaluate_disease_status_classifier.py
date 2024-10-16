"""Some helper functions to evaluate a disease status classifier."""
# Imports
from typing import Dict, Union

import numpy as np
from datasets import Dataset
from evaluate import load
from transformers import EvalPrediction
from scipy.special import softmax
from sklearn.metrics import roc_curve

# Initialize some metrics from the evaluate library
acc = load("accuracy")
f1 = load("f1")
precision = load("precision")
recall = load("recall")
roc_auc = load("roc_auc")
youden_index = load("helena-balabin/youden_index")
conf_matrix = load("confusion_matrix")

def get_prediction_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Get the prediction metrics.

    :param logits: Predicted logits for the labels
    :type logits: np.ndarray
    :param labels: True labels
    :type labels: np.ndarray
    :return: Dictionary with variants of precision, f1 and accuracy, recall, specificity etc.
    :rtype: Dict[str, float]
    """
    predictions = np.argmax(logits, axis=-1)
    probabilities_pos_class = softmax(logits, axis=-1)[:, 1]
    # Get all fpr, tpr values for the roc curve
    fpr, tpr, _ = roc_curve(y_true=labels, y_score=probabilities_pos_class, drop_intermediate=False)
    youden_metrics = youden_index.compute(prediction_scores=probabilities_pos_class, references=labels)
    optimal_predictions = np.where(probabilities_pos_class >= youden_metrics["youden_threshold"], 1, 0)

    # Find the first tpr (sensitivity) that is greater than 0.9 and find the corresponding specificity
    spec_at_90_sens = 1 - fpr[np.where(tpr >= 0.9)[0][0]]
    # Do it the other way around
    sens_at_90_spec = tpr[np.where(1 - fpr >= 0.9)[0][-1]]

    return {
        # ML metrics
        "accuracy": acc.compute(predictions=predictions, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "f1_micro": f1.compute(predictions=predictions, references=labels, average="micro")["f1"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"],
        "precision_macro": precision.compute(
            predictions=predictions,
            references=labels,
            average="macro",
        )["precision"],
        "precision_micro": precision.compute(
            predictions=predictions,
            references=labels,
            average="micro",
        )["precision"],
        "precision": precision.compute(predictions=predictions, references=labels)["precision"],
        "recall_macro": recall.compute(predictions=predictions, references=labels, average="macro")["recall"],
        "recall_micro": recall.compute(predictions=predictions, references=labels, average="micro")["recall"],
        "recall": recall.compute(predictions=predictions, references=labels)["recall"],
        # Medical metrics: ROC AUC, sensitivity, specificity, NPV, PPV, Youden's index
        **youden_metrics,
        # All TPR, FPR values for plotting the ROC curve later on
        "all_fpr": fpr,
        "all_tpr": tpr,
        # Sensitivity at 90% specificity and vice versa
        "sens_at_90_spec": sens_at_90_spec,
        "spec_at_90_sens": spec_at_90_sens,
        # Confusion matrices
        "conf_matrix": conf_matrix.compute(predictions=predictions, references=labels)["confusion_matrix"],
        "optimal_conf_matrix": conf_matrix.compute(
            predictions=optimal_predictions,
            references=labels,
        )["confusion_matrix"],
        # Log the predictions, optimal predictions and the labels
        "predictions": predictions,
        "optimal_predictions": optimal_predictions,
        "labels": labels,
    }


def compute_metrics(
    eval_pred: EvalPrediction,
    *,
    subject_labels: Union[Dataset, Dict],
) -> Dict[str, float]:
    """Compute a bunch of metrics from the evaluate library.

    :param eval_pred: Evaluation predictions
    :type eval_pred: EvalPrediction
    :param subject_labels: List of subject labels
    :type subject_labels: Union[Dataset, Dict]
    :return: Dictionary with variants of precision, f1 and accuracy, recall, specificity and NPV
    Dict[str, float]
    """
    # Get the subject_labels: See which type this is
    if isinstance(subject_labels, Dataset):
        subject_labels = subject_labels["subject_id"]
    else:
        # Otherwise it is a dictionary of two datasets. Find the one that has the same length
        # as eval_pred[0]
        right_dataset = [v for v in subject_labels.values() if len(v) == len(eval_pred[0])][0]
        subject_labels = right_dataset["subject_id"]

    # 1) Item-level evaluation
    item_logits, item_labels = eval_pred

    # 2) Subject-level evaluation: Aggregate multiple predictions for the same subject by averaging
    subject_level_logits = []
    subject_level_labels = []
    subject_level_ids = []
    for subject in sorted(set(subject_labels)):
        subject_specific_logits = item_logits[np.where(np.array(subject_labels) == subject)]
        subject_specific_labels = item_labels[np.where(np.array(subject_labels) == subject)]
        subject_level_logits.append(np.mean(subject_specific_logits, axis=0))
        subject_level_labels.append(subject_specific_labels[0])
        subject_level_ids.append(subject)
    subject_level_logits = np.array(subject_level_logits)
    subject_level_labels = np.array(subject_level_labels)

    # Initialize the result dictionary
    result = {}

    # Get both 1) item-level and 2) subject-level metrics
    for labels, logits, metric_prefix in zip(
        [item_labels, subject_level_labels], [item_logits, subject_level_logits], ["item", "subject"], strict=False
    ):
        metrics = get_prediction_metrics(logits, labels)
        metrics = {f"{metric_prefix}_{k}": v for k, v in metrics.items()}
        # Add subject ids
        metrics[f"{metric_prefix}_subject_ids"] = subject_level_ids
        result.update(metrics)

    return result
