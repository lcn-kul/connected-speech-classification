"""Additional functions used in this repository."""

import re
from collections.abc import Iterable
from itertools import chain, zip_longest
from typing import Dict, List, Tuple, Union

import numpy as np
import mlflow
import pandas as pd
import scipy.stats as st
from datasets import (
    concatenate_datasets,
    Dataset,
)
from loguru import logger
from transformers import AutoTokenizer

from connected_speech_classification.constants import (
    DUTCH_DEFAULT_MODEL,
    Q_FILE_MAPPINGS,
)
from connected_speech_classification.visualization.roc_analysis import get_roc_auc_plot_cv
from connected_speech_classification.visualization.visualize_classification_metrics import plot_confusion_matrix

# Initialize a tokenizer for counting the chunk length before the first call to the function for faster inference
tokenizer = AutoTokenizer.from_pretrained(DUTCH_DEFAULT_MODEL)


def get_sim_vector(
    sim_matrix: Union[np.array, pd.DataFrame],
) -> np.array:
    """Get the upper triangle from a similarity matrix and vectorize it.

    :param sim_matrix: Similarity matrix from which to extract the upper triangle
    :type sim_matrix: np.array
    :return: Upper triangle (no diagonal) as a vector
    :rtype: np.array
    """
    if isinstance(sim_matrix, pd.DataFrame):
        sim_matrix = sim_matrix.values
    upper_triangle_values = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
    return upper_triangle_values


def describe_dataset_labels(
    dataset: Dataset,
    label: str = "ad_label",
) -> str:
    """Describe the proportion of classes in a given dataset.

    :param dataset: Dataset to describe
    :type dataset: Dataset
    :param label: Name of the label column, defaults to "ad_label"
    :type label: str, optional
    :return: String containing the dataset description
    :rtype: str
    """
    output_str = f"Dataset size: {len(dataset)}"
    output_str += f"\nNumber of subjects: {len(set(dataset['subject_id']))}"
    output_str += f"\nSubjects: {set(dataset['subject_id'])}"
    output_str += f"\nNumber of {label} = 1 subjects: {len(set(dataset.filter(lambda x: x[label] == 1)['subject_id']))}"
    output_str += f"\nNumber of {label} = 0 subjects: {len(set(dataset.filter(lambda x: x[label] == 0)['subject_id']))}"
    output_str += (f"\nProportion of {label} = 1 subjects: "
                   f"{len(dataset.filter(lambda x: x[label] == 1)) / len(dataset):.2f}")
    output_str += (f"\nProportion of {label} = 0 subjects: "
                   f"{len(dataset.filter(lambda x: x[label] == 0)) / len(dataset):.2f}")

    logger.info(output_str)

    return output_str


def flatten_batch(
    examples: Dict[str, List],
) -> Dict[str, List]:
    """Flatten a batch of subject - List of sentence pairs into subject - sentence pairs.

    :param examples: Batch of F-PACK data with the keys being the feature names and the values a list for all subjects
    :type examples: Dict[str, List]
    :return: Batch of "flattened" F-PACK examples - One sentence per subject instead of a list of sentences
    :rtype: Dict[str, List]
    """
    # Get the sentence key
    sentence_key = [key for key in examples.keys() if "sentence" in key][0]
    # Flatten the list of lists of sentences for all subjects:
    sentences_flattened = [item for sublist in examples[sentence_key] for item in sublist]
    # Initialize the results
    res = {sentence_key: sentences_flattened}
    # Add the other features to the results
    other_keys = [key for key in examples.keys() if key != sentence_key]

    # Repeat each other feature as many times as the number of sentences for that subject
    for k in other_keys:
        feauture = [
            [examples[k][i]] * len(examples[sentence_key][i]) for i in range(len(examples[k]))
        ]
        # Flatten the list of lists into one list
        feature_flattened = [item for sublist in feauture for item in sublist]
        # Add the feature to the results
        res[k] = feature_flattened

    return res


def flatten(
    xs: List,
) -> List: # type: ignore
    """Taken from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists."""
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def combine_sentences(
    example: Dict,
):
    """Combines all sentences into one for each subject."""
    column_names = [column for column in example.keys() if "sentences" in column]
    for column_name in column_names:
        example[column_name] = " ".join(example[column_name])
    return example


def generate_long_sentence_chunks_for_several_files(
    file_paths: List[str],
    max_length: int = 4096,
    replace_punctuation: bool = False,
    lowercase: bool = False,
) -> List[str]:
    """Generate long sentence chunks for several files for a Longformer-based model.

    :param file_paths: Paths to the files to be processed.
    :type file_paths: List[str]
    :param max_length: Maximum length of the chunks, defaults to 4096 words (approximation of tokens)
    :type max_length: int, optional
    :param replace_punctuation: Whether to replace the punctuation/insert if not present or not, defaults to True.
    :type replace_punctuation: bool
    :param lowercase: Whether to lowercase the text or not, defaults to False.
    :type lowercase: bool
    :return: A list of string chunks containing the text of all files.
    :rtype: List[str]
    """
    if replace_punctuation:
        punc_model = None

    chunks = []
    chunk = ""
    all_texts = []
    q_raw_sentences = []

    for file_path in file_paths:
        with open(file_path, "r") as f:
            text = f.read()
            # Remove any leading whitespace
            text = text.strip()
            # Remove <spk> from the text
            text = re.sub(r"<spk>", "", text)
            # Add a space at the very end to make sure the last sentence is also included
            text = text + " "
            if replace_punctuation:
                text = punc_model.restore_punctuation(text)  # noqa
            if lowercase:
                text = text.lower()
            all_texts.append(text)

        q_raw_sentences = re.findall(r".*?[.\?]", text)

        if len(q_raw_sentences) > 0:
            # Chunk sentences together
            for sent in q_raw_sentences:
                # Use a tokenizer here to count the actual number of tokens rather than words
                if len(tokenizer(chunk)["input_ids"]) + len(tokenizer(sent)["input_ids"]) < max_length:
                    # Putting the sentences back together into maximally large coherent chunks
                    chunk = chunk + sent
                else:
                    # If the maximum length is exceeded, append the existing chunk and
                    # start a new chunk with the current sentence
                    chunks.append(chunk)
                    chunk = sent

    # If there is no punctuation
    if len(q_raw_sentences) == 0:
        # Combine all_texts
        all_texts = " ".join(all_texts)
        # Split the text into chunks of max_length based on the number of tokens
        tok_splits = tokenizer(all_texts)["input_ids"][1:]
        chunks = [
            tokenizer.decode(tok_splits[i * max_length: (i + 1) * max_length]).strip(" </s>") for i in
            range(len(tok_splits) // max_length + 1)
        ]
        return chunks

    # Append the last chunk
    chunks.append(chunk)
    return chunks


def generate_sentence_chunks_per_file(
    file_path: str,
    max_length: int = 128,
    replace_punctuation: bool = False,
    lowercase: bool = False,
    multi_sentence: bool = True,
) -> List[str]:
    """Based on a text file, generate a list of sentences/multi-sentence chunks using the number of tokens.

    :param file_path: Path to the file
    :type file_path: str
    :param max_length: Maximum sequence length for each chunk/sentence
    :type max_length: int
    :param replace_punctuation: Whether to use a punctuation restoration model or not
    :type replace_punctuation: bool
    :param lowercase: Whether to lowercase the text (needed for ASR metrics)
    :type lowercase: bool
    :param multi_sentence: Whether to chunk multiple sentences into a multi-sentence or not, defaults to True
    :type multi_sentence: bool
    :return: List of processed chunks that either maximally exploit max_length or are grouped by sentences
    :rtype: List[str]
    """
    chunks = []
    if replace_punctuation:
        punc_model = None

    with open(file_path, encoding="utf-8") as f:
        text = f.read()
        # Remove any leading whitespace
        text = text.strip()
        # Remove <spk> from the text
        text = re.sub(r"<spk>", "", text)
        # Add a space at the very end to make sure the last sentence is also included
        text = text + " "
        # Some ASR models generate all uppercase output, so it is converted to lowercase in that case, or if lowercase
        # is explicitly desired
        if text.isupper() or lowercase:
            text = text.lower()

        # If there is no prior punctuation but also no restoration desired -> return evenly devided chunks
        if "." not in text and not replace_punctuation:
            # Chunk evenly based on the number of tokens (max_seq_len) if no punctuation is present
            # Use a tokenizer to count the number of tokens instead of the number of words
            tok_splits = tokenizer(text)["input_ids"][1:]
            chunks = [
                tokenizer.decode(tok_splits[i * max_length: (i + 1) * max_length]).strip(" </s>") for i in
                range(len(tok_splits) // max_length + 1)
            ]
            return chunks

        # Remove and replace all previously randomly inserted punctuation if needed
        if replace_punctuation:
            clean_text = text.replace(" . ", " ")
            # Use the punctuation restoration model here
            text = punc_model.restore_punctuation(clean_text)  # noqa

        # Split the text by period in all other cases
        q_raw_sentences = re.findall(r".*?[.\?] ", text)
        # Remove the extra whitespace for the last bit if there is any
        if len(q_raw_sentences) > 0:
            q_raw_sentences[-1] = q_raw_sentences[-1][:-1] if q_raw_sentences[-1][-1] == " " else q_raw_sentences[-1]
        chunk = ""

        # Group sentences into multisentences
        if multi_sentence:
            # Chunk sentences together
            for sent in q_raw_sentences:
                # Use a tokenizer here to count the actual number of tokens rather than words
                if len(tokenizer(chunk)["input_ids"]) + len(tokenizer(sent)["input_ids"]) < max_length:
                    # Putting the sentences back together into maximally large coherent chunks
                    chunk = chunk + sent  # + "." + " "
                else:
                    # If the maximum length is exceeded, append the existing chunk and
                    # start a new chunk with the current sentence
                    chunks.append(chunk)
                    chunk = sent  # + "."
            chunks.append(chunk)
        # Or keep them as single isolated sentences
        else:
            chunks = q_raw_sentences

        return chunks


def map_autobiographical_parts_to_combined(
    input_example: Dict,
) -> Dict:
    """Combine all parts of the autobiographical interview into one combined sentences input."""
    sentences_combined_unflattened = [
        input_example[key] for key in input_example.keys() if "sentences" in key
    ]
    return {
        "sentences_combined": [
            item for sublist in sentences_combined_unflattened for item in sublist
        ]
    }


def sort_fpack_files(
    file_names: List[str],
) -> List[str]:
    """Sort the regex/glob matches of the file names according to the order they would appear in the interview."""
    interview_order = [Q_FILE_MAPPINGS[key] for key in sorted(Q_FILE_MAPPINGS.keys())]
    matches = [
        sorted([f for f in file_names if interview_part in f])
        for interview_part in interview_order
    ]
    matches_flattened = [item for sublist in matches for item in sublist]
    return matches_flattened


def get_train_test_datasets(
    combined_datasets: Dict[str, Dataset],
    unique_subjects: List[str],
    train_idx: List[int],
    test_idx: List[int],
    batch_size: int,
    unique_subjects_all_datasets: Dict[str, List[str]],
) -> Tuple[Dataset, Dataset]:
    """Get training and test datasets based on the StratifiedKFold splits for the disease status classifier.

    :param combined_datasets: Dictionary of datasets to (eventually) combine
    :type combined_datasets: Dict[str, Dataset]
    :param unique_subjects: List of unique subject IDs
    :type unique_subjects: List[str]
    :param train_idx: Indices of the training subjects
    :type train_idx: List[int]
    :param test_idx: Indices of the test subjects
    :type test_idx: List[int]
    :param batch_size: Size of the batches to use
    :type batch_size: int
    :param unique_subjects_all_datasets: Dictionary of unique subject IDs for all datasets
    :type unique_subjects_all_datasets: Dict[str, List[str]]
    :return: Tuple of the training and test datasets
    :rtype: Tuple[Dataset, Dataset]
    """
    if len(combined_datasets) == 1:
        # Get the only element from the combined_datasets dictionary
        combined_dataset = list(combined_datasets.values())[0]
        # Define training and test subjects based on the StratifiedKFold split
        train_subjects = [unique_subjects[i] for i in train_idx]
        test_subjects = [unique_subjects[i] for i in test_idx]
        train_dataset = combined_dataset.filter(lambda x: x["subject_id"] in train_subjects)
        test_dataset = combined_dataset.filter(lambda x: x["subject_id"] in test_subjects)
    else:
        # Interleave batches for train and keep a dictionary with two separate sets for test for
        # the multiple datasets case
        train_dataset = []
        for name, specific_train_idx in zip(combined_datasets.keys(), train_idx, strict=False):
            train_subjects = [unique_subjects_all_datasets[name][i] for i in specific_train_idx]
            train_dataset.append(combined_datasets[name].filter(lambda x: x["subject_id"] in train_subjects))

        # Instead of directly interleaving the datasets, which would interleave the individual examples,
        # we want to interleave batches, so that each batch is from one dataset
        # For that we split the dataset into smaller datasets of size batch_size
        train_dataset = [
            [
                dataset.select(range(i, min(i + batch_size, len(dataset))))
                for i in range(0, len(dataset), batch_size)
            ]
            for dataset in train_dataset
        ]
        # Put the last examples that are shorter than the batch size aside
        last_examples = [dataset.pop() for dataset in train_dataset]
        # Then we interleave them, first create a list altnating between the two elements in the list
        train_dataset = [x for x in chain(*zip_longest(train_dataset[0], train_dataset[1])) if x is not None]
        # Put the examples that are shorter than the batch size at the end of the list
        train_dataset += last_examples

        # Then concatenate the list of datasets
        train_dataset = concatenate_datasets(train_dataset)

        test_dataset = {}
        for name, specific_test_idx in zip(combined_datasets.keys(), test_idx, strict=False):
            test_subjects = [unique_subjects_all_datasets[name][i] for i in specific_test_idx]
            test_dataset[name] = combined_datasets[name].filter(lambda x: x["subject_id"] in test_subjects)

    return train_dataset, test_dataset


def log_data_split_specific_auc_metrics(
    task: str,
    level: str,
    fold_metrics: Dict,
    split: int = 0,
    disease_status: str = "ad_hc",
):
    """Log data split specific AUC-based metrics for the disease status classifier.

    :param task: Task for which to log the metrics
    :type task: str
    :param level: Level for which to log the metrics
    :type level: str
    :param fold_metrics: Dictionary of metrics for the fold
    :type fold_metrics: Dict
    :param split: Split number
    :type split: int
    :param disease_status: Disease status for which to log the metrics
    :type disease_status: str
    """
    # Get the confidence interval for the ROC AUC
    ci = st.t.interval(
        confidence=0.95,
        df=len(fold_metrics[f"{task}{level}_roc_auc"])-1,
        loc=np.mean(fold_metrics[f"{task}{level}_roc_auc"]),
        scale=st.sem(fold_metrics[f"{task}{level}_roc_auc"]),
    )
    mlflow.log_metric(f"mean_ci_{task}{level}_auc_lower", ci[0])
    mlflow.log_metric(f"mean_ci_{task}{level}_auc_upper", ci[1])
    # Don't create an average of "all_fpr"/"all_tpr", but use them as input for the ROC curve plot
    mean_fpr, mean_tpr, fig = get_roc_auc_plot_cv(
        fold_metrics[f"{task}{level}_all_fpr"],
        fold_metrics[f"{task}{level}_all_tpr"],
    )
    mlflow.log_figure(fig, f"{task}{level}_roc_curve.png")
    _, _, fig_without_folds = get_roc_auc_plot_cv(
        fold_metrics[f"{task}{level}_all_fpr"],
        fold_metrics[f"{task}{level}_all_tpr"],
        plot_all_folds=False,
    )
    mlflow.log_figure(fig_without_folds, f"{task}{level}_roc_curve_without_folds.png")
    # Only use the average all_fpr and all_tpr for later statistical comparisons (de long method),
    # log them as a param
    mlflow.log_param(f"mean_{task}{level}_all_fpr_{split}", mean_fpr)
    mlflow.log_param(f"mean_{task}{level}_all_tpr_{split}", mean_tpr)

    # Plot confusion matrix and also optimal one
    conf_matrix = fold_metrics[f"{task}{level}_conf_matrix"]
    optimal_conf_matrix = fold_metrics[f"{task}{level}_optimal_conf_matrix"]
    # Get the labels for the confusion matrix
    if disease_status == "joint_ad_amyloid":
        labels = ["amyloid- CU", "amyloid+ CU"] if task == "ad" else ["amyloid- CU", "prodromal AD"]
    elif disease_status == "amyloid":
        labels = ["amyloid- CU", "amyloid+ CU"]
    elif disease_status == "baseline":
        labels = ["amyloid- CU #1", "amyloid- CU #2"]
    else:
        labels = ["amyloid- CU", "prodromal AD"]

    for matrix, name in zip([conf_matrix, optimal_conf_matrix], ["conf_matrix", "optimal_conf_matrix"], strict=False):
        conf_matrix = plot_confusion_matrix(
            matrix,
            labels,
        )
        mlflow.log_figure(conf_matrix, f"{task}{level}_{name}.png")
