"""Prepare datasets that can be used for disease status classification."""
import click
import pandas as pd
from datasets import concatenate_datasets, load_dataset, DownloadMode
from loguru import logger
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split

from connected_speech_classification.constants import (
    COMBINED_INPUT_DIR,
    FPACK_CONFIGS,
    LABEL_METADATA_FILE,
    LARGE_DATASET_STORAGE_PATH,
    LCN_PREPROCESSING_SCRIPT_PATH,
)
from connected_speech_classification.utils import flatten_batch, describe_dataset_labels


@click.command()
@click.option("--combined_dataset_dir", default=COMBINED_INPUT_DIR, type=str)
@click.option(
    "--preprocessing_script",
    default=LCN_PREPROCESSING_SCRIPT_PATH,
    type=str,
)
@click.option("--label_file", default=LABEL_METADATA_FILE, type=str)
@click.option("--max_seq_len", default=512, type=int)
@click.option("--longformer", is_flag=True, default=False)
@click.option("--output_dir", default=LARGE_DATASET_STORAGE_PATH, type=str)
def prepare_ad_vs_hc_dataset(
    combined_dataset_dir: str = COMBINED_INPUT_DIR,
    preprocessing_script: str = LCN_PREPROCESSING_SCRIPT_PATH,
    label_file: str = LABEL_METADATA_FILE,
    max_seq_len: int = 512,
    longformer: bool = False,
    replace_punctuation: bool = False,
    output_dir: str = LARGE_DATASET_STORAGE_PATH,
):
    """Prepare the prodromal AD vs HC dataset.

    :param combined_dataset_dir: Path to the combined transcript dataset directory.
    :type combined_dataset_dir: str
    :param preprocessing_script: Path to the preprocessing script.
    :type preprocessing_script: str
    :param label_file: Path to the label metadata file.
    :type label_file: str
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param longformer: Whether the input should be processed for a model that can handle longer
        input lengths (combine multiple files).
    :type longformer: bool
    :param replace_punctuation: Whether to (re)place punctuation with a punctuation restoration model.
    :type replace_punctuation: bool
    :param output_dir: Path to the output directory.
    :type output_dir: str
    """
    # Load the label file
    labels = pd.read_csv(label_file, index_col=0)

    for config in FPACK_CONFIGS:
        # Load the text files
        combined_dataset = load_dataset(
            preprocessing_script,
            config,
            data_dir=combined_dataset_dir,
            split="train",
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            max_seq_len=max_seq_len,
            longformer=longformer,
            punctuation=replace_punctuation,
            cache_dir=LARGE_DATASET_STORAGE_PATH,
        )

        # Flatten the dataset
        combined_dataset = combined_dataset.map(flatten_batch, batched=True)
        # Merge all the info from the label file into the dataset, map the info based on the subject ID
        combined_dataset = combined_dataset.map(
            lambda example: {
                **example,
                **labels[labels.index == example["subject_id"]].to_dict(orient="records")[0],
            }
        )

        # Filter by inclusion/exclusion criteria
        # Healthy subjects ("ad_label" == 0) are required to have a CDR of 0 and a MMSE of 24 or above
        # (Prodromal) ADs ("ad_label" == 1) are required to have a CDR between 0.5 and 1 and a MMSE of 20 or above
        combined_dataset = combined_dataset.filter(
            lambda example: (example["ad_label"] == 0 and example["cdr"] == 0 and example["mmse"] >= 24) or
                            (example["ad_label"] == 1 and 0.5 <= example["cdr"] <= 1 and example["mmse"] >= 20)
        )

        # Filter by subjects that are either AD or HC
        combined_dataset = combined_dataset.filter(
            lambda example: example["ad_label"] in [0, 1]
        )
        # Describe the dataset
        describe_dataset_labels(combined_dataset, "ad_label")

        # Save the dataset
        combined_dataset.save_to_disk(f"{output_dir}/ad_hc_cls_{config}")


@click.command()
@click.option("--combined_dataset_dir", default=COMBINED_INPUT_DIR, type=str)
@click.option(
    "--preprocessing_script",
    default=LCN_PREPROCESSING_SCRIPT_PATH,
    type=str,
)
@click.option("--label_file", default=LABEL_METADATA_FILE, type=str)
@click.option("--max_seq_len", default=512, type=int)
@click.option("--replace_punctuation", default=False, type=bool)
@click.option("--longformer", is_flag=True, default=False)
@click.option("--output_dir", default=LARGE_DATASET_STORAGE_PATH, type=str)
def prepare_amyloid_pos_vs_neg_dataset(
    combined_dataset_dir: str = COMBINED_INPUT_DIR,
    preprocessing_script: str = LCN_PREPROCESSING_SCRIPT_PATH,
    label_file: str = LABEL_METADATA_FILE,
    max_seq_len: int = 512,
    replace_punctuation: bool = False,
    longformer: bool = False,
    output_dir: str = LARGE_DATASET_STORAGE_PATH,
):
    """Prepare the amyloid positive vs negative dataset.

    :param combined_dataset_dir: Path to the combined transcript dataset directory.
    :type combined_dataset_dir: str
    :param preprocessing_script: Path to the preprocessing script.
    :type preprocessing_script: str
    :param label_file: Path to the label metadata file.
    :type label_file: str
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param replace_punctuation: Whether to (re)place punctuation with a punctuation restoration model.
    :type replace_punctuation: bool
    :param longformer: Whether the input should be processed for a model that can handle longer
        input lengths (combine multiple files).
    :type longformer: bool
    :param output_dir: Path to the output directory.
    :type output_dir: str
    """
    # Load the label file
    labels = pd.read_csv(label_file, index_col=0)

    for config in FPACK_CONFIGS:
        # Load the text files
        combined_dataset = load_dataset(
            preprocessing_script,
            config,
            data_dir=combined_dataset_dir,
            split="train",
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            max_seq_len=max_seq_len,
            punctuation=replace_punctuation,
            longformer=longformer,
            cache_dir=LARGE_DATASET_STORAGE_PATH,
        )

        # Flatten the dataset
        combined_dataset = combined_dataset.map(flatten_batch, batched=True)
        # Merge all the info from the label file into the dataset, map the info based on the subject ID
        combined_dataset = combined_dataset.map(
            lambda example: {
                **example,
                **labels[labels.index == example["subject_id"]].to_dict(orient="records")[0],
            }
        )

        # Filter by inclusion/exclusion criteria
        # Healthy subjects ("ad_label" == 0) are required to have a CDR of 0 and a MMSE of 24 or above
        # (Prodromal) ADs ("ad_label" == 1) are required to have a CDR between 0.5 and 1 and a MMSE of 20 or above
        combined_dataset = combined_dataset.filter(
            lambda example: (example["ad_label"] == 0 and example["cdr"] == 0 and example["mmse"] >= 24) or
                            (example["ad_label"] == 1 and 0.5 <= example["cdr"] <= 1 and example["mmse"] >= 20)
        )

        # Filter by healthy subjects that are either amyloid positive or negative
        combined_dataset = combined_dataset.filter(
            lambda example: example["ad_label"] == 0 and example["amyloid_label"] in [0, 1]
        )
        # Describe the dataset
        describe_dataset_labels(combined_dataset, "amyloid_label")

        # Save the dataset
        combined_dataset.save_to_disk(f"{output_dir}/amyloid_pos_neg_cls_{config}")


@click.command()
@click.option("--combined_dataset_dir", default=COMBINED_INPUT_DIR, type=str)
@click.option(
    "--preprocessing_script",
    default=LCN_PREPROCESSING_SCRIPT_PATH,
    type=str,
)
@click.option("--label_file", default=LABEL_METADATA_FILE, type=str)
@click.option("--max_seq_len", default=512, type=int)
@click.option("--longformer", is_flag=True, default=False)
@click.option("--replace_punctuation", default=False, type=bool)
@click.option("--iterations", default=5, type=int)
@click.option("--output_dir", default=LARGE_DATASET_STORAGE_PATH, type=str)
def prepare_ad_hc_amyloid_pos_neg_datasets(
    combined_dataset_dir: str = COMBINED_INPUT_DIR,
    preprocessing_script: str = LCN_PREPROCESSING_SCRIPT_PATH,
    label_file: str = LABEL_METADATA_FILE,
    max_seq_len: int = 512,
    longformer: bool = False,
    replace_punctuation: bool = False,
    iterations: int = 5,
    output_dir: str = LARGE_DATASET_STORAGE_PATH,
):
    """Prepare the both the prodromal AD vs HC and the amyloid positive versus negative dataset.

    This set splits the amyloid negative healthy group into two groups that are matched by age and sex. The first group
    is used for AD vs HC classification and the second group is used for subsequent amyloid positive vs negative
    classification.

    :param combined_dataset_dir: Path to the combined transcript dataset directory.
    :type combined_dataset_dir: str
    :param preprocessing_script: Path to the preprocessing script.
    :type preprocessing_script: str
    :param label_file: Path to the label metadata file.
    :type label_file: str
    :param replace_punctuation: Whether to (re)place punctuation with a punctuation restoration model.
    :type replace_punctuation: bool
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param longformer: Whether the input should be processed for a model that can handle longer 
        input lengths (combine multiple files).
    :type longformer: bool
    :param iterations: Number of iterations for creating multiple splits of the two amyloid negative groups.
    :type iterations: int
    :param output_dir: Path to the output directory.
    :type output_dir: str
    """
    # Load the label file
    labels = pd.read_csv(label_file, index_col=0)

    for config in FPACK_CONFIGS:
        # Load the text files
        combined_dataset = load_dataset(
            preprocessing_script,
            config,
            data_dir=combined_dataset_dir,
            split="train",
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            max_seq_len=max_seq_len,
            longformer=longformer,
            punctuation=replace_punctuation,
            cache_dir=LARGE_DATASET_STORAGE_PATH,
        )

        # Flatten the dataset
        combined_dataset = combined_dataset.map(flatten_batch, batched=True)
        # Filter out the subjects that are not in the label file
        combined_dataset = combined_dataset.filter(
            lambda example: example["subject_id"] in labels.index
        )
        # Merge all the info from the label file into the dataset, map the info based on the subject ID
        combined_dataset = combined_dataset.map(
            lambda example: {
                **example,
                **labels[labels.index == example["subject_id"]].to_dict(orient="records")[0],
            }
        )

        # Filter by inclusion/exclusion criteria
        # Healthy subjects ("ad_label" == 0) are required to have a CDR of 0 and a MMSE of 24 or above
        # (Prodromal) ADs ("ad_label" == 1) are required to have a CDR between 0.5 and 1 and a MMSE of 20 or above
        # And ADs are required to have an amyloid label of 1
        combined_dataset = combined_dataset.filter(
            lambda example: (example["ad_label"] == 0 and example["cdr"] == 0 and example["mmse"] >= 24) or
                            (example["ad_label"] == 1 and 0.5 <= example["cdr"] <= 1 and example["mmse"] >= 20 \
                             and example["amyloid_label"] == 1)
        )
        # Log the number of subjects
        logger.info(f"Number of subjects with labels: {len(set(combined_dataset['subject_id']))}")

        # Split the different groups
        # 1. ADs
        ad_dataset = combined_dataset.filter(lambda example: example["ad_label"] == 1)
        # 2. Amyloid positive healthy subjects
        amyloid_pos_dataset = combined_dataset.filter(
            lambda example: example["ad_label"] == 0 and example["amyloid_label"] == 1
        )
        # 3. Amyloid negative healthy subjects
        amyloid_neg_dataset = combined_dataset.filter(
            lambda example: example["ad_label"] == 0 and example["amyloid_label"] == 0
        )
        amyloid_neg_subject_ids = sorted(list(set(amyloid_neg_dataset["subject_id"])))
        labels_subset = labels[labels.index.isin(amyloid_neg_subject_ids)]
 
        overlap_between_iterations = []
        for i in range(iterations):
            # Split this last third group in two halves and check that are matched by age, sex and education
            # Use multiple splits to mitigate biases
            logger.info(f"Iteration {i} of creating multiple amyloid negative splits for {config}")           
            
            matched_by_variables = False
            j = 0
            while not matched_by_variables:
                # Use sklearn's train_test_split to split the subject IDs in two halves
                group1, group2 = train_test_split(
                    labels_subset,
                    test_size=0.5,
                    random_state=i + j * 100,  # Add j to the random state to get different splits for multiple tries
                )
                amyloid_neg_first_half_subject_ids = list(group1.index)
                amyloid_neg_second_half_subject_ids = list(group2.index)

                # 3.1. First half
                amyloid_neg_first_half_dataset = amyloid_neg_dataset.filter(
                    lambda example: example["subject_id"] in amyloid_neg_first_half_subject_ids
                )
                # 3.2. Second half
                amyloid_neg_second_half_dataset = amyloid_neg_dataset.filter(
                    lambda example: example["subject_id"] in amyloid_neg_second_half_subject_ids
                )
                
                # Check if the age, sex and education are not statistically different between the two halves with a t-test
                labels_subset_first_half = labels_subset[labels_subset.index.isin(amyloid_neg_first_half_subject_ids)]
                labels_subset_second_half = labels_subset[labels_subset.index.isin(amyloid_neg_second_half_subject_ids)]
                p_value_age = ttest_ind(
                    labels_subset_first_half["age"],
                    labels_subset_second_half["age"],
                    equal_var=False,
                )[1]
                p_value_sex = ttest_ind(
                    labels_subset_first_half["gender"],
                    labels_subset_second_half["gender"],
                    equal_var=False,
                )[1]
                p_value_edu = ttest_ind(
                    labels_subset_first_half["education"],
                    labels_subset_second_half["education"],
                    equal_var=False,
                )[1]
                matched_by_variables = p_value_sex >= 0.05 and p_value_age >= 0.05 and p_value_edu >= 0.05
                logger.info(
                    f"Are the age, sex and education matched between the two halves? "
                    f"p value sex: {p_value_sex}, p value age: {p_value_age}, p value education: {p_value_edu} "
                    f"{matched_by_variables}"
                )
                j += 1

            overlap_between_iterations.append([amyloid_neg_first_half_subject_ids])
            if len(overlap_between_iterations) > 1:
                # Log the percentage of overlap between the current split and each of the previous splits
                for j, previous_split in enumerate(overlap_between_iterations[:-1]):
                    overlap = len(set(amyloid_neg_first_half_subject_ids) & set(previous_split[0])) / len(set(amyloid_neg_first_half_subject_ids))
                    logger.info(f"Overlap with previous split {j}: {overlap}")
                logger.info("\n")

            # Stitch 1 + 3.1 together for AD vs HC classification
            ad_hc_dataset = concatenate_datasets([ad_dataset, amyloid_neg_first_half_dataset])
            # Shuffle the dataset with a fixed seed
            ad_hc_dataset = ad_hc_dataset.shuffle(seed=42)
            # Describe the dataset
            logger.info(f"{config} AD vs HC dataset {i}")
            describe_dataset_labels(ad_hc_dataset, "ad_label")

            # Stitch 2 + 3.2 together for amyloid positive vs negative classification
            amyloid_pos_neg_dataset = concatenate_datasets([amyloid_pos_dataset, amyloid_neg_second_half_dataset])
            # Shuffle the dataset with a fixed seed
            amyloid_pos_neg_dataset = amyloid_pos_neg_dataset.shuffle(seed=42)
            # Describe the dataset
            logger.info(f"{config} Amyloid positive vs negative dataset {i}")
            describe_dataset_labels(amyloid_pos_neg_dataset, "amyloid_label")
            
            # Stitch 3.1 and 3.2 together for an amyloid negative classification control task
            # But first add new labels that indicate the group
            amyloid_neg_first_half_dataset = amyloid_neg_first_half_dataset.map(
                lambda example: {**example, "group_label": 0}
            )
            amyloid_neg_second_half_dataset = amyloid_neg_second_half_dataset.map(
                lambda example: {**example, "group_label": 1}
            )
            amyloid_neg_baseline_dataset = concatenate_datasets([amyloid_neg_first_half_dataset, amyloid_neg_second_half_dataset])
            # Shuffle the dataset with a fixed seed
            amyloid_neg_baseline_dataset = amyloid_neg_baseline_dataset.shuffle(seed=42)
            # Describe the dataset
            logger.info(f"{config} Amyloid negative baseline dataset {i}")
            describe_dataset_labels(amyloid_neg_baseline_dataset, "group_label")

            # Save the datasets
            ad_hc_dataset.save_to_disk(f"{output_dir}/ad_half_neg_hc_{config}_{i}")
            amyloid_pos_neg_dataset.save_to_disk(f"{output_dir}/amyloid_pos_half_neg_cls_{config}_{i}")
            amyloid_neg_baseline_dataset.save_to_disk(f"{output_dir}/amyloid_neg_baseline_{config}_{i}")


@click.group()
def cli() -> None:
    """Prepare datasets with text and diagnostic labels for classification."""


if __name__ == "__main__":
    cli.add_command(prepare_ad_vs_hc_dataset)
    cli.add_command(prepare_amyloid_pos_vs_neg_dataset)
    cli.add_command(prepare_ad_hc_amyloid_pos_neg_datasets)
    cli()
