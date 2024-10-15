"""Dataset loading script for LCN data based on huggingface's dataset library."""

import os
import re
from dataclasses import dataclass
from typing import List

import datasets

from connected_speech_classification.constants import FPACK_INPUT_DIR, Q_FILE_MAPPINGS
from connected_speech_classification.utils import (
    generate_long_sentence_chunks_for_several_files,
    generate_sentence_chunks_per_file,
    sort_fpack_files,
)

# FPACK Description
_DESCRIPTION = """\
LCN data preprocessing script.
"""


@dataclass
class LCNBuilderConfig(datasets.BuilderConfig):
    """Custom BuilderConfig class to include more class attributes."""

    max_seq_len: int = 512
    punctuation: bool = False
    lowercase: bool = False
    multi_sentence: bool = True
    longformer: bool = False


class LCNDataset(datasets.GeneratorBasedBuilder):
    """LCN Data.

    The data_files argument won't work since the extraction is starting from a directory and not a single data file.
    When using this dataset script with load_dataset and a subsequent torch dataloader, it might be necessary to pass
    collate_fn=lambda x: x to the dataloader. Example use:

    dataset = load_dataset(LCN_PREPROCESSING_SCRIPT_PATH, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=lambda x: x)
    """

    VERSION = datasets.Version("1.2.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    BUILDER_CONFIG_CLASS = LCNBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        LCNBuilderConfig(
            name="q1_short_subject_wise",
            version=VERSION,
            description="This dataset building configuration groups subjects together, only the first "
                        "sub-question of the Q1 interview (the autobiographical interview)",
        ),
        LCNBuilderConfig(
            name="q1_subject_wise",
            version=VERSION,
            description="This dataset building configuration groups subjects together, only using Q1 "
                        "(the autobiographical interview)",
        ),
        LCNBuilderConfig(
            name="q2_subject_wise",
            version=VERSION,
            description="This dataset building configuration groups subjects together, only using Q2 "
                        "(description of a day)",
        ),
        LCNBuilderConfig(
            name="q3_subject_wise",
            version=VERSION,
            description="This dataset building configuration groups subjects together, only using Q3 "
                        "(news events)",
        ),
        LCNBuilderConfig(
            name="q4_subject_wise",
            version=VERSION,
            description="This dataset building configuration groups subjects together, only using Q4 "
                        "(word descriptions)",
        ),
        LCNBuilderConfig(
            name="q5_subject_wise",
            version=VERSION,
            description="This dataset building configuration groups subjects together, only using Q5 "
                        "(cookie theft picture)",
        ),
        LCNBuilderConfig(
            name="subject_wise",
            version=VERSION,
            description="This dataset building configuration groups subjects together using all interview questions",
        ),
        LCNBuilderConfig(
            name="question_wise",
            version=VERSION,
            description="This dataset building configuration groups questions together, using all subjects",
        ),
    ]

    DEFAULT_CONFIG_NAME = "q1_subject_wise"

    def _info(self):
        if "subject" in self.config.name:
            if "q" in self.config.name:
                features = datasets.Features(
                    {
                        "subject_id": datasets.Value("string"),
                        "sentences_"
                        + self.config.name[:2]: datasets.features.Sequence(
                            datasets.Value("string")
                        ),
                    }
                )
            else:
                features = datasets.Features(
                    {
                        "subject_id": datasets.Value("string"),
                        "sentences_all": datasets.features.Sequence(
                            datasets.Value("string")
                        ),
                    }
                )
        else:
            features = datasets.Features(
                {
                    "question_id": datasets.Value("string"),
                    "sentences": datasets.features.Sequence(datasets.Value("string")),
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same
        # structure with the url replaced with path to local files.
        # By default, the archives will be extracted and a path to a cached
        # folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)
        # data_dir = _URLS[self.config.name]
        # As of now, there is only a train split that contains the full dataset
        if self.config.data_dir is None:
            self.config.data_dir = FPACK_INPUT_DIR

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": self.config.data_dir,
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # Generate examples specific for a subject based on the first three FPACK questions
        # Use a simple counter for unique keys
        key = 0

        # Process the files subject-wise
        if "subject" in self.config.name:
            # Get all subjects
            all_files = os.listdir(filepath)
            all_subjects = set([
                re.match(r"(s\d+)|(ARCK_AD_P\d+)|(MC_\d+)|(A_s\d+)|(sub_A\d+)", file_name)[0] for file_name in all_files
            ])
            # Process each subject
            for subject in all_subjects:
                # Define the parts of the interview data that are going to be used depending on the config name
                if "q" in self.config.name:
                    # Get the file name indicator for the targeted question
                    interview_parts_regex = re.compile(
                        "^" + subject + "_" + ".*" + Q_FILE_MAPPINGS[self.config.name] + ".*.txt"
                    )
                    part = self.config.name[:2]
                else:
                    # Aggregate all interview parts
                    interview_parts_regex = re.compile(
                        "^" + subject + "_" + ".*(" + "|".join(list(Q_FILE_MAPPINGS.values())) + ").*.txt"
                    )
                    part = "all"
                # Create one example per subject ("sentences")
                sentences = {}
                # Get all the relevant files indicated by the previously defined file indicator
                relevant_files = sort_fpack_files(
                    [
                        os.path.join(filepath, f)
                        for f in os.listdir(filepath)
                        if interview_parts_regex.search(f)
                    ]
                )
                q_chunks = []

                if not self.config.longformer:
                    for file in relevant_files:
                        q_chunks += generate_sentence_chunks_per_file(
                            file,
                            self.config.max_seq_len,
                            replace_punctuation=self.config.punctuation,
                            lowercase=self.config.lowercase,
                            multi_sentence=self.config.multi_sentence,
                        )
                else:
                    # Load the input file as one long input (e.g., combine the subparts of Q1)
                    #  and replace punctuation and lowercase, if desired
                    q_chunks = generate_long_sentence_chunks_for_several_files(
                        relevant_files,
                        self.config.max_seq_len,
                        replace_punctuation=self.config.punctuation,
                        lowercase=self.config.lowercase,
                    )
                # Save the question-specific chunks
                sentences["sentences_" + part] = q_chunks

                key += 1

                sentences["subject_id"] = subject

                # Yields subject-specific examples as (key, example) tuples
                yield key, sentences

        # Process the files question-wise
        else:
            for question_id, interview_part in Q_FILE_MAPPINGS.items():
                # Create one example per question
                sentences = {}

                # Look for relevant files for a given interview_part in all subject directories
                all_files: List[str] = []  # noqa
                for dir_path, _, files in os.walk(filepath):
                    for x in files:
                        all_files.append(os.path.join(dir_path, x))

                interview_part_regex = re.compile(".*" + interview_part + ".*.txt")
                relevant_files = sort_fpack_files(
                    [
                        os.path.join(filepath, f)
                        for f in all_files
                        if interview_part_regex.search(f)
                    ]
                )
                # Aggregate subject-specific chunks into a combined list for a given question
                q_chunks = []
                # Process each relevant file for all subjects for a given specific question
                if not self.config.longformer:
                    for file in relevant_files:
                        subj_chunks = generate_sentence_chunks_per_file(
                            file,
                            self.config.max_seq_len,
                            replace_punctuation=self.config.punctuation,
                            lowercase=self.config.lowercase,
                        )
                        q_chunks += subj_chunks
                else:
                    # Load the input file as one long input (e.g., combine the subparts of Q1)
                    #  and replace punctuation and lowercase, if desired
                    q_chunks = generate_long_sentence_chunks_for_several_files(
                        relevant_files,
                        self.config.max_seq_len,
                        replace_punctuation=self.config.punctuation,
                        lowercase=self.config.lowercase,
                    )

                # Save the question-specific chunks
                sentences["sentences"] = q_chunks

                key += 1
                # Add question ID
                sentences["question_id"] = question_id.strip("_subject_wise")

                yield key, sentences
