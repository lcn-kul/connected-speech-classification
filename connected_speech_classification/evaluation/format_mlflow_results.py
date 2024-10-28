"""Helper file to process the mlflow logs (generated .csv files) into nice LaTeX tables."""
import os
from typing import List

import click
import pandas as pd
from datetime import datetime

from connected_speech_classification.constants import (
    MEDICAL_METRICS,
    MLFLOW_RESULTS_DIR,
    ML_METRICS,
    Q_FILE_MAPPING_NAMES,
)


@click.command()
@click.option("--result_file", default=os.path.join(MLFLOW_RESULTS_DIR, "ad_hc_cls.csv"), type=str)
@click.option("--recent_results", default=True, type=bool)
@click.option("--remove_micro_macro_avg", default=True, type=bool)
@click.option("--data_split", default=-1, type=int)
@click.option("--output_dir", default=MLFLOW_RESULTS_DIR, type=str)
def convert_mlflow_tables(
    result_file: str = os.path.join(MLFLOW_RESULTS_DIR, "ad_hc_cls.csv"),
    recent_results: bool = True,
    remove_micro_macro_avg: bool = True,
    data_split: int = -1,
    output_dir: str = MLFLOW_RESULTS_DIR,
):
    """Convert the MLflow results to LaTeX tables.

    :param result_file: The path to the MLflow results file created using the mlflow csv command
    :type result_file: str, defaults to os.path.join(MLFLOW_RESULTS_DIR, "ad_hc_cls.csv")
    :param recent_results: Whether to filter for recent results not older than a month
    :type recent_results: bool, defaults to True
    :param remove_micro_macro_avg: Whether to remove the micro and macro average columns
    :type remove_micro_macro_avg: bool, defaults to True
    :param data_split: The data split to consider
    :type data_split: int, defaults to -1
    :param output_dir: The directory to save the LaTeX tables
    :type output_dir: str, defaults to MLFLOW_RESULTS_DIR
    """
    # Load the MLflow results
    mlflow_results = pd.read_csv(result_file)
    # Get the original file name without the extension
    file_name = os.path.basename(result_file).split(".")[0]

    # Perform lots of processing steps
    # 1. Filter for rows that have "FINISHED" in the "status" column
    mlflow_results = mlflow_results[mlflow_results["status"] == "FINISHED"]

    # 2. (Optional) Filter for rows that have been logged recently
    if recent_results:
        # 2.1 Get the current date
        current_date = datetime.now()
        # 2.2 Identify runs that are less than 1 month old
        mlflow_results["start_time"] = pd.to_datetime(
            mlflow_results["start_time"],
            format="ISO8601",
        )
        # Remove the tz info from the start_time column
        mlflow_results["start_time"] = mlflow_results["start_time"].dt.tz_localize(None)
        mlflow_results = mlflow_results[
            (current_date - mlflow_results["start_time"]).dt.days < 7
        ]

    # Prepare to filter for the model path in the amyloid case (sequential vs independent) later
    exp_setup = ["ind"]
    if "amyloid" in result_file and "joint" not in result_file:
        exp_setup.append("seq")

    # 3. Filter for rows that report average metrics across the folds
    # 3.1 Identify the columns to remove: they have "metrics" but not "mean" in them
    columns_to_remove = [col for col in mlflow_results.columns if "metrics" in col and "mean" not in col]
    # Also remove columns that have "metrics" but neither "item" nor "subject" in them
    columns_to_remove += [
        col for col in mlflow_results.columns if "metrics" in col and "item" not in col and "subject" not in col
    ]

    # Remove the any macro micro average columns if specified
    if remove_micro_macro_avg:
        columns_to_remove += [col for col in mlflow_results.columns if "micro" in col]
        columns_to_remove += [col for col in mlflow_results.columns if "macro" in col]
        # Remove the "recall_sensitivity" column from old experiments
        columns_to_remove += [col for col in mlflow_results.columns if "recall_sensitivity" in col]

    # 3.2 Drop the columns
    mlflow_results = mlflow_results.drop(columns=columns_to_remove)

    # Filter for the model path in the amyloid case (sequential vs independent)
    for exp_setting in exp_setup:
        mlflow_results_exp = mlflow_results.copy()
        if data_split != -1:
            # Fill the NaN values in the config column with the previous non-NaN value
            mlflow_results_exp["params.config"] = mlflow_results_exp["params.config"].fillna(method="bfill")
            # Also in the classification_model_base column
            mlflow_results_exp[
                "params.classification_model_base"
            ] = mlflow_results_exp["params.classification_model_base"].fillna(method="bfill")
            # Make sure that params.data_split is not nan
            mlflow_results_exp = mlflow_results_exp[~mlflow_results_exp["params.data_split"].isna()]
            mlflow_results_exp = mlflow_results_exp[mlflow_results_exp["params.data_split"] == data_split]
        if exp_setting == "ind":
            # In the independent setup for amyloid, use the results that have NOT been trained on the AD vs hc task
            mlflow_results_exp = mlflow_results_exp[
                ~mlflow_results_exp["params.classification_model_base"].str.contains("ad_", na=False)
            ]
        else:
            # In the sequential setup for amyloid, use the results that have been trained on the AD vs hc task
            mlflow_results_exp = mlflow_results_exp[
                mlflow_results_exp["params.classification_model_base"].str.contains("ad_", na=False)
            ]

        # Create two different tables for each experiment, and also for the independent/sequential setups in the amyloid case
        for item_subject in ["item", "subject"]:
            columns_to_keep = [col for col in mlflow_results_exp.columns if "params.config" in col]

            # Get the LaTeX table for the item or subject and ad or amyloid evaluation distinction in the joint case
            if "joint" in result_file:
                for task in ["ad", "amyloid"]:
                    columns_to_keep_task = columns_to_keep + [
                        col for col in mlflow_results_exp.columns if task in col and item_subject in col and "params" not in col
                        ]

                    latex_table = get_latex_table(
                        mlflow_results=mlflow_results_exp,
                        columns_to_keep=columns_to_keep_task,
                        item_subject=item_subject,
                    )

                    # Save the LaTeX table to a file
                    output_file_name = f"{file_name}_{item_subject}_{task}_{exp_setting}"
                    if data_split != -1:
                        output_file_name += f"_{data_split}"
                    output_file = os.path.join(output_dir, f"{output_file_name}.txt")
                    with open(output_file, "w") as f:
                        f.write(latex_table)
            # Get the LaTeX table for the item or subject
            else:
                columns_to_keep += [col for col in mlflow_results_exp.columns if item_subject in col and "params" not in col]

                latex_table = get_latex_table(
                    mlflow_results=mlflow_results_exp,
                    columns_to_keep=columns_to_keep,
                    item_subject=item_subject,
                )

                # Save the LaTeX table to a file
                output_file_name = f"{file_name}_{item_subject}_{exp_setting}"
                if data_split != -1:
                    output_file_name += f"_{data_split}"
                output_file = os.path.join(output_dir, f"{output_file_name}.txt")
                with open(output_file, "w") as f:
                    f.write(latex_table)


def get_latex_table(
    mlflow_results: pd.DataFrame,
    columns_to_keep: List[str],
    item_subject: str = "item",
) -> str:
    """Get the LaTeX table for the MLflow results.

    :return: The LaTeX table
    :rtype: str
    """
    mlflow_results_item_subject = mlflow_results.copy()
    # If the data_split is not -1, remove the column that has "sem" in it
    if len(mlflow_results["params.data_split"]) > 0 and mlflow_results["params.data_split"].iloc[0] > 0:
        mlflow_results_item_subject = mlflow_results_item_subject.drop(  # noqa
            columns=[col for col in mlflow_results_item_subject.columns if "sem" in col]  # noqa
        )
        columns_to_keep += ["params.config"]
    # Filter by columns to keep
    mlflow_results_item_subject = mlflow_results[columns_to_keep]
    # Remove columns that only have NaN values
    mlflow_results_item_subject = mlflow_results_item_subject.dropna(axis=1, how="all")

    # Remove the rows that have nan values for the mean columns
    mlflow_results_item_subject = mlflow_results_item_subject.dropna(
        subset=[
            col for col in mlflow_results_item_subject.columns if "mean" in col \
                and "lower" not in col and "upper" not in col
        ]
    )

    # Remove columns with NaN values
    mlflow_results_item_subject = mlflow_results_item_subject.dropna(axis=1, how="all")
    # Remove "metrics.mean_item" or "metrics.mean_subject" from the column names
    mlflow_results_item_subject.columns = [
        col.replace(f"metrics.mean_{item_subject}_", "") for col in mlflow_results_item_subject.columns
    ]

    # Replace "_" with " " in the column names
    mlflow_results_item_subject.columns = [
        col.replace("_", " ").replace(".", " ") for col in mlflow_results_item_subject.columns
    ]
    # If the column name has a white space, only use the second part
    mlflow_results_item_subject.columns = [col.split(" ")[-1] for col in mlflow_results_item_subject.columns]

    # Don't sort all columns alphabetically, but put them in groups,
    # first: config column
    # then medical metrics: "roc_auc", "auc_lower", "auc_upper", "sensitivity", "specificity", "ppv", "npv"
    # then ML metrics: "accuracy", "f1", "precision", "recall"
    # in case there are more columns, put them after that
    medical_metrics_columns = [
        col for col in mlflow_results_item_subject.columns if any(metric in col for metric in MEDICAL_METRICS)
    ]
    # Sort the medical metrics columns alphabetically
    medical_metrics_columns.sort()
    # Same for ML metrics
    ml_metrics_columns = [
        col for col in mlflow_results_item_subject.columns if any(metric in col for metric in ML_METRICS)
    ]
    ml_metrics_columns.sort()
    mlflow_results_item_subject = mlflow_results_item_subject[
        ["config"] + medical_metrics_columns + ml_metrics_columns
    ]

    # Any value > 1 should be 1.00 instead
    mlflow_results_item_subject = mlflow_results_item_subject.applymap(
        lambda x: 1 if type(x) == int and x > 1 else x  # noqa: E721
    )

    # Round all numerical values to three decimal places
    # Convert the numerical columns to string because of formatting issues
    mlflow_results_item_subject = mlflow_results_item_subject.round(3).astype(str)

    # For the auc, do the following: "roc_auc (auc_lower, auc_upper)"
    mlflow_results_item_subject["auc"] = mlflow_results_item_subject["auc"] + " (" \
        + mlflow_results_item_subject["lower"] + ", " \
        + mlflow_results_item_subject["upper"] + ")"
    # Drop the auc_lower and auc_upper columns
    mlflow_results_item_subject = mlflow_results_item_subject.drop(columns=["lower", "upper"])

    # In the params.config column, replace the config file name with the actual name
    mlflow_results_item_subject["config"] = mlflow_results_item_subject["config"].map(Q_FILE_MAPPING_NAMES.get)

    # Create the LaTeX table using the pandas to LaTeX method
    latex_table = mlflow_results_item_subject.to_latex(index=False, escape=True)

    return latex_table


@click.group()
def cli() -> None:
    """Convert the MLflow results into LaTeX tables."""


if __name__ == "__main__":
    cli.add_command(convert_mlflow_tables)
    cli()
