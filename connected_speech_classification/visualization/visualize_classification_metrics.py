"""Visualize classification metrics based on results obtained from mlflow logging."""
# Imports
from glob import glob
from typing import List

from pypalettes import load_cmap
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@click.command()
@click.option("--results_dir", type=str)
@click.option(
    "--metrics",
    multiple=True,
    default=["accuracy", "precision", "recall", "f1", "specificity"],
)
def visualize_classification_metrics(
    results_dir: str,
    metrics: List[str] = ["accuracy", "precision", "recall", "f1", "specificity"],  # noqa
) -> None:
    # Get all possible csv files in the results directory
    csv_files = glob(f"{results_dir}/*.csv")
    # Read a dataframe per csv file, put them into a dictionary, the key is the last part of the csv file path
    # and the value is the dataframe
    dataframes = {csv_file.split("/")[-1].strip(".csv"): pd.read_csv(csv_file, index_col=0) for csv_file in csv_files}

    # Filter out rows that have "micro"/"macro" in the index
    dataframes = {key: value[~value.index.str.contains("micro|macro")] for key, value in dataframes.items()}
    # Get the rows that have any of the metrics in the index
    dataframes = {key: value[value.index.str.contains("|".join(metrics))] for key, value in dataframes.items()}

    # Remove "mean_" from the index
    dataframes = {key: value.rename(index=lambda x: x.replace("mean_", "")) for key, value in dataframes.items()}

    # For experiment 3: If there are metrics with "ad" in the index, remove them
    dataframes = {key: value[~value.index.str.contains("ad")] for key, value in dataframes.items()}
    # And replace "am_" with ""
    dataframes = {key: value.rename(index=lambda x: x.replace("am_", "")) for key, value in dataframes.items()}

    # Create a subplots object, number of subplots = number of metrics, all subplots should have the same y-axis
    fig, ax = plt.subplots(len(metrics), 1, figsize=(10, 10), sharey=True, tight_layout=True)

    # For each metric, plot all the dataframes: There is one bar for each column in the dataframes
    # Get the metric names from the index of the dataframes
    metric_names = list(dataframes.values())[0].index

    for i, metric in enumerate(metric_names):
        # Create a grouped bar plot in seaborn
        # 1. Concatenate the rows of the dataframes for the current metric into one dataframe
        metric_df = pd.concat([value.loc[metric] for value in dataframes.values()], axis=1)
        # 2. Rename the columns of the dataframe to the keys of the dataframes dictionary
        metric_df.columns = dataframes.keys()
        # 3. Add the index as a separate column with the name "part"
        metric_df = metric_df.reset_index().rename(columns={"index": "part"})
        # 4. Melt the dataframe so that the columns are "part" and "value"
        metric_df = metric_df.melt(id_vars="part", var_name="experiment", value_name="value")
        # 5. Create the grouped bar plot
        sns.barplot(data=metric_df, x="part", y="value", hue="experiment", ax=ax[i], palette="crest")

        ax[i].set_title(metric)
        ax[i].legend()

    # Save the figure
    plt.savefig(f"{results_dir}/classification_metrics.png", dpi=300)


def plot_confusion_matrix(
    matrix: np.array, 
    labels: List[str],  # noqa
    xlabel: str = "Predicted label",
    ylabel: str = "True label",
    color_palette: str = "Purp",
) -> plt.Figure:
    """Plot a confusion matrix.
    
    :param matrix: The confusion matrix.
    :type matrix: np.array
    :param labels: The labels of the confusion matrix.
    :type labels: List[str]
    :param xlabel: The label of the x-axis.
    :type xlabel: str
    :param ylabel: The label of the y-axis.
    :type ylabel: str
    :param color_palette: The color palette to use.
    :type color_palette: str
    :return: The figure object.
    :rtype: plt.Figure
    """
    # Close any existing figures
    plt.close("all")
    cmap = load_cmap(color_palette, keep_first_n=5)
    fig, ax = plt.subplots()
    # Cmax is the maximum value in the matrix * 0.7
    ax.matshow(matrix, cmap=cmap)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=45, fontsize=20 if "True" in labels or "Yes" in labels else 12)
    ax.set_yticklabels(labels, rotation=45, fontsize=20 if "True" in labels or "Yes" in labels else 12)
    # Check if the matrix is a 2x2 matrix
    if matrix.shape != (2, 2):
        raise ValueError("The matrix should be a 2x2 matrix.")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="black", fontsize=32)
    plt.tight_layout()
    # Return the figure
    return fig


@click.group()
def cli() -> None:
    """Visualize classification metrics based on results obtained from mlflow logging."""


if __name__ == "__main__":
    cli.add_command(visualize_classification_metrics)
    cli()
