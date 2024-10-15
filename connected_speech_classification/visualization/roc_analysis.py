"""Helper script to visualize ROC curve with variability across folds in a cross-validation experiment."""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc


def get_roc_auc_plot_cv(
    fprs_original: np.array,
    tprs_original: np.array,
    plot_all_folds: bool = True,
) -> Tuple[np.array, np.array, plt.Figure]:
    """Generate a ROC curve plot with variability across folds in a cross-validation experiment.
    
    :param fprs_original: False positive rates for each fold, shape (n_splits, n_samples)
    :type fprs_original: np.array
    :param tprs_original: True positive rates for each fold, shape (n_splits, n_samples)
    :type tprs_original: np.array
    :param plot_all_folds: Whether to plot all the folds or not, defaults to True
    :type plot_all_folds: bool
    :return: Mean FPR, mean TPR and Figure object with the ROC curve plot
    :rtype: (np.array, np.array, plt.Figure)
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (fpr, tpr) in enumerate(zip(fprs_original, tprs_original)):
        viz = RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=auc(fpr, tpr),
        )
        if plot_all_folds:
            viz.plot(
                name=f"ROC fold {fold}",
                alpha=0.6,
                lw=1,
                ax=ax,
            )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0 
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="r",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.1,
        label=r"95% CI",
    )

    # Plot a straight line from (0,0) to (1,1)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="k", label="Chance")

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve",
    )
    ax.tick_params(axis='both', which='major', labelsize=16)
    # Increase the font size of the x and y labels
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.title.set_fontsize(16)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.close(fig)
    
    return mean_fpr, mean_tpr, fig
