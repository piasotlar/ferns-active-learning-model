import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


# ============================================================
# CONFIGURATION
# ============================================================

STATS_FILE = "active_learning_stats.tsv"
PREDICTIONS_DIR = "predictions"

RESULTS_DIR = "results"

PERFORMANCE_DIR = os.path.join(RESULTS_DIR, "performance")
PROBABILITY_HEATMAP_DIR = os.path.join(RESULTS_DIR, "probability_heatmaps")
PROBABILITY_DIR = os.path.join(RESULTS_DIR, "probability_distributions")
CONFUSION_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
CALIBRATION_DIR = os.path.join(RESULTS_DIR, "calibration")
FEATURE_DIR = os.path.join(RESULTS_DIR, "feature_importance")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_dirs():

    directories = [
        RESULTS_DIR,
        PERFORMANCE_DIR,
        PROBABILITY_HEATMAP_DIR,
        PROBABILITY_DIR,
        CONFUSION_DIR,
        CALIBRATION_DIR,
        FEATURE_DIR,
        TABLES_DIR,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def get_available_iterations(dataset="test"):

    pattern = os.path.join(
        PREDICTIONS_DIR,
        f"{dataset}_predictions_iter_*.csv"
    )

    files = glob.glob(pattern)

    iterations = []

    for file in files:

        match = re.search(
            r"iter_(\d+)\.csv$",
            file
        )

        if match:
            iterations.append(
                int(match.group(1))
            )

    return sorted(iterations)


def read_predictions(dataset, iteration):

    path = os.path.join(
        PREDICTIONS_DIR,
        f"{dataset}_predictions_iter_{iteration}.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def read_test_predictions(iteration):

    df = read_predictions(
        "test",
        iteration
    )

    if "YES/NO" not in df.columns:
        raise ValueError(
            f"'YES/NO' column missing in test predictions "
            f"for iteration {iteration}"
        )

    df["y_true"] = df["YES/NO"].map({
        "No": 0,
        "Yes": 1
    })

    return df


# ============================================================
# 1. PERFORMANCE THROUGH ACTIVE LEARNING
# ============================================================

def plot_performance():

    iterations = get_available_iterations(
        "test"
    )

    if len(iterations) == 0:
        print("Ni datotek z napovedmi za testno množico.")
        return

    stats = pd.read_csv(
        STATS_FILE,
        sep="\t"
    )

    stats = stats.sort_values(
        "iteration"
    )

    rows = []

    for iteration in iterations:

        df = read_test_predictions(
            iteration
        )

        df = df.dropna(
            subset=[
                "y_true",
                "prediction",
                "prob_Yes"
            ]
        )

        y_true = df["y_true"].astype(int)
        y_pred = df["prediction"].astype(int)
        y_prob = df["prob_Yes"]

        rows.append({
            "iteration": iteration,

            "accuracy": accuracy_score(
                y_true,
                y_pred
            ),

            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "f1": f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "roc_auc": roc_auc_score(
                y_true,
                y_prob
            )
        })

    performance = pd.DataFrame(
        rows
    )

    performance = performance.merge(
        stats[
            [
                "iteration",
                "n_labeled"
            ]
        ],
        on="iteration",
        how="left"
    )

    performance = performance.sort_values(
        "iteration"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.plot(
        performance["iteration"],
        performance["accuracy"],
        marker="o",
        label="Accuracy"
    )

    ax.plot(
        performance["iteration"],
        performance["precision"],
        marker="o",
        label="Precision"
    )

    ax.plot(
        performance["iteration"],
        performance["recall"],
        marker="o",
        label="Recall"
    )

    ax.plot(
        performance["iteration"],
        performance["f1"],
        marker="o",
        label="F1"
    )

    ax.plot(
        performance["iteration"],
        performance["roc_auc"],
        marker="o",
        label="ROC-AUC"
    )

    ax.set_xlabel(
        "Število označenih primerov"
    )

    ax.set_ylabel(
        "Vrednost metrike"
    )

    ax.set_ylim(
        0.25,
        1.05
    )

    labels = [
        f"{int(row['iteration'])}\n(n={int(row['n_labeled'])})"
        for _, row in performance.iterrows()
    ]

    ax.set_xticks(
        performance["iteration"]
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_xlabel(
        "Iteracija aktivnega učenja (število označenih primerov)"
    )

    ax.set_title(
        "Uspešnost modela med aktivnim učenjem"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PERFORMANCE_DIR,
            "performance_vs_labeled.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            PERFORMANCE_DIR,
            "performance_vs_labeled.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # Save values used for the figure
    performance.to_csv(
        os.path.join(
            PERFORMANCE_DIR,
            "performance_metrics.csv"
        ),
        index=False
    )


# ============================================================
# 2. PROBABILITY HEATMAP DATA
# ============================================================

def get_probability_heatmap_data(
    dataset,
    bin_width=0.05
):

    iterations = get_available_iterations(
        dataset
    )

    if len(iterations) == 0:
        return None, None, None

    bins = np.arange(
        0,
        1 + bin_width,
        bin_width
    )

    heatmap_data = []
    used_iterations = []

    for iteration in iterations:

        df = read_predictions(
            dataset,
            iteration
        )

        if "prob_Yes" not in df.columns:
            print(
                f"Preskakujem {dataset}, iteracija {iteration}: "
                "'prob_Yes' manjka."
            )
            continue

        probs = (
            df["prob_Yes"]
            .dropna()
            .to_numpy()
        )

        if len(probs) == 0:
            continue

        counts, _ = np.histogram(
            probs,
            bins=bins
        )

        proportions = (
            counts / counts.sum()
        )

        heatmap_data.append(
            proportions
        )

        used_iterations.append(
            iteration
        )

    if len(heatmap_data) == 0:
        return None, None, None

    return (
        np.array(heatmap_data),
        used_iterations,
        bins
    )


# ============================================================
# 3. INDIVIDUAL PROBABILITY HEATMAPS
# ============================================================

def plot_probability_heatmap(
    dataset,
    bin_width=0.05
):

    data, iterations, bins = (
        get_probability_heatmap_data(
            dataset,
            bin_width
        )
    )

    if data is None:
        print(
            f"Ni podatkov o verjetnostih za {dataset}."
        )
        return

    dataset_names = {
        "train": "učna množica",
        "validation": "validacijska množica",
        "test": "testna množica",
        "pool": "neoznačena množica"
    }

    dataset_name = dataset_names.get(
        dataset,
        dataset
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            max(
                5,
                len(iterations) * 0.30
            )
        )
    )

    positive_values = data[
        data > 0
    ]

    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(
            vmin=positive_values.min(),
            vmax=positive_values.max()
        )
    )

    bin_centers = (
        bins[:-1] +
        np.diff(bins) / 2
    )

    tick_probabilities = np.arange(
        0,
        1.01,
        0.1
    )

    tick_positions = []

    for p in tick_probabilities:

        idx = np.argmin(
            np.abs(
                bin_centers - p
            )
        )

        tick_positions.append(
            idx
        )

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        [
            f"{p:.1f}"
            for p in tick_probabilities
        ]
    )

    ax.set_yticks(
        np.arange(
            len(iterations)
        )
    )

    ax.set_yticklabels(
        iterations
    )

    ax.set_xlabel(
        "Napovedana verjetnost pozitivnega razreda"
    )

    ax.set_ylabel(
        "Iteracija aktivnega učenja"
    )

    ax.set_title(
        f"Porazdelitev napovedanih verjetnosti – {dataset_name}"
    )

    midpoint_position = np.argmin(
        np.abs(
            bin_centers - 0.5
        )
    )

    ax.axvline(
        midpoint_position,
        linestyle="--",
        linewidth=1.5
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Delež primerov"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PROBABILITY_HEATMAP_DIR,
            f"{dataset}_probability_heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            PROBABILITY_HEATMAP_DIR,
            f"{dataset}_probability_heatmap.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()

    labels = [
        f"{bins[i]:.2f}-{bins[i + 1]:.2f}"
        for i in range(
            len(bins) - 1
        )
    ]

    output_df = pd.DataFrame(
        data,
        index=iterations,
        columns=labels
    )

    output_df.index.name = "iteration"

    output_df.to_csv(
        os.path.join(
            PROBABILITY_HEATMAP_DIR,
            f"{dataset}_probability_heatmap.csv"
        )
    )


# ============================================================
# 4. COMBINED TRAIN / POOL / TEST HEATMAP
# ============================================================

def plot_combined_probability_heatmaps(
    bin_width=0.05
):

    datasets = [
        "train",
        "pool",
        "test"
    ]

    dataset_names = {
        "train": "Učna množica",
        "pool": "Neoznačena množica",
        "test": "Testna množica"
    }

    available = {}

    for dataset in datasets:

        data, iterations, bins = (
            get_probability_heatmap_data(
                dataset,
                bin_width
            )
        )

        if data is not None:

            available[dataset] = {
                "data": data,
                "iterations": iterations,
                "bins": bins
            }

    if len(available) == 0:
        print(
            "Ni podatkov o napovedanih verjetnostih."
        )
        return

    all_positive_values = np.concatenate([
        item["data"][item["data"] > 0]
        for item in available.values()
    ])

    vmax = all_positive_values.max()

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(
            6 * len(available),
            7
        )
    )

    if len(available) == 1:
        axes = [axes]

    last_im = None

    for ax, (dataset, item) in zip(
        axes,
        available.items()
    ):

        data = item["data"]
        iterations = item["iterations"]
        bins = item["bins"]

        last_im = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=vmax
        )

        bin_centers = (
            bins[:-1] +
            np.diff(bins) / 2
        )

        tick_probabilities = np.arange(
            0,
            1.01,
            0.2
        )

        tick_positions = []

        for p in tick_probabilities:

            idx = np.argmin(
                np.abs(
                    bin_centers - p
                )
            )

            tick_positions.append(
                idx
            )

        ax.set_xticks(
            tick_positions
        )

        ax.set_xticklabels(
            [
                f"{p:.1f}"
                for p in tick_probabilities
            ]
        )

        ax.set_yticks(
            np.arange(
                len(iterations)
            )
        )

        ax.set_yticklabels(
            iterations
        )

        ax.set_xlabel(
            "Napovedana verjetnost pozitivnega razreda"
        )

        ax.set_ylabel(
            "Iteracija aktivnega učenja"
        )

        ax.set_title(
            dataset_names.get(
                dataset,
                dataset
            )
        )

        midpoint_position = np.argmin(
            np.abs(
                bin_centers - 0.5
            )
        )

        ax.axvline(
            midpoint_position,
            linestyle="--",
            linewidth=1.3
        )

    fig.suptitle(
        "Porazdelitev napovedanih verjetnosti med aktivnim učenjem",
        fontsize=14
    )

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        fraction=0.025,
        pad=0.04
    )

    cbar.set_label(
        "Delež primerov"
    )

    plt.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.10,
        top=0.88,
        wspace=0.30
    )

    plt.savefig(
        os.path.join(
            PROBABILITY_HEATMAP_DIR,
            "train_pool_test_probability_heatmaps.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            PROBABILITY_HEATMAP_DIR,
            "train_pool_test_probability_heatmaps.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 5. PROBABILITY DISTRIBUTION BY TRUE CLASS
# ============================================================

def plot_probability_distribution_by_class(
    bin_width=0.05
):

    iterations = get_available_iterations(
        "test"
    )

    if len(iterations) == 0:
        print(
            "Ni datotek z napovedmi za testno množico."
        )
        return

    bins = np.arange(
        0,
        1 + bin_width,
        bin_width
    )

    classes = {
        0: "negativni",
        1: "pozitivni"
    }

    for class_value, class_name in classes.items():

        heatmap_data = []
        used_iterations = []

        for iteration in iterations:

            df = read_test_predictions(
                iteration
            )

            probs = (
                df.loc[
                    df["y_true"] == class_value,
                    "prob_Yes"
                ]
                .dropna()
                .to_numpy()
            )

            if len(probs) == 0:
                continue

            counts, _ = np.histogram(
                probs,
                bins=bins
            )

            proportions = (
                counts / counts.sum()
            )

            heatmap_data.append(
                proportions
            )

            used_iterations.append(
                iteration
            )

        if len(heatmap_data) == 0:
            print(
                f"Ni podatkov za {class_name} razred."
            )
            continue

        heatmap_data = np.array(
            heatmap_data
        )

        positive_values = (
            heatmap_data[
                heatmap_data > 0
            ]
        )

        norm = LogNorm(
            vmin=positive_values.min(),
            vmax=positive_values.max()
        )

        fig, ax = plt.subplots(
            figsize=(
                11,
                max(
                    5,
                    len(used_iterations) * 0.30
                )
            )
        )

        im = ax.imshow(
            heatmap_data,
            aspect="auto",
            interpolation="nearest",
            norm=norm
        )

        bin_centers = (
            bins[:-1] +
            np.diff(bins) / 2
        )

        tick_probabilities = np.arange(
            0,
            1.01,
            0.1
        )

        tick_positions = []

        for probability in tick_probabilities:

            idx = np.argmin(
                np.abs(
                    bin_centers -
                    probability
                )
            )

            tick_positions.append(
                idx
            )

        ax.set_xticks(
            tick_positions
        )

        ax.set_xticklabels(
            [
                f"{p:.1f}"
                for p in tick_probabilities
            ]
        )

        ax.set_xlabel(
            "Napovedana verjetnost pozitivnega razreda"
        )

        ax.set_yticks(
            np.arange(
                len(used_iterations)
            )
        )

        ax.set_yticklabels(
            used_iterations
        )

        ax.set_ylabel(
            "Iteracija aktivnega učenja"
        )

        midpoint_position = np.argmin(
            np.abs(
                bin_centers - 0.5
            )
        )

        ax.axvline(
            midpoint_position,
            linestyle="--",
            linewidth=1.5
        )

        ax.set_title(
            f"Porazdelitev napovedanih verjetnosti – "
            f"dejanski {class_name} razred"
        )

        cbar = fig.colorbar(
            im,
            ax=ax
        )

        cbar.set_label(
            "Delež primerov znotraj dejanskega razreda"
        )

        plt.tight_layout()

        class_file_name = (
            "no"
            if class_value == 0
            else "yes"
        )

        plt.savefig(
            os.path.join(
                PROBABILITY_DIR,
                f"test_probability_distribution_true_"
                f"{class_file_name}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            os.path.join(
                PROBABILITY_DIR,
                f"test_probability_distribution_true_"
                f"{class_file_name}.pdf"
            ),
            bbox_inches="tight"
        )

        plt.close()

        bin_labels = [
            f"{bins[i]:.2f}-{bins[i + 1]:.2f}"
            for i in range(
                len(bins) - 1
            )
        ]

        output_df = pd.DataFrame(
            heatmap_data,
            index=used_iterations,
            columns=bin_labels
        )

        output_df.index.name = (
            "iteration"
        )

        output_df.to_csv(
            os.path.join(
                PROBABILITY_DIR,
                f"test_probability_distribution_true_"
                f"{class_file_name}.csv"
            )
        )


# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(
    iterations
):

    fig, axes = plt.subplots(
        1,
        len(iterations),
        figsize=(
            5 * len(iterations),
            4
        )
    )

    if len(iterations) == 1:
        axes = [axes]

    for ax, iteration in zip(
        axes,
        iterations
    ):

        df = read_test_predictions(
            iteration
        )

        y_true = df["y_true"]
        y_pred = df["prediction"]

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Negativni",
                "Pozitivni"
            ]
        )

        disp.plot(
            ax=ax,
            colorbar=False,
            cmap="Blues"
        )

        ax.set_title(
            f"Iteracija {iteration}"
        )

        ax.set_xlabel(
            "Napovedani razred"
        )

        ax.set_ylabel(
            "Dejanski razred"
        )

    fig.suptitle(
        "Matriki zmede na testni množici"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CONFUSION_DIR,
            "confusion_matrix_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            CONFUSION_DIR,
            "confusion_matrix_comparison.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. CALIBRATION
# ============================================================

def read_labeled_predictions(
    dataset,
    iteration
):

    df = read_predictions(
        dataset,
        iteration
    )

    if "YES/NO" not in df.columns:
        raise ValueError(
            f"'YES/NO' column missing in "
            f"{dataset}_predictions_iter_{iteration}.csv"
        )

    df["y_true"] = df["YES/NO"].map({
        "No": 0,
        "Yes": 1
    })

    df = df.dropna(
        subset=[
            "y_true",
            "prob_Yes"
        ]
    )

    return df


def get_comparison_iterations(
    dataset
):

    iterations = get_available_iterations(
        dataset
    )

    if len(iterations) == 0:
        return []

    indices = np.linspace(
        0,
        len(iterations) - 1,
        5
    ).round().astype(int)

    return sorted(
        set(
            iterations[i]
            for i in indices
        )
    )


def plot_calibration_dataset(
    dataset
):

    iterations = get_comparison_iterations(
        dataset
    )

    if len(iterations) == 0:
        print(
            f"Ni datotek z napovedmi za {dataset}. "
            "Kalibracija bo preskočena."
        )
        return

    dataset_names = {
        "train": "učna množica",
        "validation": "validacijska množica",
        "test": "testna množica"
    }

    dataset_name = dataset_names.get(
        dataset,
        dataset
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 6)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Popolna kalibracija"
    )

    for iteration in iterations:

        try:

            df = read_labeled_predictions(
                dataset,
                iteration
            )

        except ValueError as e:

            print(e)
            continue

        y_true = (
            df["y_true"]
            .astype(int)
        )

        y_prob = df[
            "prob_Yes"
        ]

        prob_true, prob_pred = (
            calibration_curve(
                y_true,
                y_prob,
                n_bins=10,
                strategy="uniform"
            )
        )

        brier = brier_score_loss(
            y_true,
            y_prob
        )

        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=1.5,
            label=(
                f"Iteracija {iteration} "
                f"(Brier={brier:.3f})"
            )
        )

    ax.set_xlabel(
        "Povprečna napovedana verjetnost"
    )

    ax.set_ylabel(
        "Opazovani delež pozitivnih primerov"
    )

    ax.set_title(
        f"Kalibracija napovedanih verjetnosti – {dataset_name}"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CALIBRATION_DIR,
            f"{dataset}_calibration_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            CALIBRATION_DIR,
            f"{dataset}_calibration_comparison.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()


def plot_all_calibrations():

    for dataset in [
        "train",
        "validation",
        "test"
    ]:

        plot_calibration_dataset(
            dataset
        )


# ============================================================
# 8. DETAILED TEST METRICS
# ============================================================

def create_metrics_table():

    iterations = get_available_iterations(
        "test"
    )

    rows = []

    for iteration in iterations:

        df = read_test_predictions(
            iteration
        )

        df = df.dropna(
            subset=[
                "y_true",
                "prediction",
                "prob_Yes"
            ]
        )

        y_true = df["y_true"].astype(int)
        y_pred = df["prediction"].astype(int)
        y_prob = df["prob_Yes"]

        row = {

            "iteration":
                iteration,

            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred
                ),

            "precision":
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "roc_auc":
                roc_auc_score(
                    y_true,
                    y_prob
                ),

            "pr_auc":
                average_precision_score(
                    y_true,
                    y_prob
                ),

            "brier_score":
                brier_score_loss(
                    y_true,
                    y_prob
                ),
        }

        rows.append(
            row
        )

    metrics = pd.DataFrame(
        rows
    )

    if os.path.exists(
        STATS_FILE
    ):

        stats = pd.read_csv(
            STATS_FILE,
            sep="\t"
        )

        metrics = metrics.merge(
            stats[
                [
                    "iteration",
                    "n_labeled",
                    "n_unlabeled"
                ]
            ],
            on="iteration",
            how="left"
        )

    metrics.to_csv(
        os.path.join(
            TABLES_DIR,
            "test_metrics_all_iterations.csv"
        ),
        index=False
    )

    return metrics


# ============================================================
# 9. SUMMARY TABLE
# ============================================================

def create_summary_table(
    metrics
):

    metrics = metrics.sort_values(
        "iteration"
    )

    n = len(metrics)

    selected_indices = sorted(
        set([
            0,
            n // 2,
            n - 1
        ])
    )

    summary = metrics.iloc[
        selected_indices
    ].copy()

    summary.to_csv(
        os.path.join(
            TABLES_DIR,
            "thesis_summary.csv"
        ),
        index=False
    )


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

def get_feature_importance_files(
    dataset
):

    pattern = os.path.join(
        "feature_importance",
        f"{dataset}_feature_importance_iter_*.csv"
    )

    files = glob.glob(
        pattern
    )

    file_iteration_pairs = []

    for file in files:

        match = re.search(
            r"iter_(\d+)\.csv$",
            file
        )

        if match:

            iteration = int(
                match.group(1)
            )

            file_iteration_pairs.append(
                (
                    iteration,
                    file
                )
            )

    file_iteration_pairs.sort(
        key=lambda x: x[0]
    )

    return file_iteration_pairs


def plot_feature_importance_heatmap(
    dataset,
    top_n=15
):

    files = get_feature_importance_files(
        dataset
    )

    if len(files) == 0:

        print(
            f"Ni datotek s pomembnostjo značilk za {dataset}."
        )

        return

    dataset_names = {
        "train": "učna množica",
        "validation": "validacijska množica",
        "test": "testna množica"
    }

    dataset_name = dataset_names.get(
        dataset,
        dataset
    )

    all_data = []

    for iteration, file in files:

        df = pd.read_csv(
            file
        )

        required_columns = [
            "feature",
            "importance_mean"
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            print(
                f"Preskakujem {file}. "
                f"Manjkajoči stolpci: {missing}"
            )

            continue

        tmp = df[
            [
                "feature",
                "importance_mean"
            ]
        ].copy()

        tmp["iteration"] = (
            iteration
        )

        all_data.append(
            tmp
        )

    if len(all_data) == 0:

        print(
            f"Ni veljavnih podatkov o pomembnosti "
            f"značilk za {dataset}."
        )

        return

    importance = pd.concat(
        all_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Select top features
    # --------------------------------------------------------

    top_features = (
        importance
        .groupby("feature")[
            "importance_mean"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(top_n)
        .index
    )

    importance = importance[
        importance["feature"].isin(
            top_features
        )
    ]

    # --------------------------------------------------------
    # Feature x iteration matrix
    # --------------------------------------------------------

    matrix = importance.pivot_table(
        index="feature",
        columns="iteration",
        values="importance_mean",
        aggfunc="mean"
    )

    matrix = matrix[
        sorted(
            matrix.columns
        )
    ]

    feature_order = (
        matrix
        .mean(axis=1)
        .sort_values(
            ascending=False
        )
        .index
    )

    matrix = matrix.loc[
        feature_order
    ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            max(
                10,
                len(matrix.columns) * 0.5
            ),
            max(
                6,
                len(matrix.index) * 0.45
            )
        )
    )

    im = ax.imshow(
        matrix.values,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_xticks(
        np.arange(
            len(
                matrix.columns
            )
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        np.arange(
            len(
                matrix.index
            )
        )
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_xlabel(
        "Iteracija aktivnega učenja"
    )

    ax.set_ylabel(
        "Značilka"
    )

    ax.set_title(
        f"Pomembnost značilk med aktivnim učenjem – {dataset_name}"
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Povprečna permutacijska pomembnost"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FEATURE_DIR,
            f"{dataset}_feature_importance_heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            FEATURE_DIR,
            f"{dataset}_feature_importance_heatmap.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()

    matrix.to_csv(
        os.path.join(
            FEATURE_DIR,
            f"{dataset}_feature_importance_heatmap.csv"
        )
    )


def plot_all_feature_importance_heatmaps():

    for dataset in [
        "train",
        "validation",
        "test"
    ]:

        plot_feature_importance_heatmap(
            dataset,
            top_n=15
        )

def plot_uncertainty_bins_normalized():

    stats = pd.read_csv(
        STATS_FILE,
        sep="\t"
    ).sort_values("iteration")

    # --------------------------------------------------------
    # Stolpci z negotovostjo
    # --------------------------------------------------------

    uncertainty_columns = [
        col for col in stats.columns
        if col.startswith("uncertainty_over_")
    ]

    if len(uncertainty_columns) == 0:
        print("Ni stolpcev z negotovostjo.")
        return

    # --------------------------------------------------------
    # Normalizacija glede na število še neoznačenih primerov
    #
    # n_uncertain / n_unlabeled
    # --------------------------------------------------------

    data = stats[
        uncertainty_columns
    ].div(
        stats["n_unlabeled"],
        axis=0
    )

    # Imena pragov za x-os
    thresholds = [
        col.replace(
            "uncertainty_over_",
            ""
        ).replace("_", ".")
        for col in uncertainty_columns
    ]

    # --------------------------------------------------------
    # Normalizacija BARV znotraj vsakega stolpca posebej
    #
    # 0 = najmanjša vrednost tega stolpca
    # 1 = največja vrednost tega stolpca
    # --------------------------------------------------------

    color_data = np.zeros_like(
        data.values,
        dtype=float
    )

    for j in range(data.shape[1]):

        values = data.iloc[:, j].to_numpy()

        vmin = values.min()
        vmax = values.max()

        if vmax > vmin:

            color_data[:, j] = (
                (values - vmin) /
                (vmax - vmin)
            )

        else:

            color_data[:, j] = 0.5

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    im = ax.imshow(
        color_data,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=0,
        vmax=1
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(len(thresholds))
    )

    ax.set_xticklabels(
        thresholds,
        rotation=45,
        ha="right"
    )

    ax.set_xlabel(
        "Prag negotovosti"
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    ax.set_yticks(
        np.arange(len(stats))
    )

    ax.set_yticklabels(
        stats["iteration"].astype(int)
    )

    ax.set_ylabel(
        "Iteracija aktivnega učenja"
    )

    ax.set_title(
        "Spreminjanje deleža negotovih napovedi "
        "na neoznačeni množici"
    )

    # --------------------------------------------------------
    # Colorbar
    #
    # Ker je vsak stolpec normaliziran posebej,
    # barva predstavlja relativno vrednost znotraj stolpca.
    # --------------------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_ticks(
        [0, 0.5, 1]
    )

    cbar.set_ticklabels(
        [
            "Najmanjši",
            "Srednji",
            "Največji"
        ]
    )

    cbar.set_label(
        "Relativna vrednost znotraj posameznega praga"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        os.path.join(
            PROBABILITY_DIR,
            "uncertainty_bins_normalized.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            PROBABILITY_DIR,
            "uncertainty_bins_normalized.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():

    create_dirs()

    test_iterations = (
        get_available_iterations(
            "test"
        )
    )

    if len(test_iterations) == 0:

        raise RuntimeError(
            "V mapi predictions/ ni datotek z napovedmi "
            "za testno množico."
        )

    first = (
        test_iterations[0]
    )

    middle = (
        test_iterations[
            len(test_iterations) // 2
        ]
    )

    final = (
        test_iterations[-1]
    )

    comparison_iterations = sorted(
        set([
            first,
            middle,
            final
        ])
    )

    print(
        "Iteracije uporabljene za primerjavo:",
        comparison_iterations
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    plot_performance()

    # --------------------------------------------------------
    # Probability heatmaps
    # --------------------------------------------------------

    for dataset in [
        "train",
        "pool",
        "test"
    ]:

        plot_probability_heatmap(
            dataset
        )

    plot_combined_probability_heatmaps()

    # --------------------------------------------------------
    # Probability distribution by true class
    # --------------------------------------------------------

    plot_probability_distribution_by_class()

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    plot_confusion_matrices(
        [
            first,
            final
        ]
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    plot_all_calibrations()

    # --------------------------------------------------------
    # Metrics tables
    # --------------------------------------------------------

    metrics = (
        create_metrics_table()
    )

    create_summary_table(
        metrics
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    plot_all_feature_importance_heatmaps()

    print()
    print("Končano.")
    print(
        f"Rezultati so shranjeni v: {RESULTS_DIR}/"
    )


    plot_uncertainty_bins_normalized()

if __name__ == "__main__":
    main()