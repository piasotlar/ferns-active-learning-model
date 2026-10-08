from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np

#plots for probability distribution, uncertainty distribution, heatmaps, prediction histograms, feature importance 

def probability_distribution(probs, name, iteration, output_dir):
    plt.figure()
    plt.hist(probs[:, 1], bins=20, range=(0, 1), edgecolor="black")
    plt.xlabel("Probability of Yes")
    plt.ylabel("Number of viruses/contigs")
    plt.title(f"{name} distribution - iteration {iteration}")
    plt.xlim(0, 1)
    file_path = os.path.join(output_dir, f"{name}_prediction_distribution_iter_{iteration}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

def uncertainty_distribution(uncertainty, name, iteration, output_dir):
    plt.figure()
    plt.hist(uncertainty, bins=20, range=(0, 0.5), edgecolor="black")
    plt.xlabel("Uncertainty")
    plt.ylabel("Number of viruses/contigs")
    plt.title(f"{name} uncertainty distribution - iteration {iteration}")
    plt.xlim(0, 0.5)
    file_path = os.path.join(output_dir, f"{name}_uncertainty_distribution_iter_{iteration}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

def confusion_matrix_heatmap(y_test, y_pred, iteration, heatmap_dir = "heatmaps"):
    
    plt.figure(figsize=(5,4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix {iteration}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    file_path = os.path.join(heatmap_dir, f"confusion_matrix_iter_{iteration}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

def prediction_histogram(y_test, y_pred, iteration, pred_hist_dir = "prediction_histograms"):

    # število primerov
    true_no_pred_no = ((y_test == 0) & (y_pred == 0)).sum()
    true_no_pred_yes = ((y_test == 0) & (y_pred == 1)).sum()
    true_yes_pred_no = ((y_test == 1) & (y_pred == 0)).sum()
    true_yes_pred_yes = ((y_test == 1) & (y_pred == 1)).sum()

    x = np.arange(2)
    width = 0.35
    plt.figure(figsize=(6,5))
    plt.bar(x - width/2,[true_no_pred_no, true_yes_pred_no],width,label="Predicted No")
    plt.bar(x + width/2,[true_no_pred_yes, true_yes_pred_yes],width,label="Predicted Yes")
    plt.xticks(x, ["True No", "True Yes"])
    plt.ylabel("Number of contigs")
    plt.xlabel("True label")
    plt.title(f"Test set predictions - iteration {iteration}")
    plt.legend()

    plt.savefig(os.path.join(pred_hist_dir,f"test_prediction_histogram_iter_{iteration}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def calibration_plot(y_true, probs, name, iteration, output_dir="calibration_curves"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    CalibrationDisplay.from_predictions(
        y_true,
        probs[:, 1],      # verjetnost za razred Yes
        n_bins=10,
        strategy="quantile"
    )
    plt.title(f"{name} calibration curve - iteration {iteration}")
    plt.savefig(
        os.path.join(output_dir, f"{name}_calibration_iter_{iteration}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def feature_importance(model, X_test, y_test, iteration, name, output_dir="feature_importance"):
    os.makedirs(output_dir, exist_ok=True)

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1")

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(os.path.join(output_dir, f"{name}_feature_importance_iter_{iteration}.csv"), index=False)

    top = importance_df.head(20)

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["importance_mean"][::-1])
    plt.xlabel("Permutation importance based on F1")
    plt.title(f"Top feature importance - iteration {iteration}")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{name}_feature_importance_iter_{iteration}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return importance_df