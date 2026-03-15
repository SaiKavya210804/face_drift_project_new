# --------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# --------------------------------------------------
# seaborn → confusion matrix heatmap visualization
# matplotlib → plotting graphs
# sklearn → ML evaluation metrics

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

# --------------------------------------------------
# COMPUTE EVALUATION METRICS
# --------------------------------------------------
# true_labels → ground truth labels
# predicted_labels → model predictions

def evaluate_results(true_labels, predicted_labels):
    # Compute confusion matrix and accuracy
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    acc = accuracy_score(true_labels, predicted_labels)
    return cm, acc

def plot_confusion_matrix(cm):
    # Plot confusion matrix as heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    # Axis labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Drift", "Drift"])
    ax.set_yticklabels(["No Drift", "Drift"])

    # Show values inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im)
    return fig

#for binary classification, compute precision, recall, F1-score, and plot ROC curve
def compute_classification_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    return precision, recall, f1

def plot_roc_curve(true_labels, predicted_labels):
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return fig