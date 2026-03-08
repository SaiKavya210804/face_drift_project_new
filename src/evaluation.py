# --------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# --------------------------------------------------
# seaborn → confusion matrix heatmap visualization
# matplotlib → plotting graphs
# sklearn → ML evaluation metrics

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


# --------------------------------------------------
# COMPUTE EVALUATION METRICS
# --------------------------------------------------
# true_labels → ground truth labels
# predicted_labels → model predictions

def evaluate_results(true_labels, predicted_labels):

    # compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # compute accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    return cm, accuracy


# --------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

def plot_confusion_matrix(cm):

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=["No Drift","Drift"],
        yticklabels=["No Drift","Drift"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Drift Detection Confusion Matrix")

    plt.show()