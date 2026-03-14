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

# def plot_confusion_matrix(cm):

#     plt.figure(figsize=(5,4))

#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt='d',
#         cmap="Blues",
#         xticklabels=["No Drift","Drift"],
#         yticklabels=["No Drift","Drift"]
#     )

#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Drift Detection Confusion Matrix")

#     # return plt
#     fig = plt.gcf()
#     return fig