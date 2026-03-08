import numpy as np

def compute_distance(current_embedding, baseline_mean):

    # Euclidean distance between embeddings
    return np.linalg.norm(current_embedding - baseline_mean)

def detect_drift(distance, threshold):
    return distance > threshold