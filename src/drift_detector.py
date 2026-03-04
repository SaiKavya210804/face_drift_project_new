import numpy as np

def compute_distance(current_embedding, baseline_mean):
    # Cosine distance
    dot = np.dot(current_embedding, baseline_mean)
    norm1 = np.linalg.norm(current_embedding)
    norm2 = np.linalg.norm(baseline_mean)

    return 1 - (dot / (norm1 * norm2))

def detect_drift(distance, threshold):
    return distance > threshold