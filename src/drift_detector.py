import numpy as np

def normalize_embedding(embedding):
    # Normalize embedding to unit length
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def compute_distance(current_embedding, baseline_mean):
    # Euclidean distance between current and baseline mean
    current_embedding = normalize_embedding(current_embedding)
    return np.linalg.norm(current_embedding - baseline_mean)

def detect_drift(distance, baseline_mean_dist, baseline_std, factor=2):
    # Drift if distance > mean_dist + factor*std
    threshold = baseline_mean_dist + factor * baseline_std
    return distance > threshold, threshold