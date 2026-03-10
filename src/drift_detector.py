import numpy as np

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def compute_distance(current_embedding, baseline_mean):

    # Normalize current embedding
    # current_embedding = normalize_embedding(current_embedding)
    current_embedding = current_embedding

    # Euclidean distance
    return np.linalg.norm(current_embedding - baseline_mean)


def detect_drift(distance, threshold):
    return distance > threshold

# def detect_drift(distance, threshold, baseline_std=None):

#     if distance > threshold:
#         return True

#     if baseline_std is not None and distance > baseline_std * 2:
#         return True

#     return False