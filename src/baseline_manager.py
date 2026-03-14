import numpy as np

class BaselineManager:
    def __init__(self):
        # Initialize baseline storage
        self.embeddings = []
        self.mean_vector = None
        self.threshold = None
        self.mean_dist = None
        self.std_dist = None

    def add_embedding(self, embedding):
        # Normalize embedding before storing
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm
        self.embeddings.append(embedding)

    def compute_baseline(self):
        # Require minimum 5 samples
        if len(self.embeddings) < 5:
            raise ValueError("At least 5 baseline images required")

        # Compute normalized mean vector
        mean_vec = np.mean(self.embeddings, axis=0)
        self.mean_vector = mean_vec / np.linalg.norm(mean_vec)

        # Compute distances from mean
        distances = [np.linalg.norm(e - self.mean_vector) for e in self.embeddings]
        self.mean_dist = np.mean(distances)
        self.std_dist = np.std(distances)

        # Statistical threshold = mean + 2*std
        self.threshold = self.mean_dist + 2 * self.std_dist

        return self.mean_vector, self.threshold

    def reset(self):
        # Clear baseline data
        self.embeddings = []
        self.mean_vector = None
        self.threshold = None
        self.mean_dist = None
        self.std_dist = None