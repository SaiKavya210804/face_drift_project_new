import numpy as np

class BaselineManager:
    def __init__(self):
        self.embeddings = []
        self.mean_vector = None
        self.threshold = None

    def add_embedding(self, embedding):
        # self.embeddings.append(embedding)
        norm = np.linalg.norm(embedding)

        if norm == 0:
            return  # avoid division by zero

        embedding = embedding / norm
        self.embeddings.append(embedding)

    # def compute_baseline(self):
    #     self.mean_vector = np.mean(self.embeddings, axis=0)

    #     distances = [
    #         np.linalg.norm(e - self.mean_vector)
    #         for e in self.embeddings
    #     ]

    #     mean_dist = np.mean(distances)
    #     std_dist = np.std(distances)

    #     # Statistical threshold
    #     self.threshold = mean_dist + 2 * std_dist

    #     return self.mean_vector, self.threshold
    def compute_baseline(self):
        if len(self.embeddings) < 5:
            raise ValueError("At least 5 baseline images required")
        self.mean_vector = np.mean(self.embeddings, axis=0)

        distances = [
            np.linalg.norm(e - self.mean_vector)
            for e in self.embeddings
     ]

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        self.threshold = mean_dist + 2 * std_dist

        return self.mean_vector, self.threshold

    def reset(self):
        self.embeddings = []
        self.mean_vector = None
        self.threshold = None