import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

def load_embedding_model(model_path):
    # Load trained CNN model (Functional with explicit embedding layer)
    model = tf.keras.models.load_model(model_path)

    # Run dummy input to build graph
    dummy_input = np.zeros((1, 100, 100, 1))
    model(dummy_input)

    # Use penultimate dense layer as embedding
    embedding_layer = model.get_layer("embedding")   # not "dense_1"
    embedding_model = Model(inputs=model.input, outputs=embedding_layer.output)
    return embedding_model

def extract_embedding(embedding_model, processed_image):
    # Predict embedding from the penultimate dense layer
    embedding = embedding_model.predict(processed_image)[0]

    # Normalize embedding to unit length
    norm = np.linalg.norm(embedding)
    if norm != 0:
        embedding = embedding / norm
    # Debug check
    print("Embedding shape:", embedding.shape)
    print("First 5 values:", embedding[:5])

    return embedding