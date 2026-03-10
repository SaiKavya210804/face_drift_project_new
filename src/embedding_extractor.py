import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np


def load_embedding_model(model_path):

    model = tf.keras.models.load_model(model_path)

    # Force model to build
    dummy_input = np.zeros((1, 100, 100, 1))
    model(dummy_input)

    # Print architecture (for debugging)
    print(model.summary())

    # Correct embedding layer (Dense 128)
    # embedding_layer = model.get_layer("dense")
    embedding_layer = model.layers[-3]

    embedding_model = Model(
        inputs=model.layers[0].input,
        outputs=embedding_layer.output
    )

    print("Embedding dimension:", embedding_model.output_shape)

    return embedding_model


def extract_embedding(embedding_model, processed_image):

    embedding = embedding_model.predict(processed_image)[0]
    
    # Normalize embedding
    norm = np.linalg.norm(embedding)
    # print("Embedding vector:", embedding[:20])
    print("Embedding norm:", np.linalg.norm(embedding))
    print("Non-zero dimensions:", np.count_nonzero(embedding))
    if norm != 0:
        embedding = embedding / norm

    return embedding