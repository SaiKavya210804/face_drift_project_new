import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np

def load_embedding_model(model_path):
    model = tf.keras.models.load_model(model_path)

    # 🔹 Force build model by calling it once
    dummy_input = np.zeros((1, 100, 100, 1))
    model(dummy_input)

    # 🔹 Extract 128-dim embedding layer (layer before dropout & softmax)
    embedding_model = Model(
        # inputs=model.input,
        inputs=model.layers[0].input,
        outputs=model.layers[-3].output
    )

    return embedding_model