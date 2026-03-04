from tensorflow.keras.models import load_model

model = load_model("models/face_cnn.h5")
model.summary()