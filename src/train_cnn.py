import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "data/lfw_funneled"
IMG_SIZE = 100

X = []
y = []
label_map = {}
label = 0

print("Loading images...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    label_map[label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)

    label += 1

X = np.array(X) / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

print("Total images:", len(X))
print("Total classes:", len(label_map))

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Functional API model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = Conv2D(32, (3,3), activation="relu")(inputs)
x = MaxPooling2D()(x)
x = Conv2D(64, (3,3), activation="relu")(x)
x = MaxPooling2D()(x)
x = Flatten()(x)

# Explicit embedding layer
embedding = Dense(128, activation="relu", name="embedding")(x)

# Classification output
outputs = Dense(y.shape[1], activation="softmax", name="classification")(embedding)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training CNN...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

# Save Functional model with embedding layer
model.save("models/face_cnn.h5")
print("Model saved.")