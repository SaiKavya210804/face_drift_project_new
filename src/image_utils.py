import numpy as np
from PIL import Image
import cv2

IMG_SIZE = 100  # must match model training size

# def preprocess_image(image):
#     image = image.resize((IMG_SIZE, IMG_SIZE))
#     image = np.array(image)

#     # Convert RGB → Grayscale properly
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     image = image / 255.0

#     # Add channel dimension
#     image = np.expand_dims(image, axis=-1)

#     # Add batch dimension
#     image = np.expand_dims(image, axis=0)

#     return image

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    # Convert RGB → grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image / 255.0
    image = image.astype("float32")   # 🔴 important fix

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image

def apply_blur(image, ksize=9):
    img_np = np.array(image)
    blurred = cv2.GaussianBlur(img_np, (ksize, ksize), 0)
    return Image.fromarray(blurred)


def apply_low_light(image, factor=0.8):
    img_np = np.array(image)
    dark = (img_np * factor).astype(np.uint8)
    return Image.fromarray(dark)

def apply_noise(image, noise_level=25): #gaussian noise with std dev of 25
    img_np = np.array(image)
    noise = np.random.normal(0, noise_level, img_np.shape)
    noisy = img_np + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

# def apply_rotation(image, angle=20):
#     return image.rotate(angle)

def apply_rotation(image, angle=20):
    rotated = image.rotate(angle, expand=True)
    return rotated.resize((IMG_SIZE, IMG_SIZE))