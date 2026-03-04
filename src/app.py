import streamlit as st
import numpy as np
from PIL import Image

from embedding_extractor import load_embedding_model
from baseline_manager import BaselineManager
from drift_detector import compute_distance, detect_drift
from image_utils import preprocess_image, apply_blur, apply_low_light

st.title("📉 Model Degradation & Drift Detection")

@st.cache_resource
def load_model():
    return load_embedding_model("models/face_cnn.h5")

embedding_model = load_model()

if "baseline_manager" not in st.session_state:
    st.session_state.baseline_manager = BaselineManager()

baseline_manager = st.session_state.baseline_manager

# --------------------------
# LIVE CAPTURE
# --------------------------

st.subheader("🎥 Capture Clean Face for Baseline")

camera_image = st.camera_input("Capture clean face")

if camera_image:
    image = Image.open(camera_image)
    st.image(image)

    processed = preprocess_image(image)
    embedding = embedding_model.predict(processed)[0]

    if st.button("Add to Baseline"):
        baseline_manager.add_embedding(embedding)
        st.success("Added to baseline pool")

    # if st.button("Compute Baseline"):
    #     mean_vector, threshold = baseline_manager.compute_baseline()
    #     st.success("Baseline Computed")
    #     st.write("Drift Threshold:", round(threshold, 4))
    
    if st.button("Compute Baseline"):
        try:
            mean_vector, threshold = baseline_manager.compute_baseline()
            st.success("Baseline Computed")
            st.write("Drift Threshold:", round(threshold, 4))
        except ValueError as e:
            st.error(str(e))

# --------------------------
# TEST IMAGE
# --------------------------

st.subheader("🖼 Test Image for Drift")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Apply Blur"):
            image = apply_blur(image)
            st.image(image)

    with col2:
        if st.button("Apply Low Light"):
            image = apply_low_light(image)
            st.image(image)

    processed = preprocess_image(image)
    embedding = embedding_model.predict(processed)[0]

    if baseline_manager.mean_vector is not None:

        distance = compute_distance(
            embedding,
            baseline_manager.mean_vector
        )

        st.write("Embedding Distance:", round(distance, 4))
        st.write("Drift Threshold:", round(baseline_manager.threshold, 4))

        if detect_drift(distance, baseline_manager.threshold):
            st.error("⚠ Drift Detected")
        else:
            st.success("✅ No Drift")

        if st.button("Handle Drift (Reset Baseline)"):
            baseline_manager.reset()
            st.success("Baseline Reset")