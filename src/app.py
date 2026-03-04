import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from embedding_extractor import load_embedding_model
from baseline_manager import BaselineManager
from drift_detector import compute_distance, detect_drift
from image_utils import preprocess_image, apply_blur, apply_low_light

st.title("📉 Model Degradation & Drift Detection")

# --------------------------
# LOAD MODEL
# --------------------------

@st.cache_resource
def load_model():
    return load_embedding_model("models/face_cnn.h5")

embedding_model = load_model()

# --------------------------
# SESSION STATE INIT
# --------------------------

if "baseline_manager" not in st.session_state:
    st.session_state.baseline_manager = BaselineManager()

if "distance_history" not in st.session_state:
    st.session_state.distance_history = []

baseline_manager = st.session_state.baseline_manager


# --------------------------
# UTILITY: NORMALIZE VECTOR
# --------------------------

def normalize_embedding(embedding):
    embedding = embedding.astype("float32")
    norm = np.linalg.norm(embedding)
    if norm != 0:
        embedding = embedding / norm
    return embedding


# --------------------------
# BASELINE COLLECTION
# --------------------------

st.subheader("🎥 Capture Clean Face for Baseline")

camera_image = st.camera_input("Capture clean face")

if camera_image:
    image = Image.open(camera_image)
    st.image(image)

    processed = preprocess_image(image)
    embedding = embedding_model.predict(processed)[0]
    embedding = normalize_embedding(embedding)

    if st.button("Add to Baseline"):
        baseline_manager.add_embedding(embedding)
        st.success(f"Added to baseline pool ({len(baseline_manager.embeddings)} samples)")

    if st.button("Compute Baseline"):
        if len(baseline_manager.embeddings) < 5:
            st.warning("At least 5 baseline images required")
        else:
            mean_vector, threshold = baseline_manager.compute_baseline()
            st.success("Baseline Computed Successfully")
            st.write("Drift Threshold:", round(threshold, 6))


# --------------------------
# DRIFT TESTING
# --------------------------

st.subheader("🖼 Test Image for Drift")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:

    # Store original upload once
    if "current_test_image" not in st.session_state:
        st.session_state.current_test_image = Image.open(uploaded)

    # Show current working image
    st.image(st.session_state.current_test_image)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Apply Blur"):
            st.session_state.current_test_image = apply_blur(
                st.session_state.current_test_image
            )

    with col2:
        if st.button("Apply Low Light"):
            st.session_state.current_test_image = apply_low_light(
                st.session_state.current_test_image
            )

    # Always use updated image
    image = st.session_state.current_test_image
    st.image(image)

    processed = preprocess_image(image)
    embedding = embedding_model.predict(processed)[0]
    embedding = normalize_embedding(embedding)

    st.write("Embedding sample:", embedding[:10])
    
    if baseline_manager.mean_vector is not None:

        distance = compute_distance(
            embedding,
            baseline_manager.mean_vector
        )

        threshold = baseline_manager.threshold

        # --------------------------
        # MONITORING LOGIC
        # --------------------------

        st.session_state.distance_history.append(distance)

        # Keep last 20 values only
        if len(st.session_state.distance_history) > 20:
            st.session_state.distance_history.pop(0)

        st.write("Embedding Distance:", round(distance, 6))
        st.write("Drift Threshold:", round(threshold, 6))

        if detect_drift(distance, threshold):
            st.error("⚠ Drift Detected: Deviation beyond statistical tolerance (μ + 2σ)")
        else:
            st.success("✅ No Drift: Within statistical tolerance")

        # --------------------------
        # TREND VISUALIZATION
        # --------------------------

        st.subheader("📈 Drift Monitoring Trend")

        df = pd.DataFrame({
            "Distance": st.session_state.distance_history
        })

        fig, ax = plt.subplots()
        ax.plot(df["Distance"])
        ax.axhline(y=threshold, linestyle="--")
        ax.set_xlabel("Observation")
        ax.set_ylabel("Embedding Distance")

        st.pyplot(fig)

        # --------------------------
        # RECALIBRATION
        # --------------------------

        if st.button("Recalibrate System"):
            baseline_manager.reset()
            st.session_state.distance_history = []
            st.success("System Recalibrated. Baseline cleared.")