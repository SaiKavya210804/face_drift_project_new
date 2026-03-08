# --------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# --------------------------------------------------
# Streamlit → UI for interactive web app
# numpy / pandas → numerical operations and logging
# PIL → image loading and manipulation
# matplotlib → plotting drift monitoring graphs

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------
# embedding_extractor → loads trained CNN model
# baseline_manager → handles baseline embeddings
# drift_detector → computes embedding distance & drift decision
# image_utils → preprocessing and synthetic degradation functions
# evaluation → confusion matrix & accuracy metrics

from embedding_extractor import load_embedding_model
from baseline_manager import BaselineManager
from drift_detector import compute_distance, detect_drift
from image_utils import (
    preprocess_image,
    apply_blur,
    apply_low_light,
    apply_noise,
    apply_rotation,
)
from evaluation import evaluate_results, plot_confusion_matrix


# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.title("📉 Deep Learning Based Model Degradation & Drift Detection")


# --------------------------------------------------
# LOAD EMBEDDING MODEL (CACHED)
# --------------------------------------------------
# Loads trained CNN face recognition model only once

@st.cache_resource
def load_model():
    return load_embedding_model("models/face_cnn.h5")


embedding_model = load_model()


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
# Streamlit reruns script after every UI interaction
# Session state keeps values persistent

if "baseline_manager" not in st.session_state:
    st.session_state.baseline_manager = BaselineManager()

if "distance_history" not in st.session_state:
    st.session_state.distance_history = []

if "experiment_log" not in st.session_state:
    st.session_state.experiment_log = []

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "true_labels" not in st.session_state:
    st.session_state.true_labels = []

if "predicted_labels" not in st.session_state:
    st.session_state.predicted_labels = []

# Prevent duplicate experiment logging
if "result_logged" not in st.session_state:
    st.session_state.result_logged = False

baseline_manager = st.session_state.baseline_manager


# --------------------------------------------------
# NORMALIZE EMBEDDING VECTOR
# --------------------------------------------------
# Ensures embedding vectors have unit length
# Improves distance comparison stability

def normalize_embedding(embedding):

    embedding = embedding.astype("float32")

    norm = np.linalg.norm(embedding)

    if norm != 0:
        embedding = embedding / norm

    return embedding


# --------------------------------------------------
# BASELINE COLLECTION
# --------------------------------------------------
# Collect clean face images to build baseline embedding
# distribution for drift detection

st.subheader("🎥 Capture Clean Face for Baseline")
st.info(
    "Minimum 5 images required. 8–10 recommended for better stability."
)

camera_image = st.camera_input("Capture clean face")

if camera_image:

    image = Image.open(camera_image)

    st.image(image)

    processed = preprocess_image(image)

    embedding = embedding_model.predict(processed)[0]

    embedding = normalize_embedding(embedding)

    if st.button("Add to Baseline"):

        baseline_manager.add_embedding(embedding)

        st.success(
            f"Added to baseline pool ({len(baseline_manager.embeddings)} samples)"
        )

    if st.button("Compute Baseline"):

        if len(baseline_manager.embeddings) < 5:

            st.warning("At least 5 baseline images required")

        else:

            mean_vector, threshold = baseline_manager.compute_baseline()

            baseline_mean = baseline_manager.mean_vector
            baseline_std = np.std(baseline_manager.embeddings, axis=0)

            st.success("Baseline Computed Successfully")

            st.write("Drift Threshold:", round(threshold, 6))
            st.write("Baseline Samples:", len(baseline_manager.embeddings))
            st.write("Embedding Mean (first 5):", baseline_mean[:5])
            st.write("Embedding Std Dev (first 5):", baseline_std[:5])


# --------------------------------------------------
# DRIFT TESTING
# --------------------------------------------------

st.subheader("🖼 Test Image for Drift")

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:

    if st.session_state.last_uploaded_file != uploaded.name:

        st.session_state.current_test_image = Image.open(uploaded)
        st.session_state.last_uploaded_file = uploaded.name
        st.session_state.current_degradation = "Clean"

        # reset logging flag
        st.session_state.result_logged = False

    image = st.session_state.current_test_image

    st.image(image)


# --------------------------------------------------
# IMAGE DEGRADATION SIMULATION
# --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Apply Blur"):
            st.session_state.current_test_image = apply_blur(image)
            st.session_state.current_degradation = "Blur"

    with col2:
        if st.button("Apply Low Light"):
            st.session_state.current_test_image = apply_low_light(image)
            st.session_state.current_degradation = "Low Light"

    with col3:
        if st.button("Apply Noise"):
            st.session_state.current_test_image = apply_noise(image)
            st.session_state.current_degradation = "Noise"

    with col4:
        if st.button("Rotate 20°"):
            st.session_state.current_test_image = apply_rotation(image)
            st.session_state.current_degradation = "Rotation"


# --------------------------------------------------
# USE MODIFIED IMAGE
# --------------------------------------------------

    image = st.session_state.current_test_image

    st.image(image)

    st.write("Degradation Type:", st.session_state.current_degradation)


# --------------------------------------------------
# EMBEDDING EXTRACTION (DEEP LEARNING)
# --------------------------------------------------

    processed = preprocess_image(image)

    embedding = embedding_model.predict(processed)[0]

    embedding = normalize_embedding(embedding)

    st.write("Embedding sample:", embedding[:10])


# --------------------------------------------------
# DRIFT DETECTION
# --------------------------------------------------

    if baseline_manager.mean_vector is not None:

        threshold = baseline_manager.threshold

        distance = compute_distance(
            embedding,
            baseline_manager.mean_vector
        )


# --------------------------------------------------
# DRIFT SEVERITY CLASSIFICATION
# --------------------------------------------------

        ratio = distance / threshold

        if ratio < 0.6:
            severity = "Stable"

        elif ratio < 0.8:
            severity = "Minor Deviation"

        elif ratio < 1.0:
            severity = "Warning: Approaching Drift"

        else:
            severity = "Drift Detected"

        st.write("Drift Status:", severity)


# --------------------------------------------------
# STORE DISTANCE HISTORY
# --------------------------------------------------

        st.session_state.distance_history.append(distance)

        if len(st.session_state.distance_history) > 20:
            st.session_state.distance_history.pop(0)


# --------------------------------------------------
# EARLY DRIFT TREND WARNING
# --------------------------------------------------

        if len(st.session_state.distance_history) >= 5:

            recent = st.session_state.distance_history[-5:]
            avg_recent = np.mean(recent)

            if avg_recent > threshold * 0.8 and avg_recent < threshold:

                st.warning(
                    "⚠ Drift trend increasing (approaching threshold)"
                )


# --------------------------------------------------
# DISPLAY DRIFT RESULT
# --------------------------------------------------

        st.write("Embedding Distance:", round(distance, 6))
        st.write("Drift Threshold:", round(threshold, 6))

        drift_detected = detect_drift(distance, threshold)

        if drift_detected:

            st.error(
                "⚠ Drift Detected: Deviation beyond statistical tolerance (μ + 2σ)"
            )

        else:

            st.success(
                "✅ No Drift: Within statistical tolerance"
            )


# --------------------------------------------------
# STORE LABELS FOR CONFUSION MATRIX
# --------------------------------------------------

        if not st.session_state.result_logged:

            true_label = 0 if st.session_state.current_degradation == "Clean" else 1

            predicted_label = 1 if drift_detected else 0

            st.session_state.true_labels.append(true_label)
            st.session_state.predicted_labels.append(predicted_label)

            st.session_state.result_logged = True


# --------------------------------------------------
# STORE EXPERIMENT RESULT
# --------------------------------------------------

            st.session_state.experiment_log.append({

                "Degradation": st.session_state.current_degradation,
                "Distance": round(float(distance), 6),
                "Threshold": round(float(threshold), 6),
                "Severity": severity

            })


# --------------------------------------------------
# DRIFT MONITORING GRAPH
# --------------------------------------------------

        st.subheader("📈 Drift Monitoring Trend")

        df = pd.DataFrame({
            "Distance": st.session_state.distance_history
        })

        fig, ax = plt.subplots()

        ax.plot(df["Distance"], marker="o", label="Embedding Distance")

        ax.axhline(
            y=threshold,
            linestyle="--",
            label="Drift Threshold"
        )

        ax.axhspan(
            threshold * 0.8,
            threshold,
            alpha=0.2,
            label="Warning Zone"
        )

        ax.axhspan(
            0,
            threshold * 0.8,
            alpha=0.1,
            label="Stable Zone"
        )

        ax.set_xlabel("Observation")
        ax.set_ylabel("Embedding Distance")

        ax.legend()

        st.pyplot(fig)


# --------------------------------------------------
# SYSTEM RECALIBRATION
# --------------------------------------------------

        if st.button("Recalibrate System"):

            baseline_manager.reset()

            st.session_state.distance_history = []
            st.session_state.true_labels = []
            st.session_state.predicted_labels = []
            st.session_state.experiment_log = []

            st.success("System Recalibrated. Baseline and evaluation data cleared.")


# --------------------------------------------------
# EXPERIMENT RESULTS TABLE
# --------------------------------------------------

st.subheader("📊 Experiment Log")

if len(st.session_state.experiment_log) > 0:

    df_log = pd.DataFrame(st.session_state.experiment_log)

    st.dataframe(df_log)

    csv = df_log.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Experiment Results",
        csv,
        "drift_experiment_results.csv",
        "text/csv"
    )


# --------------------------------------------------
# CONFUSION MATRIX EVALUATION
# --------------------------------------------------

st.subheader("📊 Model Evaluation")

if st.button("Generate Confusion Matrix"):

    if len(st.session_state.true_labels) > 0:

        cm, acc = evaluate_results(
            st.session_state.true_labels,
            st.session_state.predicted_labels
        )

        st.write("Model Accuracy:", round(acc, 4))

        plot_confusion_matrix(cm)

    else:
        st.warning("Run experiments first to generate evaluation metrics")


# --------------------------------------------------
# CLEAR LOG
# --------------------------------------------------

if st.button("Clear Experiment Log"):

    st.session_state.experiment_log = []