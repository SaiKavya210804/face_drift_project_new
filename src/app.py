# --------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# --------------------------------------------------
# streamlit → builds interactive web UI
# numpy → numerical computations
# pandas → experiment logging and tables
# PIL → image loading
# matplotlib → plotting drift monitoring graphs

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------
# embedding_extractor → loads trained CNN and extracts embeddings
# baseline_manager → manages baseline embeddings and threshold
# drift_detector → computes distance and drift decision
# image_utils → preprocessing and degradation simulation
# evaluation → confusion matrix and accuracy metrics

from embedding_extractor import load_embedding_model
from baseline_manager import BaselineManager
from drift_detector import compute_distance, detect_drift, normalize_embedding
from embedding_extractor import load_embedding_model, extract_embedding
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
# LOAD CNN EMBEDDING MODEL
# --------------------------------------------------
# @cache_resource prevents reloading model on every UI refresh

@st.cache_resource
def load_model():
    return load_embedding_model("models/face_cnn.h5")

embedding_model = load_model()


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
# Streamlit reruns script after each user interaction.
# Session state keeps important data persistent.

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

if "current_degradation" not in st.session_state:
    st.session_state.current_degradation = "Clean"

if "result_logged" not in st.session_state:
    st.session_state.result_logged = False

baseline_manager = st.session_state.baseline_manager


# --------------------------------------------------
# EMBEDDING NORMALIZATION
# --------------------------------------------------
# Normalizing embeddings ensures fair distance comparison.

# def normalize_embedding(embedding):

#     embedding = embedding.astype("float32")

#     norm = np.linalg.norm(embedding)

#     if norm != 0:
#         embedding = embedding / norm

#     return embedding


# --------------------------------------------------
# BASELINE COLLECTION
# --------------------------------------------------
# Collect clean face images to form baseline distribution.

st.subheader("🎥 Capture Clean Face for Baseline")

st.info("Minimum 5 images required. 8–10 recommended for stability.")

camera_image = st.camera_input("Capture clean face")

if camera_image:

    image = Image.open(camera_image)
    st.image(image)

    # preprocess image before feeding into CNN
    processed = preprocess_image(image)

    # extract embedding vector from CNN
    # embedding = embedding_model.predict(processed)[0]

    # embedding = normalize_embedding(embedding)

    embedding = extract_embedding(embedding_model, processed)

    # store embedding in baseline pool
    if st.button("Add to Baseline"):

        baseline_manager.add_embedding(embedding)

        st.success(
            f"Added to baseline pool ({len(baseline_manager.embeddings)} samples)"
        )

    # compute statistical baseline
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
# DRIFT TESTING SECTION
# --------------------------------------------------

st.subheader("🖼 Test Image for Drift")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:

    # detect new image upload
    if st.session_state.last_uploaded_file != uploaded.name:

        st.session_state.current_test_image = Image.open(uploaded)
        st.session_state.last_uploaded_file = uploaded.name
        st.session_state.current_degradation = "Clean"
        st.session_state.result_logged = False

    image = st.session_state.current_test_image

    st.image(image)


    # --------------------------------------------------
    # IMAGE DEGRADATION SIMULATION
    # --------------------------------------------------
    # These functions simulate real-world image quality issues.

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


    # use modified image for testing
    image = st.session_state.current_test_image

    st.image(image)

    st.write("Degradation Type:", st.session_state.current_degradation)


    # --------------------------------------------------
    # EMBEDDING EXTRACTION
    # --------------------------------------------------

    processed = preprocess_image(image)

    embedding = embedding_model.predict(processed)[0]

    embedding = normalize_embedding(embedding)

    # st.write("Embedding sample:", embedding[:10])
    st.write("Non-zero embedding dimensions:", np.count_nonzero(embedding))


# --------------------------------------------------
# DRIFT DETECTION LOGIC
# --------------------------------------------------

if uploaded and baseline_manager.mean_vector is not None:

    threshold = baseline_manager.threshold

    # compute embedding distance from baseline mean
    distance = compute_distance(
        embedding,
        baseline_manager.mean_vector
    )

    # determine drift condition
    drift = detect_drift(distance, threshold)


    # --------------------------------------------------
    # DRIFT SEVERITY CLASSIFICATION
    # --------------------------------------------------
    # ratio helps classify drift severity levels

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
    # used for plotting drift monitoring graph

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
    # DISPLAY FINAL DRIFT RESULT
    # --------------------------------------------------

    st.write("Embedding Distance:", round(distance, 6))
    st.write("Drift Threshold:", round(threshold, 6))

    drift_detected = drift

    if drift_detected:

        st.error(
            "⚠ Drift Detected: Deviation beyond learned baseline tolerance"
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
    # allows rebuilding baseline if drift becomes permanent

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
# MODEL EVALUATION
# --------------------------------------------------

st.subheader("📊 Model Evaluation")

if st.button("Generate Confusion Matrix"):

    if len(st.session_state.true_labels) > 0:

        cm, acc = evaluate_results(
            st.session_state.true_labels,
            st.session_state.predicted_labels
        )

        st.write("Model Accuracy:", round(acc, 4))

        # plot_confusion_matrix(cm)
        fig = plot_confusion_matrix(cm)
        st.pyplot(fig)

    else:
        st.warning("Run experiments first to generate evaluation metrics")


# --------------------------------------------------
# CLEAR EXPERIMENT LOG
# --------------------------------------------------

if st.button("Clear Experiment Log"):

    st.session_state.experiment_log = []