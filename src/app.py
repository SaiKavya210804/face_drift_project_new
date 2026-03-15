# --------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# --------------------------------------------------
# streamlit → builds interactive web UI
# numpy → numerical computations
# pandas → experiment logging and tables
# PIL → image loading
# matplotlib → plotting drift monitoring graphs

import drift_detector
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
from evaluation import compute_classification_metrics, evaluate_results, plot_confusion_matrix, plot_roc_curve


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

if "current_test_image" not in st.session_state:
    st.session_state.current_test_image = None

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

# Shortcut reference to baseline manager
baseline_manager = st.session_state.baseline_manager

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
            # mean_vector, threshold = baseline_manager.compute_baseline()

            # baseline_mean = baseline_manager.mean_vector
            # baseline_std = np.std(baseline_manager.embeddings, axis=0)

            # st.success("Baseline Computed Successfully")

            # st.write("Drift Threshold:", round(threshold, 6))
            # st.write("Baseline Samples:", len(baseline_manager.embeddings))
            # st.write("Embedding Mean (first 5):", baseline_mean[:5])
            # st.write("Embedding Std Dev (first 5):", baseline_std[:5])
            mean_vector, threshold = baseline_manager.compute_baseline()

            st.success("Baseline Computed Successfully")
            st.write("Drift Threshold:", round(threshold, 6))
            st.write("Baseline Samples:", len(baseline_manager.embeddings))
            st.write("Embedding Mean (first 5):", mean_vector[:5])
            st.write("Baseline Distances → mean:", round(baseline_manager.mean_dist, 6),
                     "std:", round(baseline_manager.std_dist, 6))

# --------------------------------------------------
# DRIFT TESTING SECTION
# --------------------------------------------------
st.subheader("🖼 Test Image for Drift")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    # Detect new image upload
    if st.session_state.last_uploaded_file != uploaded.name:
        st.session_state.current_test_image = Image.open(uploaded)
        st.session_state.last_uploaded_file = uploaded.name
        st.session_state.current_degradation = "Clean"
        st.session_state.result_logged = False

    image = st.session_state.current_test_image
    st.image(image)

    # Preprocess and extract embedding
    processed = preprocess_image(image)
    embedding = extract_embedding(embedding_model, processed)
    
    st.write("Embedding shape:", embedding.shape)
    st.write("First 5 values:", embedding[:5])
    st.write("Non-zero embedding dimensions:", np.count_nonzero(embedding))

    # Compute distance and drift decision
    distance = compute_distance(embedding, baseline_manager.mean_vector)
    drift, threshold = detect_drift(distance,
                                    baseline_manager.mean_dist,
                                    baseline_manager.std_dist)

    # Assign labels (0 = No Drift, 1 = Drift)
    true_label = 0 if st.session_state.current_degradation == "Clean" else 1
    predicted_label = 1 if drift else 0

    # Show results
    st.write("Embedding Distance:", round(distance, 6))
    st.write("Threshold:", round(threshold, 6))
    st.write("Drift Status:", "Drift Detected" if drift else "No Drift")

    # Log experiment if not already logged
    if not st.session_state.result_logged:
        st.session_state.experiment_log.append(
            [uploaded.name, distance, threshold, true_label, predicted_label]
        )
        st.session_state.true_labels.append(true_label)
        st.session_state.predicted_labels.append(predicted_label)
        st.session_state.result_logged = True
    # --------------------------------------------------
# DRIFT DETECTION LOGIC
# --------------------------------------------------
    if baseline_manager.mean_vector is None:
        st.warning("⚠️ Please compute baseline before testing drift.")
    else:
        # Compute distance from baseline mean
        distance = compute_distance(embedding, baseline_manager.mean_vector)

        # Determine drift condition using mean + std
        drift, threshold = detect_drift(distance,
                                        baseline_manager.mean_dist,
                                        baseline_manager.std_dist)

        # Assign labels (0 = No Drift, 1 = Drift)
        true_label = 0 if st.session_state.current_degradation == "Clean" else 1
        predicted_label = 1 if drift else 0

        # Show results

        st.write("Embedding Distance:", round(distance, 6))
        st.write("Threshold:", round(threshold, 6))
        st.write("Drift Status:", "Drift Detected" if drift else "No Drift")

        # Log experiment if not already logged
        if not st.session_state.result_logged:
            st.session_state.experiment_log.append(
                [uploaded.name, distance, threshold, true_label, predicted_label]
            )
            st.session_state.true_labels.append(true_label)
            st.session_state.predicted_labels.append(predicted_label)
            st.session_state.result_logged = True

    # --------------------------------------------------
    # IMAGE DEGRADATION SIMULATION
    # --------------------------------------------------
    # These functions simulate real-world image quality issues.
    # - Added st.session_state.result_logged = False after each degradation.
    # - This ensures the new degraded image gets logged again when tested, instead of skipping because the previous result was already logged.

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Apply Blur"):
            st.session_state.current_test_image = apply_blur(image)
            st.session_state.current_degradation = "Blur"
            st.session_state.result_logged = False

    with col2:
        if st.button("Apply Low Light"):
            st.session_state.current_test_image = apply_low_light(image)
            st.session_state.current_degradation = "Low Light"
            st.session_state.result_logged = False

    with col3:
        if st.button("Apply Noise"):
            st.session_state.current_test_image = apply_noise(image)
            st.session_state.current_degradation = "Noise"
            st.session_state.result_logged = False

    with col4:
        if st.button("Rotate 20°"):
            st.session_state.current_test_image = apply_rotation(image)
            st.session_state.current_degradation = "Rotation"
            st.session_state.result_logged = False


    # use modified image for testing
    image = st.session_state.current_test_image
    st.image(image)
    st.write("Degradation Type:", st.session_state.current_degradation)

    # --------------------------------------------------
    # EMBEDDING EXTRACTION
    # --------------------------------------------------
    # Extract normalized embedding using helper function

    processed = preprocess_image(image)
    embedding = extract_embedding(embedding_model, processed)

    st.write("Non-zero embedding dimensions:", np.count_nonzero(embedding))

        # --------------------------------------------------
        # DRIFT SEVERITY CLASSIFICATION
        # --------------------------------------------------
    if threshold is None or threshold == 0:
        severity = "Undefined (Baseline threshold not set)"
        st.warning("Baseline threshold is invalid. Please recalibrate with more diverse clean images.")
    else:
        ratio = distance / threshold
        if ratio < 0.6:
            severity = "Stable"
        elif ratio < 0.8:
            severity = "Minor Deviation"
        elif ratio < 1.0:
                severity = "Warning: Approaching Drift"
        else:
            severity = "Drift Detected"

        st.write("Drift Severity:", severity)
    # --------------------------------------------------
    # STORE DISTANCE HISTORY
    # --------------------------------------------------
    # Used for plotting drift monitoring graph

    st.session_state.distance_history.append(distance)
    if len(st.session_state.distance_history) > 20:
        st.session_state.distance_history.pop(0)

    # --------------------------------------------------
    # EARLY DRIFT TREND WARNING
    # --------------------------------------------------
    if len(st.session_state.distance_history) >= 5:
        recent = st.session_state.distance_history[-5:]
        avg_recent = np.mean(recent)

        if threshold * 0.8 < avg_recent < threshold:
            st.warning("⚠ Drift trend increasing (approaching threshold)")

    # --------------------------------------------------
    # DISPLAY FINAL DRIFT RESULT
    # --------------------------------------------------
    st.write("Embedding Distance:", round(distance, 6))
    st.write("Drift Threshold:", round(threshold, 6))

    if drift:
        st.error("⚠ Drift Detected: Deviation beyond learned baseline tolerance")
    else:
        st.success("✅ No Drift: Within statistical tolerance")

    # --------------------------------------------------
    # STORE LABELS FOR CONFUSION MATRIX
    # --------------------------------------------------
    if not st.session_state.result_logged:
        # True label: 0 = No Drift, 1 = Drift
        true_label = 0 if st.session_state.current_degradation == "Clean" else 1
        # Predicted label: 0 = No Drift, 1 = Drift
        predicted_label = 1 if drift else 0

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

    df_trend = pd.DataFrame({"Distance": st.session_state.distance_history})

    fig, ax = plt.subplots()
    ax.plot(df_trend["Distance"], marker="o", label="Embedding Distance")

    if "last_threshold" in st.session_state and st.session_state.last_threshold is not None:
        threshold = st.session_state.last_threshold
        ax.axhline(y=threshold, linestyle="--", label="Drift Threshold")
        ax.axhspan(threshold * 0.8, threshold, alpha=0.2, label="Warning Zone")
        ax.axhspan(0, threshold * 0.8, alpha=0.1, label="Stable Zone")

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
        df_log = pd.DataFrame(
            st.session_state.experiment_log,
            columns=["Filename", "Distance", "Threshold", "True Label", "Predicted Label"]
        )
    st.dataframe(df_log)

    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Experiment Results",
        csv,
        "drift_experiment_results.csv",
        "text/csv"
    )
else:
    st.info("No experiments logged yet.")

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
st.subheader("📊 Model Evaluation")

if len(st.session_state.true_labels) > 0 and len(st.session_state.predicted_labels) > 0:
    if st.button("Generate Confusion Matrix"):
        cm, acc = evaluate_results(
            st.session_state.true_labels,
            st.session_state.predicted_labels
        )
        st.write("Accuracy:", round(acc, 4))
        # st.write("→ Accuracy shows the overall proportion of correct predictions out of all cases.")

        precision, recall, f1 = compute_classification_metrics(
            st.session_state.true_labels,
            st.session_state.predicted_labels
        )
        st.write("Precision:", round(precision, 4))
        # st.write("→ Precision reflects how many of the predicted drift cases were actually true drift (low precision means more false alarms).")

        st.write("Recall:", round(recall, 4))
        # st.write("→ Recall measures how many of the actual drift cases were correctly detected (high recall means fewer misses).")

        st.write("F1 Score:", round(f1, 4))
        # st.write("→ F1 balances precision and recall into a single score, useful when classes are imbalanced.")

        fig = plot_confusion_matrix(cm)
        st.pyplot(fig)

    # ROC curve button is separate
    if st.button("Show ROC Curve"):
        fig_roc = plot_roc_curve(
            st.session_state.true_labels,
            st.session_state.predicted_labels
        )
        st.pyplot(fig_roc)
else:
    st.warning("Run experiments first to generate evaluation metrics")

#     # --------------------------------------------------
#     # MODEL EVALUATION
#     # --------------------------------------------------
# st.subheader("📊 Model Evaluation")

# if st.button("Generate Confusion Matrix"):
#     if len(st.session_state.true_labels) > 0:
#         cm, acc = evaluate_results(
#             st.session_state.true_labels,
#             st.session_state.predicted_labels
#         )
#         st.write("Model Accuracy:", round(acc, 4))
#         fig = plot_confusion_matrix(cm)
#         st.pyplot(fig)
#     else:
#         st.warning("Run experiments first to generate evaluation metrics")

    # --------------------------------------------------
    # CLEAR EXPERIMENT LOG
    # --------------------------------------------------
    # if st.button("Clear Experiment Log"):
    #     st.session_state.experiment_log = []
    if st.button("Clear Experiment Log"):
        st.session_state.experiment_log = []
        st.session_state.true_labels = []
        st.session_state.predicted_labels = []
        st.success("Experiment log cleared.")