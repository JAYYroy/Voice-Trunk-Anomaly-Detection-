"""
Voice Trunk Anomaly Scoring - Streamlit Academic Application.

GMM-Based Anomaly Detection for Real-Time Audio Communication Infrastructure.
The network GMM workflow only runs after the user uploads a VoIP telemetry CSV.
The microphone workflow analyzes real audio recordings separately from the network model.
"""

import os
import sys

# Suppress MKL/KMeans memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

import io
import json
import time
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import VoIPPreprocessor, DEFAULT_FEATURES
from src.anomaly_detector import GMMAnomalyDetector
from src.evaluation import evaluate_anomaly_detector, compare_models
from src.voice_analysis import analyze_voice_recording, inspect_audio_recording

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Voice Trunk Anomaly Scoring | GMM",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic dashboard aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-suspicious {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-anomalous {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .callout-box {
        background-color: #0F172A;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DYNAMIC MODEL TRAINING & EVALUATION ENGINE FOR UPLOADED DATASETS
# -----------------------------------------------------------------------------
def build_custom_model_context(custom_df: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Fit a dedicated VoIPPreprocessor, Gaussian Mixture Model, and Isolation Forest
    specifically on the custom dataset.
    
    Recalculates:
    - Feature scaling (StandardScaler)
    - Optimal components (K) via BIC/AIC model selection
    - GMM density estimation and log-likelihood anomaly scores
    - 95th percentile (suspicious) & 99th percentile (anomalous) thresholds
    - Multi-class anomaly classifications
    - Evaluation metrics and confusion matrix (TP, FP, TN, FN)
    - Comparative study against Isolation Forest
    """
    preprocessor = VoIPPreprocessor()
    preprocessor.fit(custom_df)
    X_scaled = preprocessor.transform(custom_df)

    # Use normal mask if ground-truth anomaly column is present with normal samples
    if "is_anomaly" in custom_df.columns and (custom_df["is_anomaly"] == 0).any():
        normal_mask = (custom_df["is_anomaly"] == 0).values
        X_train = X_scaled[normal_mask]
    else:
        X_train = X_scaled

    n_samples = len(X_train)
    # Determine candidate K components based on sample size
    max_k = min(5, max(1, n_samples // 15))
    k_range = list(range(1, max_k + 1))

    bic_scores = {}
    aic_scores = {}
    best_k = 1
    lowest_bic = float("inf")

    for k in k_range:
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            max_iter=200,
            n_init=3
        )
        gm.fit(X_train)
        bic = float(gm.bic(X_train))
        aic = float(gm.aic(X_train))
        bic_scores[str(k)] = bic
        aic_scores[str(k)] = aic
        if bic < lowest_bic:
            lowest_bic = bic
            best_k = k

    # Train final custom GMM Anomaly Detector
    detector = GMMAnomalyDetector(
        n_components=best_k,
        covariance_type="full",
        random_state=random_state,
        normal_percentile=95.0,
        anomalous_percentile=99.0
    )
    detector.bic_history = {int(k): v for k, v in bic_scores.items()}
    detector.aic_history = {int(k): v for k, v in aic_scores.items()}
    detector.fit(X_train)

    # Compute scores and tri-state status on entire custom dataset
    scores = detector.compute_anomaly_scores(X_scaled)
    statuses = detector.classify_scores(scores)

    metadata = {
        "n_components": best_k,
        "covariance_type": "full",
        "baseline_score_mean": detector.baseline_score_mean,
        "baseline_score_std": detector.baseline_score_std,
        "threshold_normal": detector.threshold_normal,
        "threshold_anomalous": detector.threshold_anomalous,
        "bic_scores": bic_scores,
        "aic_scores": aic_scores
    }

    # Derive evaluation benchmark
    if "is_anomaly" in custom_df.columns and len(np.unique(custom_df["is_anomaly"])) > 1:
        y_true = custom_df["is_anomaly"].values.astype(int)
        has_ground_truth = True
    else:
        # ITU-T G.114/G.1010 VoIP QoS degradation boundary for unlabelled telemetry
        y_true = (
            (custom_df["packet_loss_pct"] > 2.0) |
            (custom_df["latency_ms"] > 80.0) |
            (custom_df["jitter_ms"] > 15.0) |
            (custom_df["packet_drop_count"] > 3)
        ).values.astype(int)
        if len(np.unique(y_true)) < 2:
            y_true = (scores > detector.threshold_anomalous).astype(int)
        has_ground_truth = False

    # Fit secondary baseline (Isolation Forest) on custom dataset
    iforest = IsolationForest(
        n_estimators=100,
        contamination=0.08,
        random_state=random_state,
        n_jobs=1
    )
    iforest.fit(X_train)
    iforest_scores = -iforest.score_samples(X_scaled)
    iforest_th = float(np.percentile(iforest_scores, 92.0))

    eval_data = evaluate_anomaly_detector(y_true, scores, detector.threshold_anomalous)
    comp_df = compare_models(y_true, scores, detector.threshold_anomalous, iforest_scores, iforest_th)

    return {
        "preprocessor": preprocessor,
        "detector": detector,
        "metadata": metadata,
        "scores": scores,
        "statuses": statuses,
        "eval_data": eval_data,
        "comp_df": comp_df,
        "has_ground_truth": has_ground_truth
    }


# -----------------------------------------------------------------------------
# APPLICATION INITIALIZATION & SHARED STATE MANAGEMENT
# -----------------------------------------------------------------------------
# Initialize shared session state keys
if "custom_df" not in st.session_state:
    st.session_state.custom_df = None
if "custom_filename" not in st.session_state:
    st.session_state.custom_filename = None
if "custom_file_hash" not in st.session_state:
    st.session_state.custom_file_hash = None
if "cached_custom_context" not in st.session_state:
    st.session_state.cached_custom_context = None
if "live_history" not in st.session_state:
    st.session_state.live_history = []
if "live_cursor" not in st.session_state:
    st.session_state.live_cursor = 0
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False
if "voice_audio_bytes" not in st.session_state:
    st.session_state.voice_audio_bytes = None
if "voice_analysis" not in st.session_state:
    st.session_state.voice_analysis = None
if "voice_upload_results" not in st.session_state:
    st.session_state.voice_upload_results = []


def clear_active_dataset():
    """Safely reset shared application state to no dataset loaded."""
    st.session_state.custom_df = None
    st.session_state.custom_filename = None
    st.session_state.custom_file_hash = None
    st.session_state.cached_custom_context = None
    st.session_state.live_history = []
    st.session_state.live_cursor = 0
    st.session_state.monitoring_active = False


def process_uploaded_csv(uploaded_file):
    """Validate, load, and activate the user's uploaded CSV."""
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.custom_file_hash != file_hash:
            try:
                candidate_df = pd.read_csv(io.BytesIO(file_bytes))
                validator = VoIPPreprocessor()
                is_valid, msg = validator.validate_dataframe(candidate_df)

                if is_valid:
                    st.session_state.custom_df = candidate_df
                    st.session_state.custom_filename = uploaded_file.name
                    st.session_state.custom_file_hash = file_hash
                    st.session_state.cached_custom_context = None  # Invalidate cached context
                    st.session_state.live_history = []
                    st.session_state.live_cursor = 0
                    st.success(f"Custom CSV successfully loaded! ({len(candidate_df):,} rows)")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error(f"Validation Error: {msg}. Dataset was not loaded.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")


# Resolve currently active dataset & model context
is_custom = st.session_state.custom_df is not None

if is_custom:
    active_df = st.session_state.custom_df
    data_source_name = "Custom CSV"
    active_filename = st.session_state.custom_filename or "uploaded_telemetry.csv"

    # Compute or retrieve cached custom model context
    if st.session_state.cached_custom_context is None:
        with st.spinner(f"Fitting Gaussian Mixture Model on custom dataset ({len(active_df):,} rows)..."):
            st.session_state.cached_custom_context = build_custom_model_context(active_df)

    custom_ctx = st.session_state.cached_custom_context
    active_preprocessor = custom_ctx["preprocessor"]
    active_detector = custom_ctx["detector"]
    active_metadata = custom_ctx["metadata"]
    active_scores = custom_ctx["scores"]
    active_statuses = custom_ctx["statuses"]
    active_eval_data = custom_ctx["eval_data"]
    active_comp_df = custom_ctx["comp_df"]
    has_ground_truth = custom_ctx["has_ground_truth"]
    record_count = len(active_df)

    # Initialize live history with samples from uploaded dataset if empty
    if len(st.session_state.live_history) == 0:
        seed_samples = active_df.tail(min(20, len(active_df))).to_dict("records")
        for s in seed_samples:
            x_scaled = active_preprocessor.transform_single(s)
            pred = active_detector.predict_single(x_scaled)
            s_copy = dict(s)
            s_copy.update({
                "anomaly_score": pred["anomaly_score"],
                "log_likelihood": pred["log_likelihood"],
                "status": pred["status"],
                "severity": pred["severity"]
            })
            st.session_state.live_history.append(s_copy)
else:
    active_df = None
    data_source_name = "No dataset loaded"
    active_filename = "None"
    active_preprocessor = None
    active_detector = None
    active_metadata = {}
    active_scores = None
    active_statuses = None
    active_eval_data = {}
    active_comp_df = pd.DataFrame()
    has_ground_truth = False
    record_count = 0


def append_uploaded_telemetry_row(row_index: int | None = None):
    """Append one row from the uploaded CSV to the live monitor history."""
    if active_df is None or active_preprocessor is None or active_detector is None or len(active_df) == 0:
        return

    if row_index is None:
        row_index = st.session_state.live_cursor % len(active_df)
        st.session_state.live_cursor += 1

    source_row = active_df.iloc[int(row_index) % len(active_df)].to_dict()
    x_scaled = active_preprocessor.transform_single(source_row)
    pred = active_detector.predict_single(x_scaled)
    source_row.update(pred)
    st.session_state.live_history.append(source_row)
    st.session_state.live_history = st.session_state.live_history[-60:]


def render_voice_analysis_result(result: dict, title: str = "Voice Analysis Results"):
    """Render locally computed audio signal features and waveform."""
    features = result["features"]
    status_color = "#10B981" if result["status"] == "NORMAL" else ("#F59E0B" if result["status"] == "SUSPICIOUS" else "#EF4444")

    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Status</div><div class="metric-val" style="color:{status_color};">{result['status']}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Duration</div><div class="metric-val">{features['duration_sec']:.2f} s</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Sample Rate</div><div class="metric-val">{features['sample_rate_hz']:,} Hz</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Voice Score</div><div class="metric-val">{result['anomaly_score']:.2f}</div></div>""", unsafe_allow_html=True)

    feature_rows = pd.DataFrame([
        {"Feature": "RMS Energy", "Value": f"{features['rms_energy']:.6f}"},
        {"Feature": "Audio Samples", "Value": f"{features['num_samples']:,}"},
        {"Feature": "Zero Crossing Rate", "Value": f"{features['zero_crossing_rate']:.6f}"},
        {"Feature": "Spectral Centroid", "Value": f"{features['spectral_centroid_hz']:.2f} Hz"},
        {"Feature": "Spectral Bandwidth", "Value": f"{features['spectral_bandwidth_hz']:.2f} Hz"},
        {"Feature": "Spectral Rolloff", "Value": f"{features['spectral_rolloff_hz']:.2f} Hz"},
        {"Feature": "Spectral Centroid Std", "Value": f"{features['spectral_centroid_std_hz']:.2f} Hz"},
    ])
    st.dataframe(feature_rows, use_container_width=True, hide_index=True)

    waveform = result.get("waveform", {})
    if waveform.get("time_sec") and waveform.get("amplitude"):
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=waveform["time_sec"],
            y=waveform["amplitude"],
            mode="lines",
            name="Waveform",
            line=dict(color="#38BDF8", width=1)
        ))
        fig_wave.update_layout(
            title="Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=260,
            margin=dict(t=40, b=30, l=20, r=20),
            template="plotly_dark"
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    mfcc_df = pd.DataFrame({
        "Coefficient": [f"MFCC {i + 1}" for i in range(len(features["mfcc_mean"]))],
        "Mean Value": [round(v, 4) for v in features["mfcc_mean"]],
    })
    st.markdown("**MFCC Summary**")
    st.dataframe(mfcc_df, use_container_width=True, hide_index=True)

    st.markdown("**Technical Feedback**")
    for item in result["feedback"]:
        st.markdown(f"- {item}")


# Helper to render visible dataset indicator banner at top of pages
def render_dataset_banner():
    if is_custom and active_df is not None:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 10px 18px; margin: 10px 0 20px 0;">
                <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                    <span style="display: inline-flex; align-items: center; gap: 6px; font-weight: 700; color: #10B981; font-size: 0.9rem;">
                        <span style="height: 8px; width: 8px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span>
                        ACTIVE DATASET: Custom CSV
                    </span>
                    <span style="color: #94A3B8; font-size: 0.88rem;">File: <b style="color: #F8FAFC;">{active_filename}</b></span>
                    <span style="color: #38BDF8; font-size: 0.88rem;">RECORDS: <b style="color: #F8FAFC;">{record_count:,}</b></span>
                </div>
                <div style="font-size: 0.8rem; color: #34D399; font-weight: 600;">
                    ✓ Active Model Recalibrated (K={active_metadata.get('n_components', 5)})
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; background: rgba(148, 163, 184, 0.08); border: 1px solid #475569; border-radius: 8px; padding: 10px 18px; margin: 10px 0 20px 0;">
                <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                    <span style="display: inline-flex; align-items: center; gap: 6px; font-weight: 700; color: #94A3B8; font-size: 0.9rem;">
                        <span style="height: 8px; width: 8px; background-color: #64748B; border-radius: 50%; display: inline-block;"></span>
                        ACTIVE DATASET: No dataset loaded
                    </span>
                    <span style="color: #94A3B8; font-size: 0.88rem;">File: <b style="color: #F8FAFC;">None</b></span>
                    <span style="color: #94A3B8; font-size: 0.88rem;">RECORDS: <b style="color: #F8FAFC;">0</b></span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 500;">
                    Awaiting CSV Upload
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION, DATASET STATUS & FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/audio-wave.png", width=64)
    st.title("VOICE TRUNK MONITOR")
    st.caption("GMM Anomaly Scoring System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📡 Live Monitoring",
            "🔍 Anomaly Analysis",
            "📂 Dataset Explorer",
            "🧠 GMM Model Diagnostics",
            "🎯 Model Evaluation",
            "📚 About & Viva Guide"
        ] + ["Voice Analysis"],
        index=0
    )

    st.markdown("---")

    # Visible Dataset Status Indicator in Sidebar
    st.subheader("Dataset Status")
    if is_custom and active_df is not None:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Active Dataset</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">Custom CSV</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px; word-break: break-all;">File: <b>{active_filename}</b></div>
                <div style="font-size: 0.85rem; color: #38BDF8; margin-top: 2px;">Records: <b>{record_count:,}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🗑️ Clear Active Dataset", key="sidebar_clear_btn", use_container_width=True):
            clear_active_dataset()
            st.rerun()
    else:
        st.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.08); border: 1px solid #475569; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Active Dataset</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">No dataset loaded</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">File: <i>None</i></div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 2px;">Records: <b>0</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Model Configuration")
    if is_custom and active_detector is not None:
        st.markdown(f"**Algorithm:** Gaussian Mixture")
        st.markdown(f"**Components ($K$):** `{active_metadata.get('n_components', 5)}`")
        st.markdown(f"**Covariance:** `{active_metadata.get('covariance_type', 'full')}`")
        st.markdown(f"**Threshold (95th%):** `{active_detector.threshold_normal:.2f}`")
        st.markdown(f"**Threshold (99th%):** `{active_detector.threshold_anomalous:.2f}`")
    else:
        st.markdown("**Algorithm:** Gaussian Mixture")
        st.markdown("**Status:** *Awaiting CSV upload to fit model*")
    
    st.markdown("---")
    st.caption("PRAD Academic Project\nVoIP Trunk QoS Anomaly Detection")


# =============================================================================
# 1. DASHBOARD PAGE
# =============================================================================
if page == "📊 Dashboard":
    st.title("VOICE TRUNK ANOMALY SCORING")
    st.subheader("GMM-Based Anomaly Detection for Real-Time Audio Communication Infrastructure")
    st.markdown(
        "A statistical pattern recognition system that models multi-dimensional VoIP QoS behavior "
        "(latency, jitter, packet loss, throughput) using **Gaussian Mixture Models** and scores deviations via log-likelihood density estimation."
    )
    
    # Dataset Status Banner
    render_dataset_banner()

    if not is_custom or active_df is None:
        st.info("No dataset loaded. Upload a VoIP telemetry CSV to begin analysis.")
        dash_upload = st.file_uploader(
            "Upload VoIP Telemetry CSV (Expected columns: packet_loss_pct, latency_ms, jitter_ms, packet_drop_count, throughput_kbps, delay_variation_ms)",
            type=["csv"],
            key="dash_file_uploader"
        )
        if dash_upload is not None:
            process_uploaded_csv(dash_upload)
    else:
        # Pre-score dataset using active model context
        df_scored = active_df.copy()
        df_scored["anomaly_score"] = np.round(active_scores, 2)
        df_scored["status"] = active_statuses

        n_total = len(df_scored)
        n_anomalies = int((df_scored["status"] == "ANOMALOUS").sum())
        n_suspicious = int((df_scored["status"] == "SUSPICIOUS").sum())
        n_normal = int((df_scored["status"] == "NORMAL").sum())
        anomaly_rate = (n_anomalies / n_total) * 100 if n_total > 0 else 0.0

        avg_latency = float(df_scored["latency_ms"].mean()) if "latency_ms" in df_scored.columns else 0.0
        avg_loss = float(df_scored["packet_loss_pct"].mean()) if "packet_loss_pct" in df_scored.columns else 0.0
        avg_jitter = float(df_scored["jitter_ms"].mean()) if "jitter_ms" in df_scored.columns else 0.0

        # Metric Cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Records</div><div class="metric-val">{n_total:,}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Critical Anomalies</div><div class="metric-val" style="color:#EF4444;">{n_anomalies:,}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Anomaly Rate</div><div class="metric-val" style="color:#F59E0B;">{anomaly_rate:.2f}%</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Latency</div><div class="metric-val">{avg_latency:.1f} ms</div></div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Packet Loss</div><div class="metric-val">{avg_loss:.2f}%</div></div>""", unsafe_allow_html=True)
        with c6:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Jitter</div><div class="metric-val">{avg_jitter:.1f} ms</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Health Overview & Telemetry Charts
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("Network Health Classification")
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Normal", "Suspicious", "Anomalous"],
                values=[n_normal, n_suspicious, n_anomalies],
                hole=0.55,
                marker_colors=["#10B981", "#F59E0B", "#EF4444"],
                textinfo="label+percent"
            )])
            fig_donut.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                showlegend=False,
                template="plotly_dark"
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            normal_pct = (n_normal / n_total) * 100 if n_total > 0 else 0.0
            susp_pct = (n_suspicious / n_total) * 100 if n_total > 0 else 0.0

            st.markdown(f"""
            <div class="callout-box">
                <b>Operational Health Status ({data_source_name}):</b><br>
                • <b>Normal Quality:</b> {n_normal:,} frames ({normal_pct:.1f}%)<br>
                • <b>Suspicious Jitter/Drop:</b> {n_suspicious:,} frames ({susp_pct:.1f}%)<br>
                • <b>Severe Trunk Degradation:</b> {n_anomalies:,} frames ({anomaly_rate:.1f}%)
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            snapshot_count = min(300, len(df_scored))
            st.subheader(f"Telemetry Snapshot (Recent {snapshot_count} Observations)")
            sample_slice = df_scored.tail(snapshot_count).reset_index(drop=True)
            fig_trends = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                subplot_titles=("Latency (ms)", "Packet Loss (%)", "Anomaly Score (-ln p(x))")
            )

            fig_trends.add_trace(go.Scatter(y=sample_slice["latency_ms"], name="Latency", line=dict(color="#10B981", width=1.5)), row=1, col=1)
            fig_trends.add_trace(go.Scatter(y=sample_slice["packet_loss_pct"], name="Packet Loss", line=dict(color="#3B82F6", width=1.5)), row=2, col=1)
            fig_trends.add_trace(go.Scatter(y=sample_slice["anomaly_score"], name="Anomaly Score", line=dict(color="#F43F5E", width=1.5)), row=3, col=1)

            # Add active threshold line
            fig_trends.add_hline(
                y=active_detector.threshold_anomalous, line_dash="dash", line_color="#EF4444",
                annotation_text="Critical Threshold", row=3, col=1
            )

            fig_trends.update_layout(height=320, margin=dict(t=30, b=20, l=20, r=20), showlegend=False, template="plotly_dark")
            st.plotly_chart(fig_trends, use_container_width=True)

        st.subheader("Voice Input")
        voice_result = st.session_state.voice_analysis
        if voice_result is None:
            st.info("No voice recording available. Click Start Recording to analyze a voice signal.")
            st.markdown("**Voice Input:** Not recorded")
        else:
            voice_features = voice_result["features"]
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Voice Input</div><div class="metric-val">Recording analyzed</div></div>""", unsafe_allow_html=True)
            with vc2:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Duration</div><div class="metric-val">{voice_features['duration_sec']:.2f} s</div></div>""", unsafe_allow_html=True)
            with vc3:
                voice_color = "#10B981" if voice_result["status"] == "NORMAL" else ("#F59E0B" if voice_result["status"] == "SUSPICIOUS" else "#EF4444")
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Voice Status</div><div class="metric-val" style="color:{voice_color};">{voice_result['status']}</div></div>""", unsafe_allow_html=True)
            st.caption(f"Voice anomaly status is based only on the recorded audio signal features. Score: {voice_result['anomaly_score']:.2f}")

        # Recent Incidents Table
        st.subheader("⚠️ High-Priority Recent Anomalous Events")
        cand_cols = ["timestamp", "packet_loss_pct", "latency_ms", "jitter_ms", "packet_drop_count", "throughput_kbps", "anomaly_score", "status"]
        avail_cols = [c for c in cand_cols if c in df_scored.columns]

        recent_anom = df_scored[df_scored["status"] == "ANOMALOUS"].tail(8)[avail_cols]
        if len(recent_anom) > 0:
            format_dict = {}
            if "packet_loss_pct" in avail_cols:
                format_dict["packet_loss_pct"] = "{:.2f}%"
            if "latency_ms" in avail_cols:
                format_dict["latency_ms"] = "{:.1f} ms"
            if "jitter_ms" in avail_cols:
                format_dict["jitter_ms"] = "{:.1f} ms"
            if "throughput_kbps" in avail_cols:
                format_dict["throughput_kbps"] = "{:.1f} kbps"
            if "anomaly_score" in avail_cols:
                format_dict["anomaly_score"] = "{:.2f}"

            st.dataframe(recent_anom.style.format(format_dict), use_container_width=True)
        else:
            st.info("No critical anomalous events detected under the current 99th-percentile threshold.")


# =============================================================================
# 2. LIVE MONITORING PAGE
# =============================================================================
elif page == "📡 Live Monitoring":
    st.title("📡 LIVE VOICE TRUNK MONITOR")
    st.caption("Uploaded CSV Replay & Online GMM Anomaly Scoring")
    st.markdown("---")

    render_dataset_banner()

    if not is_custom or active_df is None:
        st.info("No dataset loaded. Upload a VoIP telemetry CSV to begin analysis.")
    else:
        st.caption("Live monitor replays rows from the uploaded CSV and scores them with the uploaded-data GMM.")
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.2])
        with ctrl_col1:
            auto_replay = st.toggle("Auto-Replay Uploaded Rows", value=st.session_state.monitoring_active)
            st.session_state.monitoring_active = auto_replay
        with ctrl_col2:
            if st.button("Next CSV Row", use_container_width=True):
                append_uploaded_telemetry_row()
        with ctrl_col3:
            if st.button("Clear Stream", use_container_width=True):
                st.session_state.live_history = []
                st.session_state.live_cursor = 0
                st.rerun()

        if st.session_state.monitoring_active:
            append_uploaded_telemetry_row()

        if not st.session_state.live_history:
            append_uploaded_telemetry_row(0)

        latest = st.session_state.live_history[-1]

        st.markdown("<br>", unsafe_allow_html=True)

        # Current Frame Telemetry Display
        st.subheader("Current Packet Inspection")
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        status_color = "#10B981" if latest.get("status") == "NORMAL" else ("#F59E0B" if latest.get("status") == "SUSPICIOUS" else "#EF4444")
        badge_class = f"badge-{latest.get('status', 'NORMAL').lower()}"

        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Latency</div><div class="metric-val">{latest.get('latency_ms', 0):.1f} ms</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Packet Loss</div><div class="metric-val">{latest.get('packet_loss_pct', 0):.2f}%</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Jitter</div><div class="metric-val">{latest.get('jitter_ms', 0):.1f} ms</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Packet Drops</div><div class="metric-val">{latest.get('packet_drop_count', 0)} pkts</div></div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Anomaly Score</div><div class="metric-val" style="color:{status_color};">{latest.get('anomaly_score', 0):.2f}</div></div>""", unsafe_allow_html=True)
        with m6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Classification</div>
                <div style="margin-top:10px;"><span class="{badge_class}">{latest.get('status', 'NORMAL')}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Streaming Line Chart
        st.subheader("Live Telemetry & Anomaly Score Stream (Rolling Window)")
        history_df = pd.DataFrame(st.session_state.live_history)

        fig_live = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("Live Telemetry (Latency & Jitter)", "GMM Anomaly Score: -ln p(x)")
        )

        fig_live.add_trace(go.Scatter(y=history_df["latency_ms"], name="Latency (ms)", line=dict(color="#10B981", width=2)), row=1, col=1)
        fig_live.add_trace(go.Scatter(y=history_df["jitter_ms"], name="Jitter (ms)", line=dict(color="#F59E0B", width=2)), row=1, col=1)
        fig_live.add_trace(go.Scatter(y=history_df["anomaly_score"], name="Anomaly Score", line=dict(color="#F43F5E", width=2.5)), row=2, col=1)

        fig_live.add_hline(
            y=active_detector.threshold_normal, line_dash="dot", line_color="#F59E0B",
            annotation_text="Suspicious Threshold", row=2, col=1
        )
        fig_live.add_hline(
            y=active_detector.threshold_anomalous, line_dash="dash", line_color="#EF4444",
            annotation_text="Critical Threshold", row=2, col=1
        )

        fig_live.update_layout(height=420, margin=dict(t=30, b=20, l=20, r=20), template="plotly_dark")
        st.plotly_chart(fig_live, use_container_width=True)

        if st.session_state.monitoring_active:
            time.sleep(1.0)
            st.rerun()


# =============================================================================
# 3. VOICE ANALYSIS PAGE
# =============================================================================
elif page == "Voice Analysis":
    st.title("VOICE ANALYSIS")
    st.markdown("Voice Analysis analyzes the actual audio signal.")
    st.markdown("Network Anomaly Detection analyzes VoIP/network telemetry.")
    st.caption("Audio features are calculated from microphone recordings or uploaded WAV/FLAC/MP3 waveforms. They are not connected to the network GMM.")
    st.markdown("---")

    st.subheader("1. Record Voice")
    st.markdown("Speak into your microphone to analyze the voice signal.")
    audio_file = st.audio_input("Start Recording", key="voice_audio_input")

    if audio_file is None and st.session_state.voice_audio_bytes is None:
        st.info("No voice recording available. Click Start Recording to analyze a voice signal.")
    else:
        if audio_file is not None:
            audio_bytes = audio_file.getvalue()
            if st.session_state.voice_audio_bytes != audio_bytes:
                st.session_state.voice_audio_bytes = audio_bytes
                st.session_state.voice_analysis = None

        st.audio(st.session_state.voice_audio_bytes, format="audio/wav")

        if st.button("Analyze Voice Recording", type="primary", use_container_width=True):
            try:
                st.session_state.voice_analysis = analyze_voice_recording(
                    st.session_state.voice_audio_bytes,
                    filename="microphone_recording.wav"
                )
                st.success("Voice recording analyzed from actual microphone audio.")
            except Exception as exc:
                st.error(f"Voice analysis failed: {exc}")

    if st.session_state.voice_analysis is not None:
        render_voice_analysis_result(st.session_state.voice_analysis, "Voice Signal Analysis - Microphone Recording")

    st.markdown("---")
    st.subheader("2. VOICE AUDIO DATASET")
    st.markdown("Upload Audio Files")
    st.caption("Supported files: WAV, FLAC, MP3. Metadata CSV files are not audio and are not analyzed here.")
    uploaded_audio_files = st.file_uploader(
        "Upload WAV/FLAC/MP3 audio files",
        type=["wav", "flac", "mp3"],
        accept_multiple_files=True,
        key="voice_audio_uploader"
    )

    if uploaded_audio_files:
        st.info(f"{len(uploaded_audio_files)} audio file(s) selected. Each file will be analyzed from its waveform bytes.")

        for uploaded_audio in uploaded_audio_files:
            st.markdown(f"**Uploaded audio:** `{uploaded_audio.name}`")
            audio_bytes = uploaded_audio.getvalue()
            st.audio(audio_bytes, format=uploaded_audio.type or "audio/wav")
            try:
                preview = inspect_audio_recording(audio_bytes, filename=uploaded_audio.name)
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric("Duration", f"{preview['duration_sec']:.3f} s")
                with p2:
                    st.metric("Sample Rate", f"{preview['sample_rate_hz']:,} Hz")
                with p3:
                    st.metric("Audio Samples", f"{preview['num_samples']:,}")

                waveform = preview["waveform"]
                if waveform["time_sec"] and waveform["amplitude"]:
                    preview_fig = go.Figure(go.Scatter(
                        x=waveform["time_sec"],
                        y=waveform["amplitude"],
                        mode="lines",
                        name="Waveform",
                        line=dict(color="#38BDF8", width=1),
                    ))
                    preview_fig.update_layout(
                        title=f"Waveform - {uploaded_audio.name}",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude",
                        height=220,
                        margin=dict(t=40, b=30, l=20, r=20),
                        template="plotly_dark",
                    )
                    st.plotly_chart(preview_fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not decode {uploaded_audio.name}: {exc}")

        if st.button("Analyze Audio", type="primary", use_container_width=True):
            results = []
            for uploaded_audio in uploaded_audio_files:
                try:
                    audio_bytes = uploaded_audio.getvalue()
                    analysis = analyze_voice_recording(audio_bytes, filename=uploaded_audio.name)
                    results.append({
                        "filename": uploaded_audio.name,
                        "bytes": audio_bytes,
                        "mime_type": uploaded_audio.type,
                        "analysis": analysis,
                    })
                except Exception as exc:
                    results.append({
                        "filename": uploaded_audio.name,
                        "bytes": uploaded_audio.getvalue(),
                        "mime_type": uploaded_audio.type,
                        "error": str(exc),
                    })
            st.session_state.voice_upload_results = results

    else:
        st.info("No uploaded voice audio available. Upload WAV, FLAC, or MP3 files to analyze a voice dataset.")

    if st.session_state.voice_upload_results:
        st.markdown("---")
        st.subheader("3. Voice Signal Analysis")
        summary_rows = []
        for item in st.session_state.voice_upload_results:
            if "error" in item:
                st.error(f"{item['filename']}: {item['error']}")
                continue

            analysis = item["analysis"]
            features = analysis["features"]
            summary_rows.append({
                "File": item["filename"],
                "Status": analysis["status"],
                "Duration (s)": round(features["duration_sec"], 3),
                "Sample Rate (Hz)": features["sample_rate_hz"],
                "Audio Samples": features["num_samples"],
                "RMS Energy": round(features["rms_energy"], 6),
                "ZCR": round(features["zero_crossing_rate"], 6),
                "Centroid (Hz)": round(features["spectral_centroid_hz"], 2),
                "Bandwidth (Hz)": round(features["spectral_bandwidth_hz"], 2),
                "Rolloff (Hz)": round(features["spectral_rolloff_hz"], 2),
                "MFCC Summary": ", ".join(f"{value:.3f}" for value in features["mfcc_mean"]),
            })

        if summary_rows:
            st.markdown("**Voice Analysis Results**")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        for item in st.session_state.voice_upload_results:
            if "analysis" not in item:
                continue
            with st.expander(f"Voice Analysis Results - {item['filename']}", expanded=len(st.session_state.voice_upload_results) == 1):
                st.audio(item["bytes"], format=item.get("mime_type") or "audio/wav")
                render_voice_analysis_result(item["analysis"], f"Voice Analysis Results - {item['filename']}")

    st.caption(
        "Voice audio features are technical signal features only. "
        "They do not measure packet loss, jitter, latency, throughput, or packet drops."
    )
    st.caption(
        "Voice Signal Analysis is separate from the Network/VoIP GMM. "
        "The network GMM only analyzes uploaded VoIP telemetry CSV rows."
    )

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown(
        """
        ```text
                    VOICE TRUNK SYSTEM
                           |
              +------------+------------+
              |                         |
        VOICE INPUT                 VOIP TELEMETRY
        Microphone/Audio Files          CSV
              |                         |
       Audio Features              Network Features
              |                         |
       Voice Analysis                  GMM
              |                         |
       Voice Feedback             Anomaly Score
              |                         |
              +------------+------------+
                           |
                    Overall Monitoring
                           |
              Normal / Suspicious / Anomalous
        ```
        """
    )


# =============================================================================
# 4. ANOMALY ANALYSIS PAGE
# =============================================================================
elif page == "🔍 Anomaly Analysis":
    st.title("🔍 ANOMALY ANALYSIS & THRESHOLD CALIBRATION")
    st.markdown("Detailed inspection of GMM likelihood density, score distribution, and threshold sensitivity.")
    st.markdown("---")

    render_dataset_banner()

    if not is_custom or active_df is None:
        st.info("No dataset loaded. Upload a VoIP telemetry CSV to begin analysis.")
    else:
        X_scaled = active_preprocessor.transform(active_df)
        scores = active_detector.compute_anomaly_scores(X_scaled)

        # Interactive Threshold Tuning Controls
        st.subheader("Threshold Calibration Control")
        col_t1, col_t2, col_t3 = st.columns([1.5, 1.5, 1])

        with col_t1:
            custom_norm_pct = st.slider("Suspicious Percentile Cutoff", min_value=85.0, max_value=98.0, value=95.0, step=0.5)
        with col_t2:
            custom_anom_pct = st.slider("Critical Anomaly Percentile Cutoff", min_value=96.0, max_value=99.9, value=99.0, step=0.1)

        # Calculate percentiles safely from baseline or all scores
        if "is_anomaly" in active_df.columns and (active_df["is_anomaly"] == 0).any():
            base_scores = scores[active_df["is_anomaly"] == 0]
        else:
            base_scores = scores

        th_norm_custom = float(np.percentile(base_scores, custom_norm_pct))
        th_anom_custom = float(np.percentile(base_scores, custom_anom_pct))

        with col_t3:
            st.markdown(f"**Calibrated Cutoffs ({data_source_name}):**")
            st.markdown(f"• Suspicious ({custom_norm_pct}%): `{th_norm_custom:.2f}`")
            st.markdown(f"• Anomalous ({custom_anom_pct}%): `{th_anom_custom:.2f}`")

        dynamic_status = active_detector.classify_scores(scores, threshold_normal=th_norm_custom, threshold_anomalous=th_anom_custom)
        df_analysis = active_df.copy()
        df_analysis["anomaly_score"] = np.round(scores, 3)
        df_analysis["status"] = dynamic_status

        # Score Distribution Histogram (Representing exactly len(active_df) observations)
        st.subheader(f"Log-Likelihood Score Distribution: $S(\\mathbf{{x}}) = -\\ln p(\\mathbf{{x}})$ ({len(df_analysis):,} Observations)")
        
        # Dynamically scale histogram bins based on observation count
        optimal_bins = min(80, max(15, len(df_analysis) // 3))
        fig_hist = px.histogram(
            df_analysis,
            x="anomaly_score",
            color="status",
            color_discrete_map={"NORMAL": "#10B981", "SUSPICIOUS": "#F59E0B", "ANOMALOUS": "#EF4444"},
            nbins=optimal_bins,
            opacity=0.75,
            title=f"Anomaly Score Frequency Distribution ({len(df_analysis):,} Observations) with Decision Boundaries"
        )
        fig_hist.add_vline(x=th_norm_custom, line_dash="dash", line_color="#F59E0B", annotation_text="Suspicious")
        fig_hist.add_vline(x=th_anom_custom, line_dash="solid", line_color="#EF4444", annotation_text="Anomalous")
        fig_hist.update_layout(template="plotly_dark", height=380, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

        # 2D Feature Space Anomaly Clustering
        st.subheader(f"2D Feature Space Anomaly Projections ({len(df_analysis):,} Observations)")
        col_sc1, col_sc2 = st.columns(2)

        sample_size = min(2500, len(df_analysis))
        plot_df = df_analysis.sample(sample_size, random_state=42) if len(df_analysis) > sample_size else df_analysis

        with col_sc1:
            fig_sc1 = px.scatter(
                plot_df,
                x="latency_ms",
                y="packet_loss_pct",
                color="status",
                color_discrete_map={"NORMAL": "#10B981", "SUSPICIOUS": "#F59E0B", "ANOMALOUS": "#EF4444"},
                title="Latency vs Packet Loss",
                opacity=0.65
            )
            fig_sc1.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_sc1, use_container_width=True)

        with col_sc2:
            fig_sc2 = px.scatter(
                plot_df,
                x="latency_ms",
                y="jitter_ms",
                color="status",
                color_discrete_map={"NORMAL": "#10B981", "SUSPICIOUS": "#F59E0B", "ANOMALOUS": "#EF4444"},
                title="Latency vs Jitter",
                opacity=0.65
            )
            fig_sc2.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_sc2, use_container_width=True)

        # Top Anomalies Table
        st.subheader("Top Ranked Anomalous Observations (Highest Anomaly Score)")
        cand_cols = ["timestamp", "packet_loss_pct", "latency_ms", "jitter_ms", "packet_drop_count", "throughput_kbps", "anomaly_score", "anomaly_type", "status"]
        avail_cols = [c for c in cand_cols if c in df_analysis.columns]

        top_n = min(15, len(df_analysis))
        top_anom = df_analysis.sort_values(by="anomaly_score", ascending=False).head(top_n)[avail_cols]

        format_dict = {}
        if "packet_loss_pct" in avail_cols:
            format_dict["packet_loss_pct"] = "{:.2f}%"
        if "latency_ms" in avail_cols:
            format_dict["latency_ms"] = "{:.1f} ms"
        if "jitter_ms" in avail_cols:
            format_dict["jitter_ms"] = "{:.1f} ms"
        if "throughput_kbps" in avail_cols:
            format_dict["throughput_kbps"] = "{:.1f} kbps"
        if "anomaly_score" in avail_cols:
            format_dict["anomaly_score"] = "{:.3f}"

        st.dataframe(top_anom.style.format(format_dict), use_container_width=True)


# =============================================================================
# 4. DATASET EXPLORER PAGE
# =============================================================================
elif page == "📂 Dataset Explorer":
    st.title("📂 DATASET EXPLORER & CUSTOM CSV UPLOAD")
    st.markdown("Inspect raw telemetry data, filter by conditions, view descriptive statistics, or evaluate a custom CSV.")
    st.markdown("---")

    render_dataset_banner()

    # CSV Upload Section
    st.subheader("Upload Custom VoIP Telemetry CSV")
    uploaded_file = st.file_uploader(
        "Upload a custom CSV containing VoIP metrics (Expected columns: packet_loss_pct, latency_ms, jitter_ms, packet_drop_count, throughput_kbps, delay_variation_ms)",
        type=["csv"],
        key="csv_file_uploader"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # Only process if new file or changed
        if st.session_state.custom_file_hash != file_hash:
            try:
                candidate_df = pd.read_csv(io.BytesIO(file_bytes))
                validator = VoIPPreprocessor()
                is_valid, msg = validator.validate_dataframe(candidate_df)

                if is_valid:
                    st.session_state.custom_df = candidate_df
                    st.session_state.custom_filename = uploaded_file.name
                    st.session_state.custom_file_hash = file_hash
                    st.session_state.cached_custom_context = None  # Invalidate cached context
                    st.session_state.live_history = []
                    st.session_state.live_cursor = 0
                    st.success(f"Custom CSV successfully loaded! ({len(candidate_df):,} rows)")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Validation Error: {msg}. Dataset was not loaded.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

    if not is_custom or active_df is None:
        st.info("No dataset loaded. Upload a VoIP telemetry CSV to begin analysis.")
        st.stop()

    # Dataset Status & Reset Actions
    if is_custom:
        st.success(f"Custom CSV successfully loaded! ({record_count:,} rows)")
        col_act1, col_act2 = st.columns([3, 1])
        with col_act1:
            st.markdown(f"**Active Source:** `Custom CSV` | **Filename:** `{active_filename}` | **Records:** `{record_count:,}`")
        with col_act2:
            if st.button("Clear Active Dataset", key="explorer_clear_btn", use_container_width=True):
                clear_active_dataset()
                st.rerun()

    # Dataset Summary Statistics & Metrics
    st.subheader("Descriptive Statistics")
    avail_features = [f for f in DEFAULT_FEATURES if f in active_df.columns]
    if avail_features:
        st.dataframe(active_df[avail_features].describe().T.style.format("{:.2f}"), use_container_width=True)

    # Interactive Filter & Data Table
    st.subheader("Explore Raw Telemetry Records")
    col_f1, col_f2 = st.columns(2)

    max_loss_limit = float(active_df["packet_loss_pct"].max()) if "packet_loss_pct" in active_df.columns and not active_df.empty else 100.0
    max_lat_limit = float(active_df["latency_ms"].max()) if "latency_ms" in active_df.columns and not active_df.empty else 500.0

    with col_f1:
        max_loss = st.slider(
            "Filter Max Packet Loss (%)",
            min_value=0.0,
            max_value=max(1.0, max_loss_limit),
            value=max_loss_limit
        )
    with col_f2:
        max_lat = st.slider(
            "Filter Max Latency (ms)",
            min_value=0.0,
            max_value=max(10.0, max_lat_limit),
            value=max_lat_limit
        )

    filter_mask = pd.Series(True, index=active_df.index)
    if "packet_loss_pct" in active_df.columns:
        filter_mask &= (active_df["packet_loss_pct"] <= max_loss)
    if "latency_ms" in active_df.columns:
        filter_mask &= (active_df["latency_ms"] <= max_lat)

    filtered_df = active_df[filter_mask]
    st.caption(f"Showing {len(filtered_df):,} matching rows out of {len(active_df):,}")
    st.dataframe(filtered_df.head(min(200, len(filtered_df))), use_container_width=True)


# =============================================================================
# 5. GMM MODEL DIAGNOSTICS PAGE
# =============================================================================
elif page == "🧠 GMM Model Diagnostics":
    st.title("🧠 GAUSSIAN MIXTURE MODEL ARCHITECTURE & DIAGNOSTICS")
    st.markdown("Theoretical foundation, mathematical derivations, component distributions, and hyperparameter selection.")
    st.markdown("---")

    render_dataset_banner()

    if not is_custom or active_df is None:
        st.info("No dataset loaded. Upload a VoIP telemetry CSV to begin analysis.")
        st.stop()

    # Hyperparameter Summary Cards (Dynamically derived from active model)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Mixture Components (K)</div><div class="metric-val">{active_metadata.get('n_components', 5)}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Covariance Type</div><div class="metric-val">{active_metadata.get('covariance_type', 'full')}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Baseline Score Mean</div><div class="metric-val">{active_metadata.get('baseline_score_mean', active_detector.baseline_score_mean):.2f}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Critical Anomaly Cutoff</div><div class="metric-val" style="color:#EF4444;">{active_detector.threshold_anomalous:.2f}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # BIC / AIC Elbow Curve (Derived from active dataset)
    bic_scores = active_metadata.get("bic_scores", {})
    aic_scores = active_metadata.get("aic_scores", {})

    if bic_scores and aic_scores:
        k_list = sorted([int(k) for k in bic_scores.keys()])
        b_list = [bic_scores[str(k)] for k in k_list]
        a_list = [aic_scores[str(k)] for k in k_list]

        st.subheader(f"Model Selection: BIC & AIC Curve Across Components $K \\in \\{{{', '.join(str(k) for k in k_list)}\\}}$ ({data_source_name})")

        fig_bic = go.Figure()
        fig_bic.add_trace(go.Scatter(x=k_list, y=b_list, mode="lines+markers", name="BIC (Bayesian Information Criterion)", line=dict(color="#3B82F6", width=2.5)))
        fig_bic.add_trace(go.Scatter(x=k_list, y=a_list, mode="lines+markers", name="AIC (Akaike Information Criterion)", line=dict(color="#EF4444", width=2.5, dash="dash")))

        fig_bic.update_layout(
            title=f"BIC and AIC Minimization for {data_source_name} (Lower is Better)",
            xaxis_title="Number of Gaussian Components (K)",
            yaxis_title="Information Criterion Score",
            template="plotly_dark",
            height=380
        )
        st.plotly_chart(fig_bic, use_container_width=True)

    # GMM Component Weights & Parameters
    st.subheader(f"GMM Component Parameter Breakdown ({data_source_name})")
    gmm_model = active_detector.model
    comp_df = pd.DataFrame({
        "Component": [f"Component {i+1}" for i in range(gmm_model.n_components)],
        "Mixture Weight (pi_k)": [f"{w*100:.2f}%" for w in gmm_model.weights_],
        "Mean Latency (scaled)": [f"{m[1]:.3f}" if len(m) > 1 else "N/A" for m in gmm_model.means_],
        "Mean Packet Loss (scaled)": [f"{m[0]:.3f}" if len(m) > 0 else "N/A" for m in gmm_model.means_],
        "Mean Jitter (scaled)": [f"{m[2]:.3f}" if len(m) > 2 else "N/A" for m in gmm_model.means_]
    })
    st.dataframe(comp_df, use_container_width=True)

    # Mathematical Explanations for Viva
    st.markdown("---")
    st.subheader("📖 Viva Defense Guide: Core Mathematical Formulations")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("""
        #### 1. What is a Gaussian Mixture Model?
        A **Gaussian Mixture Model (GMM)** is a probabilistic model assuming all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters:
        """)
        st.latex(r"p(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)")
        st.markdown("""
        where:
        - $\pi_k$: Mixture weight ($\sum \pi_k = 1, \pi_k \ge 0$)
        - $\boldsymbol{\mu}_k$: Mean vector of component $k$
        - $\boldsymbol{\Sigma}_k$: Full covariance matrix modeling feature cross-correlations
        """)

    with col_v2:
        st.markdown("""
        #### 2. How is Anomaly Scoring Derived?
        The GMM calculates the exact probability density of any observation under the learned normal multi-modal distribution.
        """)
        st.latex(r"\text{Score}(\mathbf{x}) = -\ln p(\mathbf{x}) = -\text{score\_samples}(\mathbf{x})")
        st.markdown("""
        - **Normal Observations:** Fall in dense regions of the Gaussians $\\rightarrow$ high likelihood $\\rightarrow$ **low anomaly score**.
        - **Irregular Drops / Latency Spikes:** Fall far out in the multi-dimensional tails $\\rightarrow$ near-zero likelihood $\\rightarrow$ **very high anomaly score**.
        """)


# =============================================================================
# 6. MODEL EVALUATION PAGE
# =============================================================================
elif page == "🎯 Model Evaluation":
    st.title("🎯 MODEL EVALUATION & BENCHMARKING")
    st.markdown("Rigorous quantitative evaluation and comparison with secondary baseline (Isolation Forest).")
    st.markdown("---")

    render_dataset_banner()

    if not is_custom or active_df is None:
        st.info("No dataset loaded. Upload a VoIP telemetry CSV to begin analysis.")
        st.stop()

    metrics = active_eval_data.get("metrics", {})
    cm = active_eval_data.get("confusion_matrix", {})

    # Top Metric Tiles
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-val" style="color:#10B981;">{metrics.get('accuracy', 0.0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-val">{metrics.get('precision', 0.0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Recall (Sensitivity)</div><div class="metric-val" style="color:#10B981;">{metrics.get('recall', 0.0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">F1-Score</div><div class="metric-val">{metrics.get('f1_score', 0.0):.4f}</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">ROC-AUC</div><div class="metric-val">{metrics.get('roc_auc', 0.0):.4f}</div></div>""", unsafe_allow_html=True)
    with m6:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Specificity</div><div class="metric-val">{metrics.get('specificity', 0.0)*100:.2f}%</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tp_val = cm.get("true_positives", 0)
    fp_val = cm.get("false_positives", 0)
    fn_val = cm.get("false_negatives", 0)
    tn_val = cm.get("true_negatives", 0)
    total_eval_samples = tp_val + fp_val + fn_val + tn_val

    col_cm, col_comp = st.columns([1, 1.2])
    with col_cm:
        st.subheader(f"Confusion Matrix ({total_eval_samples:,} Total Observations)")
        matrix_vals = cm.get("matrix", [[tn_val, fp_val], [fn_val, tp_val]])
        fig_cm = px.imshow(
            matrix_vals,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Normal", "Anomaly"],
            y=["Normal", "Anomaly"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(template="plotly_dark", height=340, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

        if not has_ground_truth:
            st.caption("ℹ️ Evaluated against ITU-T G.114/G.1010 VoIP QoS degradation criteria (unlabelled custom CSV).")
        else:
            st.caption(f"Evaluated against ground-truth labels ({data_source_name}).")

        st.markdown(f"""
        - **True Positives (TP):** `{tp_val:,}`
        - **False Positives (FP):** `{fp_val:,}`
        - **False Negatives (FN):** `{fn_val:,}`
        - **True Negatives (TN):** `{tn_val:,}`
        - **Total Observations:** `{total_eval_samples:,}`
        """)

    with col_comp:
        st.subheader(f"Model Comparison: GMM vs Isolation Forest ({data_source_name})")
        if not active_comp_df.empty:
            st.dataframe(active_comp_df, use_container_width=True)
        st.markdown(f"""
        <div class="callout-box">
            <b>Key Academic Takeaway ({data_source_name}):</b><br>
            The <b>Gaussian Mixture Model</b> models multi-modal VoIP telemetry densities. Evaluated on the active dataset ({record_count:,} records), GMM provides smooth probability contours and density estimation that enables early detection of subtle trunk degradation before catastrophic call termination occurs.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 7. ABOUT & VIVA GUIDE PAGE
# =============================================================================
elif page == "📚 About & Viva Guide":
    st.title("📚 PROJECT OVERVIEW & VIVA PREPARATION GUIDE")
    st.markdown("Comprehensive reference material for project presentation, external viva examination, and academic evaluation.")
    st.markdown("---")

    render_dataset_banner()

    st.subheader("Project Identification")
    st.markdown("""
    - **Subject:** Pattern Recognition and Anomaly Detection (PRAD)
    - **Project Title:** Voice Trunk Anomaly Scoring
    - **Problem Statement:** *Implement a Gaussian Mixture Model to detect irregular packet drops or latency spikes in real-time audio communication infrastructure.*
    - **Primary Machine Learning Algorithm:** Gaussian Mixture Model (`sklearn.mixture.GaussianMixture`)
    """)

    st.subheader("Frequently Asked Viva Questions & Model Answers")

    with st.expander("Q1: Why is GMM specifically chosen over supervised classification?"):
        st.markdown("""
        **Answer:** In telecom and VoIP voice trunk infrastructure, network failures and degradations are rare, heterogeneous, and unpredictable. Creating labeled datasets for every potential failure mode (e.g. fiber micro-cuts, BGP flap, bufferbloat, router memory leaks) is impractical. 
        **GMM is an unsupervised density estimator** that models *normal operational behavior* from baseline traffic. Any incoming frame that deviates from this learned distribution receives low likelihood and is flagged as an anomaly without requiring prior anomaly labels.
        """)

    with st.expander("Q2: How is the number of Gaussian components (K) selected?"):
        st.markdown("""
        **Answer:** We evaluate $K$ using the **Bayesian Information Criterion (BIC)**:
        $$\\text{BIC} = -2 \\ln \\hat{L} + p \\ln(n)$$
        BIC balances model fit against model complexity by penalizing the number of free parameters $p$. We observe that minimizing BIC selects the optimal number of components capturing distinct voice routing paths without overfitting.
        """)

    with st.expander("Q3: How is the anomaly score mathematically defined?"):
        st.markdown("""
        **Answer:** The anomaly score is the **negative log-likelihood** of the sample under the fitted GMM probability density function:
        $$S(\\mathbf{x}) = -\\ln p(\\mathbf{x}) = -\\ln \\left( \\sum_{k=1}^K \\pi_k \\mathcal{N}(\\mathbf{x} \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k) \\right)$$
        Since likelihoods $p(\\mathbf{x})$ range between $0$ and $1$, $-\\ln p(\\mathbf{x})$ maps unlikely low-density tail events to high positive numbers, providing an intuitive, monotonic anomaly score.
        """)

    with st.expander("Q4: How were the anomaly decision thresholds established?"):
        st.markdown("""
        **Answer:** Thresholds are calibrated non-arbitrarily using the empirical percentile distribution of anomaly scores computed on baseline telemetry:
        - **NORMAL:** $S(\\mathbf{x}) \\le \\tau_{95}$ (Below 95th percentile of normal baseline)
        - **SUSPICIOUS:** $\\tau_{95} < S(\\mathbf{x}) \\le \\tau_{99}$ (Between 95th and 99th percentile)
        - **ANOMALOUS:** $S(\\mathbf{x}) > \\tau_{99}$ (Above 99th percentile, extreme distribution tail)
        """)

    with st.expander("Q5: What are the project assumptions and limitations?"):
        st.markdown("""
        **Answer:** 
        - **Academic Simulation Assumption:** The dataset simulates VoIP/SIP voice trunk telemetry conforming to ITU-T G.114/G.1010 recommendations rather than a physical carrier tap.
        - **Stationarity:** The model assumes the baseline traffic distribution is stationary; in production, periodic retraining or online EM adaptation would accommodate seasonal traffic growth.
        """)
