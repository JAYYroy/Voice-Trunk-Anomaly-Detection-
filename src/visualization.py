"""
Visualization Module for Voice Trunk Anomaly Scoring.

Generates publication-quality static charts (saved to outputs/plots/)
and interactive Plotly charts for the Streamlit dashboard.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Style configuration for Matplotlib/Seaborn
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#F1F5F9"
plt.rcParams["grid.linestyle"] = "--"


def save_all_eda_plots(df: pd.DataFrame, output_dir: str = "outputs/plots") -> List[str]:
    """
    Generate and save all essential Exploratory Data Analysis (EDA) charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # 1. Packet Loss Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["packet_loss_pct"], kde=True, color="#2563EB", bins=40, ax=ax)
    ax.set_title("VoIP Packet Loss Distribution (%)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Packet Loss Percentage (%)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    p1 = os.path.join(output_dir, "01_packet_loss_distribution.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved_files.append(p1)

    # 2. Latency Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["latency_ms"], kde=True, color="#059669", bins=45, ax=ax)
    ax.set_title("Voice Trunk One-Way Latency Distribution (ms)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    p2 = os.path.join(output_dir, "02_latency_distribution.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved_files.append(p2)

    # 3. Jitter Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["jitter_ms"], kde=True, color="#D97706", bins=40, ax=ax)
    ax.set_title("Voice Trunk Jitter Distribution (ms)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Jitter (ms)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    p3 = os.path.join(output_dir, "03_jitter_distribution.png")
    fig.tight_layout()
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved_files.append(p3)

    # 4. Correlation Heatmap
    num_cols = ["packet_loss_pct", "latency_ms", "jitter_ms", "packets_sent", "packets_received", "packet_drop_count", "throughput_kbps", "delay_variation_ms"]
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("VoIP Telemetry Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    p4 = os.path.join(output_dir, "04_correlation_heatmap.png")
    fig.tight_layout()
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved_files.append(p4)

    # Sample a slice for clear time-series visibility (first 600 seconds)
    df_slice = df.iloc[:600].copy()

    # 5. Latency Over Time
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df_slice.index, df_slice["latency_ms"], color="#059669", lw=1.2, label="Latency (ms)")
    if "is_anomaly" in df_slice.columns:
        anom_idx = df_slice[df_slice["is_anomaly"] == 1].index
        ax.scatter(anom_idx, df_slice.loc[anom_idx, "latency_ms"], color="#DC2626", s=25, zorder=5, label="Ground-Truth Anomaly")
    ax.set_title("Voice Trunk Latency Time Series (Initial 600s Sample)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timeline (Seconds)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.legend(loc="upper right")
    p5 = os.path.join(output_dir, "05_latency_over_time.png")
    fig.tight_layout()
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    saved_files.append(p5)

    # 6. Packet Loss Over Time
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df_slice.index, df_slice["packet_loss_pct"], color="#2563EB", lw=1.2, label="Packet Loss (%)")
    if "is_anomaly" in df_slice.columns:
        anom_idx = df_slice[df_slice["is_anomaly"] == 1].index
        ax.scatter(anom_idx, df_slice.loc[anom_idx, "packet_loss_pct"], color="#DC2626", s=25, zorder=5, label="Ground-Truth Anomaly")
    ax.set_title("VoIP Packet Loss Over Time (Initial 600s Sample)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timeline (Seconds)", fontsize=11)
    ax.set_ylabel("Packet Loss (%)", fontsize=11)
    ax.legend(loc="upper right")
    p6 = os.path.join(output_dir, "06_packet_loss_over_time.png")
    fig.tight_layout()
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    saved_files.append(p6)

    # 7. Jitter Over Time
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df_slice.index, df_slice["jitter_ms"], color="#D97706", lw=1.2, label="Jitter (ms)")
    if "is_anomaly" in df_slice.columns:
        anom_idx = df_slice[df_slice["is_anomaly"] == 1].index
        ax.scatter(anom_idx, df_slice.loc[anom_idx, "jitter_ms"], color="#DC2626", s=25, zorder=5, label="Ground-Truth Anomaly")
    ax.set_title("Voice Trunk Jitter Over Time (Initial 600s Sample)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timeline (Seconds)", fontsize=11)
    ax.set_ylabel("Jitter (ms)", fontsize=11)
    ax.legend(loc="upper right")
    p7 = os.path.join(output_dir, "07_jitter_over_time.png")
    fig.tight_layout()
    fig.savefig(p7, dpi=300)
    plt.close(fig)
    saved_files.append(p7)

    # 8. Scatter Plot: Latency vs Packet Loss
    fig, ax = plt.subplots(figsize=(8, 6))
    if "anomaly_type" in df.columns:
        sns.scatterplot(
            data=df,
            x="latency_ms",
            y="packet_loss_pct",
            hue="anomaly_type",
            palette={"Normal": "#10B981", "High Packet Loss": "#3B82F6", "High Latency": "#8B5CF6", "High Jitter": "#F59E0B", "Combined Degradation": "#EF4444", "Packet Drop Burst": "#EC4899"},
            alpha=0.6,
            s=30,
            ax=ax
        )
    else:
        sns.scatterplot(data=df, x="latency_ms", y="packet_loss_pct", color="#2563EB", alpha=0.5, ax=ax)
    ax.set_title("Feature Space: Latency vs Packet Loss", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("Packet Loss (%)", fontsize=11)
    p8 = os.path.join(output_dir, "08_scatter_latency_vs_packet_loss.png")
    fig.tight_layout()
    fig.savefig(p8, dpi=300)
    plt.close(fig)
    saved_files.append(p8)

    # 9. Scatter Plot: Latency vs Jitter
    fig, ax = plt.subplots(figsize=(8, 6))
    if "anomaly_type" in df.columns:
        sns.scatterplot(
            data=df,
            x="latency_ms",
            y="jitter_ms",
            hue="anomaly_type",
            palette={"Normal": "#10B981", "High Packet Loss": "#3B82F6", "High Latency": "#8B5CF6", "High Jitter": "#F59E0B", "Combined Degradation": "#EF4444", "Packet Drop Burst": "#EC4899"},
            alpha=0.6,
            s=30,
            ax=ax
        )
    else:
        sns.scatterplot(data=df, x="latency_ms", y="jitter_ms", color="#D97706", alpha=0.5, ax=ax)
    ax.set_title("Feature Space: Latency vs Jitter", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("Jitter (ms)", fontsize=11)
    p9 = os.path.join(output_dir, "09_scatter_latency_vs_jitter.png")
    fig.tight_layout()
    fig.savefig(p9, dpi=300)
    plt.close(fig)
    saved_files.append(p9)

    return saved_files


def save_model_diagnostics_plots(
    bic_scores: Dict[int, float],
    aic_scores: Dict[int, float],
    anomaly_scores: np.ndarray,
    th_normal: float,
    th_anomalous: float,
    cm: np.ndarray,
    gmm_roc: Dict[str, list],
    iforest_roc: Optional[Dict[str, list]] = None,
    output_dir: str = "outputs/plots"
) -> List[str]:
    """
    Generate and save model selection, score distribution, confusion matrix, and ROC curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # 10. BIC & AIC Model Selection Curve
    k_vals = list(bic_scores.keys())
    bic_vals = [bic_scores[k] for k in k_vals]
    aic_vals = [aic_scores[k] for k in k_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, bic_vals, marker="o", lw=2, color="#2563EB", label="BIC (Bayesian Information Criterion)")
    ax.plot(k_vals, aic_vals, marker="s", lw=2, linestyle="--", color="#DC2626", label="AIC (Akaike Information Criterion)")
    best_k = k_vals[int(np.argmin(bic_vals))]
    ax.axvline(best_k, color="#10B981", linestyle=":", lw=2, label=f"Selected K={best_k} (Min BIC)")
    ax.set_title("GMM Hyperparameter Selection: BIC & AIC vs Number of Components", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Gaussian Mixture Components (K)", fontsize=11)
    ax.set_ylabel("Information Criterion Score", fontsize=11)
    ax.set_xticks(k_vals)
    ax.legend(loc="upper right")
    p10 = os.path.join(output_dir, "10_gmm_bic_aic_selection.png")
    fig.tight_layout()
    fig.savefig(p10, dpi=300)
    plt.close(fig)
    saved_files.append(p10)

    # 11. Anomaly Score Distribution with Thresholds
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(anomaly_scores, bins=60, kde=True, color="#4F46E5", ax=ax)
    ax.axvline(th_normal, color="#F59E0B", linestyle="--", lw=2, label=f"Suspicious Threshold ({th_normal:.2f})")
    ax.axvline(th_anomalous, color="#DC2626", linestyle="-", lw=2, label=f"Critical Anomaly Threshold ({th_anomalous:.2f})")
    ax.set_title("Log-Likelihood Anomaly Score Distribution: -ln p(x)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Anomaly Score S(x) = -ln p(x)", fontsize=11)
    ax.set_ylabel("Sample Frequency", fontsize=11)
    ax.legend(loc="upper right")
    p11 = os.path.join(output_dir, "11_anomaly_score_distribution.png")
    fig.tight_layout()
    fig.savefig(p11, dpi=300)
    plt.close(fig)
    saved_files.append(p11)

    # 12. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Normal", "Anomaly"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("GMM Anomaly Detection Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("Ground Truth Class", fontsize=11)
    p12 = os.path.join(output_dir, "12_confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(p12, dpi=300)
    plt.close(fig)
    saved_files.append(p12)

    # 13. ROC Curve Comparison
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gmm_roc["fpr"], gmm_roc["tpr"], color="#2563EB", lw=2.2, label=f"GMM (ROC-AUC = {gmm_roc.get('auc', 0.99):.4f})")
    if iforest_roc:
        ax.plot(iforest_roc["fpr"], iforest_roc["tpr"], color="#F59E0B", lw=2, linestyle="--", label=f"Isolation Forest (ROC-AUC = {iforest_roc.get('auc', 0.95):.4f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", lw=1.5, label="Random Guess (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.legend(loc="lower right")
    p13 = os.path.join(output_dir, "13_roc_curve_comparison.png")
    fig.tight_layout()
    fig.savefig(p13, dpi=300)
    plt.close(fig)
    saved_files.append(p13)

    return saved_files
