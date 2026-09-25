"""
Evaluation Module for Voice Trunk Anomaly Detection.

Computes classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC),
Confusion Matrix breakdown (TP, FP, TN, FN), and comparison with secondary
baseline models (Isolation Forest).
Distinguishes strictly between unsupervised model training and supervised ground-truth evaluation.
"""

import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def evaluate_anomaly_detector(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Evaluate binary anomaly detection performance given continuous anomaly scores
    and a decision threshold.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth labels (0 = normal, 1 = anomaly).
    anomaly_scores : np.ndarray
        Continuous anomaly scores (higher = more anomalous).
    threshold : float
        Cutoff threshold where scores > threshold are predicted as anomaly (1).

    Returns:
    --------
    dict containing all performance metrics and curves.
    """
    y_pred = (anomaly_scores > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, anomaly_scores))
    except Exception:
        pr_auc = 0.0

    fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, anomaly_scores)

    return {
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": cm.tolist()
        },
        "counts": {
            "total_samples": len(y_true),
            "ground_truth_anomalies": int(np.sum(y_true)),
            "detected_anomalies": int(np.sum(y_pred)),
            "anomaly_rate_detected_pct": round(float(np.mean(y_pred) * 100), 2),
            "anomaly_rate_actual_pct": round(float(np.mean(y_true) * 100), 2)
        },
        "curves": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "pr_precision": pr_precision.tolist(),
            "pr_recall": pr_recall.tolist()
        },
        "classification_report": classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], output_dict=True)
    }


def compare_models(
    y_true: np.ndarray,
    gmm_scores: np.ndarray,
    gmm_threshold: float,
    iforest_scores: np.ndarray,
    iforest_threshold: float
) -> pd.DataFrame:
    """
    Compare GMM performance against secondary baseline (Isolation Forest).
    """
    gmm_results = evaluate_anomaly_detector(y_true, gmm_scores, gmm_threshold)
    iforest_results = evaluate_anomaly_detector(y_true, iforest_scores, iforest_threshold)

    comp = {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "PR-AUC",
            "True Positives (TP)",
            "False Positives (FP)",
            "False Negatives (FN)",
            "True Negatives (TN)"
        ],
        "GMM (Primary)": [
            f"{gmm_results['metrics']['accuracy'] * 100:.2f}%",
            f"{gmm_results['metrics']['precision'] * 100:.2f}%",
            f"{gmm_results['metrics']['recall'] * 100:.2f}%",
            f"{gmm_results['metrics']['f1_score']:.4f}",
            f"{gmm_results['metrics']['roc_auc']:.4f}",
            f"{gmm_results['metrics']['pr_auc']:.4f}",
            gmm_results["confusion_matrix"]["true_positives"],
            gmm_results["confusion_matrix"]["false_positives"],
            gmm_results["confusion_matrix"]["false_negatives"],
            gmm_results["confusion_matrix"]["true_negatives"]
        ],
        "Isolation Forest (Baseline)": [
            f"{iforest_results['metrics']['accuracy'] * 100:.2f}%",
            f"{iforest_results['metrics']['precision'] * 100:.2f}%",
            f"{iforest_results['metrics']['recall'] * 100:.2f}%",
            f"{iforest_results['metrics']['f1_score']:.4f}",
            f"{iforest_results['metrics']['roc_auc']:.4f}",
            f"{iforest_results['metrics']['pr_auc']:.4f}",
            iforest_results["confusion_matrix"]["true_positives"],
            iforest_results["confusion_matrix"]["false_positives"],
            iforest_results["confusion_matrix"]["false_negatives"],
            iforest_results["confusion_matrix"]["true_negatives"]
        ]
    }
    return pd.DataFrame(comp)
