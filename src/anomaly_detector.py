"""
Gaussian Mixture Model (GMM) Anomaly Detection Engine.

Implements unsupervised density estimation for VoIP network anomaly scoring.
Observations in low-density regions of the learned probability distribution receive
low log-likelihood values, which map directly to high anomaly scores.
"""

import os
import json
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib


class GMMAnomalyDetector:
    """
    Voice Trunk Anomaly Detector using Gaussian Mixture Models.
    
    Mathematical Formulation:
    - Probability Density:
        p(x) = sum_{k=1}^K pi_k * N(x | mu_k, Sigma_k)
    - Log-Likelihood:
        ll(x) = ln p(x)  (via GaussianMixture.score_samples)
    - Anomaly Score:
        S(x) = -ll(x) = -ln p(x)
      (A higher score indicates the observation is far in the distribution tails)
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
        normal_percentile: float = 95.0,
        anomalous_percentile: float = 99.0
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.normal_percentile = normal_percentile
        self.anomalous_percentile = anomalous_percentile

        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
            n_init=3
        )

        self.is_fitted = False
        self.threshold_normal: float = 0.0
        self.threshold_anomalous: float = 0.0
        self.baseline_score_mean: float = 0.0
        self.baseline_score_std: float = 0.0
        self.bic_history: Dict[int, float] = {}
        self.aic_history: Dict[int, float] = {}

    def fit(self, X: np.ndarray) -> "GMMAnomalyDetector":
        """
        Fit the Gaussian Mixture Model on baseline (normal) feature data.
        """
        self.model.fit(X)
        self.is_fitted = True

        # Calculate baseline scores on training distribution
        log_likelihoods = self.model.score_samples(X)
        anomaly_scores = -log_likelihoods

        self.baseline_score_mean = float(np.mean(anomaly_scores))
        self.baseline_score_std = float(np.std(anomaly_scores))

        # Calibrate thresholds from baseline distribution percentiles
        self.threshold_normal = float(np.percentile(anomaly_scores, self.normal_percentile))
        self.threshold_anomalous = float(np.percentile(anomaly_scores, self.anomalous_percentile))

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return the per-sample log-likelihood: ln p(x).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.model.score_samples(X)

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly score S(x) = -ln p(x).
        """
        return -self.score_samples(X)

    def classify_scores(
        self,
        scores: np.ndarray,
        threshold_normal: Optional[float] = None,
        threshold_anomalous: Optional[float] = None
    ) -> List[str]:
        """
        Classify observations into tri-state status:
        - NORMAL: score <= threshold_normal
        - SUSPICIOUS: threshold_normal < score <= threshold_anomalous
        - ANOMALOUS: score > threshold_anomalous
        """
        th_norm = self.threshold_normal if threshold_normal is None else threshold_normal
        th_anom = self.threshold_anomalous if threshold_anomalous is None else threshold_anomalous

        statuses = []
        for s in scores:
            if s <= th_norm:
                statuses.append("NORMAL")
            elif s <= th_anom:
                statuses.append("SUSPICIOUS")
            else:
                statuses.append("ANOMALOUS")
        return statuses

    def predict_single(
        self,
        x_scaled: np.ndarray,
        threshold_normal: Optional[float] = None,
        threshold_anomalous: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Score and classify a single scaled observation (shape: (1, n_features)).
        """
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)

        ll = float(self.model.score_samples(x_scaled)[0])
        score = -ll
        th_norm = self.threshold_normal if threshold_normal is None else threshold_normal
        th_anom = self.threshold_anomalous if threshold_anomalous is None else threshold_anomalous

        if score <= th_norm:
            status = "NORMAL"
            severity = "Low"
        elif score <= th_anom:
            status = "SUSPICIOUS"
            severity = "Medium"
        else:
            status = "ANOMALOUS"
            severity = "High"

        responsibilities = self.model.predict_proba(x_scaled)[0].tolist()

        return {
            "log_likelihood": round(ll, 4),
            "anomaly_score": round(score, 4),
            "status": status,
            "severity": severity,
            "threshold_normal": round(th_norm, 4),
            "threshold_anomalous": round(th_anom, 4),
            "component_responsibilities": [round(r, 4) for r in responsibilities]
        }

    def save(self, filepath: str) -> None:
        """
        Save detector artifact to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "GMMAnomalyDetector":
        """
        Load detector instance from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        return joblib.load(filepath)


def build_custom_model_context(custom_df: Any, random_state: int = 42) -> Dict[str, Any]:
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
    from src.preprocessing import VoIPPreprocessor
    from src.evaluation import evaluate_anomaly_detector, compare_models
    from sklearn.ensemble import IsolationForest

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

