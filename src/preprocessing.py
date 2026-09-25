"""
VoIP Telemetry Data Preprocessing Module.

Handles dataset cleaning, validation, missing value imputation,
feature selection, scaling (StandardScaler), and persistence.
Ensures unified transformation between offline batch training and online real-time inference.
"""

import os
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Standard expected numerical features for the GMM model
DEFAULT_FEATURES: List[str] = [
    "packet_loss_pct",
    "latency_ms",
    "jitter_ms",
    "packet_drop_count",
    "throughput_kbps",
    "delay_variation_ms"
]

class VoIPPreprocessor:
    """
    Standardized preprocessing pipeline for Voice Trunk Anomaly Detection.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or DEFAULT_FEATURES
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_means = {}
        self.feature_stds = {}

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate whether the incoming DataFrame meets minimum schema requirements.
        """
        if df is None or df.empty:
            return False, "Uploaded DataFrame is empty."

        if "timestamp" not in df.columns:
            return False, "Missing required telemetry column: timestamp"

        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            return False, f"Missing required telemetry columns: {', '.join(missing_cols)}"

        # Check if features are numeric or convertible
        for col in self.feature_names:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    pd.to_numeric(df[col])
                except Exception:
                    return False, f"Column '{col}' contains non-numeric values that cannot be parsed."

        return True, "Validation successful."

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame:
        - Casts features to float
        - Removes duplicate timestamps if present
        - Handles missing values via median imputation
        - Enforces realistic non-negative physical bounds
        """
        df_clean = df.copy()

        # Deduplicate if timestamp is available
        if "timestamp" in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        else:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)

        for col in self.feature_names:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            # Median imputation for NaNs
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df_clean[col] = df_clean[col].fillna(median_val)

            # Domain physical constraints: all network metrics cannot be negative
            df_clean[col] = df_clean[col].clip(lower=0.0)

        # Packet loss percentage cannot exceed 100%
        if "packet_loss_pct" in df_clean.columns:
            df_clean["packet_loss_pct"] = df_clean["packet_loss_pct"].clip(upper=100.0)

        return df_clean

    def fit(self, df: pd.DataFrame) -> "VoIPPreprocessor":
        """
        Fit the StandardScaler on training data.
        """
        valid, msg = self.validate_dataframe(df)
        if not valid:
            raise ValueError(f"Data validation error: {msg}")

        df_clean = self.clean_data(df)
        X = df_clean[self.feature_names].values
        self.scaler.fit(X)
        self.is_fitted = True

        for i, col in enumerate(self.feature_names):
            self.feature_means[col] = float(self.scaler.mean_[i])
            self.feature_stds[col] = float(self.scaler.scale_[i])

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform cleaned feature DataFrame into scaled numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        valid, msg = self.validate_dataframe(df)
        if not valid:
            raise ValueError(f"Data validation error: {msg}")

        df_clean = self.clean_data(df)
        X = df_clean[self.feature_names].values
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in a single call.
        """
        return self.fit(df).transform(df)

    def transform_single(self, sample_dict: dict) -> np.ndarray:
        """
        Transform a single telemetry dictionary (used during real-time monitoring).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming samples.")

        row = []
        for feat in self.feature_names:
            val = float(sample_dict.get(feat, 0.0))
            val = max(0.0, val)
            if feat == "packet_loss_pct":
                val = min(100.0, val)
            row.append(val)

        X_single = np.array([row])
        return self.scaler.transform(X_single)

    def save(self, filepath: str) -> None:
        """
        Save preprocessor to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "VoIPPreprocessor":
        """
        Load preprocessor instance from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found at: {filepath}")
        return joblib.load(filepath)
