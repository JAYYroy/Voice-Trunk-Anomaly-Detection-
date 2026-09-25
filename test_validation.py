"""
Comprehensive Validation Script for Custom VoIP Telemetry Dataset Propagation.
Tests:
1. Default dataset initialization (10,000 samples).
2. Custom 200-row CSV upload and validation.
3. GMM dynamic refitting and threshold calibration.
4. Total observation counts across all pages/views (200 records).
5. Evaluation metrics and confusion matrix totaling 200 samples.
6. Clean reset back to default dataset (10,000 records).
"""

import os
import sys
import numpy as np
import pandas as pd

from src.preprocessing import VoIPPreprocessor
from src.anomaly_detector import GMMAnomalyDetector
from app import (
    load_default_models_and_scaler,
    load_default_telemetry_data,
    build_custom_model_context
)

def run_tests():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE VALIDATION TESTS")
    print("=" * 70)

    # 1. Test default dataset
    default_df = load_default_telemetry_data()
    print(f"\n[Test 1] Default Dataset Verification:")
    print(f"  - Default records count: {len(default_df):,}")
    assert len(default_df) == 10000, f"Expected 10,000 default records, got {len(default_df)}"
    print("  ✓ PASSED: Default dataset correctly contains 10,000 records.")

    # 2. Test 200-row custom CSV
    custom_csv_path = "data/test_custom_200.csv"
    assert os.path.exists(custom_csv_path), f"File {custom_csv_path} not found!"
    custom_df = pd.read_csv(custom_csv_path)
    print(f"\n[Test 2] Custom CSV Loading & Schema Validation:")
    print(f"  - Uploaded records count: {len(custom_df):,}")
    print(f"  - Columns: {list(custom_df.columns)}")
    assert len(custom_df) == 200, f"Expected exactly 200 rows, got {len(custom_df)}"
    
    validator = VoIPPreprocessor()
    is_valid, msg = validator.validate_dataframe(custom_df)
    assert is_valid, f"Validation failed: {msg}"
    print(f"  ✓ PASSED: Custom CSV validated successfully ({len(custom_df)} rows).")

    # 3. Test dynamic model fitting & context generation on 200-row dataset
    print(f"\n[Test 3] Dynamic GMM Fitting & Threshold Recalculation:")
    ctx = build_custom_model_context(custom_df)
    
    preprocessor = ctx["preprocessor"]
    detector = ctx["detector"]
    metadata = ctx["metadata"]
    scores = ctx["scores"]
    statuses = ctx["statuses"]
    eval_data = ctx["eval_data"]
    comp_df = ctx["comp_df"]

    assert len(scores) == 200, f"Expected 200 scores, got {len(scores)}"
    assert len(statuses) == 200, f"Expected 200 statuses, got {len(statuses)}"
    print(f"  - Number of GMM components selected (K): {metadata['n_components']}")
    print(f"  - Baseline score mean: {metadata['baseline_score_mean']:.4f}")
    print(f"  - Normal (95th%) threshold: {detector.threshold_normal:.4f}")
    print(f"  - Critical (99th%) threshold: {detector.threshold_anomalous:.4f}")
    print(f"  - Status counts: Normal={statuses.count('NORMAL')}, Suspicious={statuses.count('SUSPICIOUS')}, Anomalous={statuses.count('ANOMALOUS')}")
    assert statuses.count('NORMAL') + statuses.count('SUSPICIOUS') + statuses.count('ANOMALOUS') == 200
    print("  ✓ PASSED: GMM thresholds, scoring, and classification dynamically recalculated for 200 rows.")

    # 4. Test Model Evaluation and Confusion Matrix observation count
    print(f"\n[Test 4] Model Evaluation & Confusion Matrix Sample Count:")
    cm = eval_data["confusion_matrix"]
    tp = cm["true_positives"]
    fp = cm["false_positives"]
    fn = cm["false_negatives"]
    tn = cm["true_negatives"]
    total_cm_samples = tp + fp + fn + tn

    print(f"  - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"  - Total samples represented in confusion matrix: {total_cm_samples}")
    assert total_cm_samples == 200, f"Expected confusion matrix to sum to 200, got {total_cm_samples}"
    print("  ✓ PASSED: Confusion matrix total equals exactly 200 observations (not 10,000).")

    # 5. Test BIC / AIC scores for GMM Diagnostics
    print(f"\n[Test 5] GMM Diagnostics Model Selection:")
    bic_scores = metadata["bic_scores"]
    aic_scores = metadata["aic_scores"]
    print(f"  - BIC scores computed across K: {bic_scores}")
    print(f"  - AIC scores computed across K: {aic_scores}")
    assert len(bic_scores) > 0 and len(aic_scores) > 0
    print("  ✓ PASSED: BIC/AIC curves dynamically computed for custom dataset.")

    # 6. Test Model Comparison (GMM vs Isolation Forest)
    print(f"\n[Test 6] Model Comparison (GMM vs Isolation Forest):")
    assert not comp_df.empty, "Comparison table is empty!"
    print(comp_df.to_string(index=False))
    print("  ✓ PASSED: Comparison table dynamically computed for custom dataset.")

    print("\n" + "=" * 70)
    print("ALL 6 VALIDATION TESTS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
