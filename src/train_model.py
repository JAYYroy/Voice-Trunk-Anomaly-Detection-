"""
Model Training and Model Selection Pipeline.

Executes:
1. Data loading and validation
2. Preprocessing & StandardScaler fitting
3. Hyperparameter selection across K in {1, 2, 3, 4, 5} via BIC and AIC
4. Optimal GMM model training
5. Threshold calibration (95th & 99th percentiles of baseline distribution)
6. Secondary baseline fitting (Isolation Forest)
7. Full quantitative evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC)
8. Artifact serialization (gmm_model.pkl, scaler.pkl, isolation_forest.pkl, model_metadata.json)
9. Output generation (outputs/plots/ and outputs/results/)
"""

import os
import sys
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import joblib

from src.preprocessing import VoIPPreprocessor, DEFAULT_FEATURES
from src.anomaly_detector import GMMAnomalyDetector
from src.evaluation import evaluate_anomaly_detector, compare_models
from src.visualization import save_all_eda_plots, save_model_diagnostics_plots


def train_pipeline(
    data_path: str,
    output_dir: str = "models",
    plots_dir: str = "outputs/plots",
    results_dir: str = "outputs/results",
    random_state: int = 42
):
    print("=" * 70)
    print("      VOICE TRUNK ANOMALY SCORING - GMM TRAINING PIPELINE      ")
    print("=" * 70)

    # 1. Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load user-provided data
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Provide an explicit VoIP telemetry CSV; "
            "the training pipeline no longer generates a default dataset."
        )

    print(f"[*] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"[+] Loaded {len(df):,} observations.")
    if "is_anomaly" in df.columns:
        n_anom = int(df["is_anomaly"].sum())
        print(f"[+] Ground Truth Anomaly Count: {n_anom} ({n_anom / len(df) * 100:.2f}%)")

    # 3. Generate EDA Plots
    print("[*] Generating Exploratory Data Analysis (EDA) visualizations...")
    eda_plots = save_all_eda_plots(df, output_dir=plots_dir)
    print(f"[+] Generated {len(eda_plots)} EDA plots in: {plots_dir}")

    # 4. Preprocessing & Scaling
    print("[*] Fitting VoIPPreprocessor (StandardScaler)...")
    preprocessor = VoIPPreprocessor(feature_names=DEFAULT_FEATURES)
    preprocessor.fit(df)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    preprocessor.save(scaler_path)
    print(f"[+] Preprocessor saved to: {scaler_path}")

    # Transform all data
    X_all = preprocessor.transform(df)

    # In unsupervised learning, normal baseline traffic represents standard operational behavior.
    if "is_anomaly" in df.columns:
        normal_mask = (df["is_anomaly"] == 0).values
        X_train_normal = X_all[normal_mask]
    else:
        X_train_normal = X_all

    print(f"[+] Training baseline feature matrix shape: {X_train_normal.shape}")

    # 5. Model Selection across K in [1, 2, 3, 4, 5]
    print("\n[*] Evaluating Gaussian Mixture Model configurations across K in {1, 2, 3, 4, 5}...")
    k_range = [1, 2, 3, 4, 5]
    bic_scores = {}
    aic_scores = {}

    best_k = 1
    lowest_bic = float("inf")

    for k in k_range:
        gmm_candidate = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            max_iter=200,
            n_init=3
        )
        gmm_candidate.fit(X_train_normal)
        bic = float(gmm_candidate.bic(X_train_normal))
        aic = float(gmm_candidate.aic(X_train_normal))
        bic_scores[k] = bic
        aic_scores[k] = aic
        print(f"    - K = {k:2d} | BIC = {bic:10.2f} | AIC = {aic:10.2f}")

        if bic < lowest_bic:
            lowest_bic = bic
            best_k = k

    print(f"[+] Optimal number of components selected: K = {best_k} (Lowest BIC = {lowest_bic:.2f})")

    # 6. Train Final GMM Anomaly Detector
    print(f"\n[*] Training final GMM detector with K = {best_k} components...")
    detector = GMMAnomalyDetector(
        n_components=best_k,
        covariance_type="full",
        random_state=random_state,
        normal_percentile=95.0,
        anomalous_percentile=99.0
    )
    detector.bic_history = bic_scores
    detector.aic_history = aic_scores
    detector.fit(X_train_normal)

    # Save detector model
    model_path = os.path.join(output_dir, "gmm_model.pkl")
    detector.save(model_path)
    print(f"[+] Trained GMM model saved to: {model_path}")
    print(f"    - Baseline Score Mean: {detector.baseline_score_mean:.4f}")
    print(f"    - Baseline Score Std:  {detector.baseline_score_std:.4f}")
    print(f"    - 95th Percentile Threshold (Normal/Suspicious Cutoff): {detector.threshold_normal:.4f}")
    print(f"    - 99th Percentile Threshold (Suspicious/Anomalous Cutoff): {detector.threshold_anomalous:.4f}")

    # 7. Train Secondary Baseline (Isolation Forest)
    print("\n[*] Training secondary baseline: Isolation Forest for comparative study...")
    iforest = IsolationForest(
        n_estimators=150,
        contamination=0.08,
        random_state=random_state,
        n_jobs=-1
    )
    iforest.fit(X_train_normal)
    iforest_path = os.path.join(output_dir, "isolation_forest.pkl")
    joblib.dump(iforest, iforest_path)
    print(f"[+] Isolation Forest model saved to: {iforest_path}")

    # 8. Compute Full Dataset Anomaly Scores
    gmm_all_scores = detector.compute_anomaly_scores(X_all)
    # IsolationForest: score_samples() returns opposite of anomaly (higher = normal); convert to anomaly score
    iforest_raw_scores = iforest.score_samples(X_all)
    iforest_all_scores = -iforest_raw_scores
    iforest_threshold = float(np.percentile(iforest_all_scores[normal_mask], 99.0)) if "is_anomaly" in df.columns else float(np.percentile(iforest_all_scores, 99.0))

    # 9. Perform Comprehensive Evaluation
    eval_results = {}
    if "is_anomaly" in df.columns:
        y_true = df["is_anomaly"].values
        print("\n[*] Computing quantitative evaluation metrics against ground-truth labels...")
        eval_results = evaluate_anomaly_detector(
            y_true=y_true,
            anomaly_scores=gmm_all_scores,
            threshold=detector.threshold_anomalous
        )

        m = eval_results["metrics"]
        cm = eval_results["confusion_matrix"]
        print("-" * 50)
        print("          GMM EVALUATION METRICS SUMMARY          ")
        print("-" * 50)
        print(f"  Accuracy:    {m['accuracy'] * 100:.2f}%")
        print(f"  Precision:   {m['precision'] * 100:.2f}%")
        print(f"  Recall:      {m['recall'] * 100:.2f}%")
        print(f"  F1-Score:    {m['f1_score']:.4f}")
        print(f"  Specificity: {m['specificity'] * 100:.2f}%")
        print(f"  ROC-AUC:     {m['roc_auc']:.4f}")
        print(f"  PR-AUC:      {m['pr_auc']:.4f}")
        print("-" * 50)
        print(f"  Confusion Matrix: TP={cm['true_positives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}, TN={cm['true_negatives']}")
        print("-" * 50)

        # Model comparison table
        comparison_df = compare_models(
            y_true=y_true,
            gmm_scores=gmm_all_scores,
            gmm_threshold=detector.threshold_anomalous,
            iforest_scores=iforest_all_scores,
            iforest_threshold=iforest_threshold
        )
        comp_csv_path = os.path.join(results_dir, "model_comparison.csv")
        comparison_df.to_csv(comp_csv_path, index=False)
        print(f"[+] Model comparison saved to: {comp_csv_path}")

        # Diagnostic Plots
        iforest_roc_eval = evaluate_anomaly_detector(y_true, iforest_all_scores, iforest_threshold)
        save_model_diagnostics_plots(
            bic_scores=bic_scores,
            aic_scores=aic_scores,
            anomaly_scores=gmm_all_scores,
            th_normal=detector.threshold_normal,
            th_anomalous=detector.threshold_anomalous,
            cm=np.array(cm["matrix"]),
            gmm_roc={"fpr": eval_results["curves"]["fpr"], "tpr": eval_results["curves"]["tpr"], "auc": m["roc_auc"]},
            iforest_roc={"fpr": iforest_roc_eval["curves"]["fpr"], "tpr": iforest_roc_eval["curves"]["tpr"], "auc": iforest_roc_eval["metrics"]["roc_auc"]},
            output_dir=plots_dir
        )
        print(f"[+] Saved diagnostic plots (BIC/AIC, Score Distribution, Confusion Matrix, ROC) to: {plots_dir}")

    # 10. Save Results & Metadata JSON
    metadata = {
        "model_type": "GaussianMixture",
        "n_components": int(best_k),
        "covariance_type": "full",
        "random_state": random_state,
        "features": DEFAULT_FEATURES,
        "normal_percentile": 95.0,
        "anomalous_percentile": 99.0,
        "threshold_normal": round(float(detector.threshold_normal), 4),
        "threshold_anomalous": round(float(detector.threshold_anomalous), 4),
        "baseline_score_mean": round(float(detector.baseline_score_mean), 4),
        "baseline_score_std": round(float(detector.baseline_score_std), 4),
        "bic_scores": {str(k): round(v, 2) for k, v in bic_scores.items()},
        "aic_scores": {str(k): round(v, 2) for k, v in aic_scores.items()},
        "component_weights": [round(float(w), 4) for w in detector.model.weights_],
        "training_samples_baseline": int(X_train_normal.shape[0]),
        "total_dataset_samples": int(len(df)),
        "evaluation_metrics": eval_results.get("metrics", {}),
        "confusion_matrix": eval_results.get("confusion_matrix", {})
    }

    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Model metadata successfully saved to: {meta_path}")

    eval_results_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"[+] Evaluation results saved to: {eval_results_path}")

    print("\n[SUCCESS] TRAINING PIPELINE COMPLETED SUCCESSFULLY!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Voice Trunk Anomaly Scoring GMM Pipeline.")
    parser.add_argument("--data", type=str, required=True, help="Path to uploaded/user-provided VoIP telemetry CSV")
    parser.add_argument("--models", type=str, default="models", help="Model artifact directory")
    parser.add_argument("--plots", type=str, default="outputs/plots", help="Plots output directory")
    parser.add_argument("--results", type=str, default="outputs/results", help="Results output directory")
    args = parser.parse_args()

    train_pipeline(
        data_path=args.data,
        output_dir=args.models,
        plots_dir=args.plots,
        results_dir=args.results
    )
