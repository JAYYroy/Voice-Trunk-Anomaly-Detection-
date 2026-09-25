# UI Screenshots Gallery

This directory contains real, full-resolution screenshots captured directly from the live running **Voice Trunk Anomaly Scoring** Streamlit application. These images are organized for direct inclusion in the academic report, presentation slides (PPT), and project documentation.

## Available Screenshots

1. **`01_dashboard_overview.png`**
   - **Page:** Dashboard Overview
   - **Features:** 6 KPI metric tiles (Total Records, Critical Anomalies, Anomaly Rate, Avg Latency, Avg Loss, Avg Jitter), donut distribution of Network Health, and rolling 3-channel time-series charts.

2. **`02_live_monitoring.png`**
   - **Page:** Live Voice Trunk Monitor
   - **Features:** Dynamic frame inspection metrics, tri-state alert badge (`ANOMALOUS`), live injection buttons (Normal, Latency Spike, Loss Burst, Combined Failure), and dual streaming charts showing live score response.

3. **`03_anomaly_analysis.png`**
   - **Page:** Anomaly Analysis & Threshold Calibration
   - **Features:** Interactive percentile threshold sliders (95th & 99th cutoffs), multi-class log-likelihood score histogram, and 2D feature scatter projections.

4. **`04_gmm_diagnostics.png`**
   - **Page:** GMM Model Diagnostics
   - **Features:** Hyperparameter summary ($K=5$, Covariance=`full`), BIC/AIC minimization elbow curve, GMM component mixture weights, and viva mathematical formulation breakdown.

5. **`05_model_evaluation.png`**
   - **Page:** Model Evaluation & Benchmarks
   - **Features:** Quantitative evaluation metrics (99.08% Accuracy, 100% Recall, 0.9459 F1), confusion matrix heatmap, and side-by-side benchmark comparison against Isolation Forest.
