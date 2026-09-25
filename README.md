# Voice Trunk Anomaly Scoring

**Subject:** Pattern Recognition and Anomaly Detection (PRAD)  
**Project Title:** Voice Trunk Anomaly Scoring  
**Sub-title:** GMM-Based Anomaly Detection for Real-Time Audio Communication Infrastructure  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![PRAD Academic Project](https://img.shields.io/badge/PRAD-Academic_Project-10B981.svg)]()

---

## 1. Problem Statement

> *"Implement a Gaussian Mixture Model to detect irregular packet drops or latency spikes in real-time audio communication infrastructure."*

Real-time audio communication infrastructure (VoIP trunks, SIP session border controllers, WebRTC media gateways) is acutely sensitive to Quality of Service (QoS) degradation. Unlike standard TCP-based web browsing which can buffer or retransmit lost packets, real-time audio uses UDP/RTP where packet loss results in audible dropouts, latency causes conversational collision, and jitter causes buffer underruns.

Traditional monitoring relies on simplistic, static thresholds (e.g. `latency > 100ms`). Such approaches generate false alarms during normal load variations and fail to capture multi-metric compound failures (e.g. moderate jitter coupled with moderate loss). This project implements an **unsupervised Gaussian Mixture Model (GMM)** to probabilistically model healthy multi-modal traffic and detect anomalies via negative log-likelihood density scoring.

---

## 2. Project Objectives

- **Unsupervised Telemetry Modeling:** Learn the multi-dimensional probability distribution of normal voice trunk traffic without relying on anomaly labels during training.
- **Continuous Probabilistic Scoring:** Quantify network health by calculating the exact log-likelihood under the GMM and mapping it to an intuitive anomaly score: $S(\mathbf{x}) = -\ln p(\mathbf{x})$.
- **Tri-State Classification:** Systematically classify incoming telemetry frames into `NORMAL`, `SUSPICIOUS`, and `ANOMALOUS` using empirical percentile thresholds (95th and 99th percentiles).
- **Comparative Benchmarking:** Compare GMM performance against an alternative baseline (Isolation Forest).
- **Interactive Real-Time Monitoring:** Provide a Streamlit dashboard with live streaming telemetry, interactive fault injection, and threshold calibration controls.

---

## 3. Technology Stack

- **Core Machine Learning:** Python, NumPy, Pandas, Scikit-learn (`sklearn.mixture.GaussianMixture`), Joblib, SciPy.
- **Visualization:** Matplotlib, Seaborn, Plotly.
- **Frontend Dashboard:** Streamlit (clean academic dashboard design).
- **Environment:** 100% offline reproducible execution with zero external cloud dependencies.

---

## 4. System Architecture

```mermaid
graph TD
    A[Uploaded VoIP Telemetry CSV<br/>User-provided data only] --> B[VoIPPreprocessor<br/>src/preprocessing.py]
    B -->|Cleaning, Imputation, Bound Enforcement| C[Feature Matrix X<br/>6 Dimensional QoS Vector]
    C -->|StandardScaler Normalization| D[Standardized Features X_scaled]
    
    subgraph Offline Model Selection & Training
        D --> E[Hyperparameter Optimization<br/>K in {1, 2, 3, 4, 5}]
        E -->|Calculate BIC & AIC| F[Component Selection: K=5<br/>Lowest BIC = -138,744.39]
        F --> G[Fitted GMM Artifact<br/>models/gmm_model.pkl]
        F --> H[Secondary Baseline<br/>models/isolation_forest.pkl]
    end
    
    subgraph Online Anomaly Scoring Pipeline
        D --> I[Likelihood Density Evaluator<br/>GaussianMixture.score_samples]
        G -.-> I
        I -->|ln p(x)| J[Negative Log-Likelihood<br/>Anomaly Score S(x) = -ln p(x)]
        J --> K[Dual-Threshold Calibration<br/>tau_95: Suspicious | tau_99: Critical]
        K --> L{Decision Classifier}
        L -->|S(x) <= tau_95| M[NORMAL Quality]
        L -->|tau_95 < S(x) <= tau_99| N[SUSPICIOUS Condition]
        L -->|S(x) > tau_99| O[ANOMALOUS Incident Alert]
    end
    
    subgraph User Presentation Layer
        M & N & O --> P[Interactive Streamlit Dashboard<br/>app.py]
        P --> Q[Live Telemetry Streaming]
        P --> R[Dynamic Threshold Tuning]
        P --> S[Model Diagnostics & Viva Guide]
    end
```

---

## 5. Dataset Specification

The Streamlit application starts with no active telemetry dataset. The user must upload a VoIP telemetry CSV with the required network columns before GMM analysis, diagnostics, evaluation, and charts are calculated:

- **Active Samples:** Determined by the uploaded CSV. No 10,000-row default dataset is loaded on startup.
- **Baseline Healthy Telemetry (91.95%):**
  - Packet Loss: $0.0 - 1.95\%$ (mean: $0.41\%$)
  - One-Way Latency: $12.0 - 78.0\text{ ms}$ (multi-modal local vs interstate routes)
  - Jitter: $0.8 - 14.5\text{ ms}$
  - Packets Sent: ~50 packets/sec (20ms G.711 / Opus voice frames)
  - Throughput: ~84 kbps nominal voice stream
- **Synthetic Injected Anomalies (8.05%):**
  - High Packet Loss ($6 - 24\%$)
  - High Latency ($150 - 520\text{ ms}$)
  - High Jitter ($32 - 92\text{ ms}$)
  - Combined Degradation (simultaneous latency, loss, and jitter degradation)
  - Sudden Packet Drop Bursts ($18 - 38\%$)

---

## 6. Preprocessing Pipeline

1. **Validation:** Checks column presence, data types, and non-empty rows.
2. **Missing Value Imputation:** Imputes missing values using feature medians.
3. **Physical Constraints:** Enforces non-negative network values and bounds packet loss to $[0, 100]\%$.
4. **Standardization (`StandardScaler`):** Normalizes all 6 numerical features to zero mean and unit variance:
   $$z = \frac{x - \mu}{\sigma}$$
   The fitted scaler is saved to `models/scaler.pkl` and applied identically during online inference.

---

## 7. Gaussian Mixture Model & Anomaly Scoring

### Mathematical Formulation
The probability density of feature vector $\mathbf{x} \in \mathbb{R}^6$ is modeled as a mixture of $K$ multivariate Gaussian components:
$$p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

Where:
- $\pi_k$ is the mixture weight of component $k$ ($\sum \pi_k = 1$).
- $\boldsymbol{\mu}_k$ is the mean vector of component $k$.
- $\boldsymbol{\Sigma}_k$ is the full covariance matrix modeling joint feature interactions.

### Model Selection via BIC
We evaluated $K \in \{1, 2, 3, 4, 5\}$ using the Bayesian Information Criterion (BIC):
$$\text{BIC} = -2 \ln \hat{L} + p \ln(N)$$
- $K=1$: $\text{BIC} = -38,230.71$
- $K=2$: $\text{BIC} = -131,428.77$
- $K=3$: $\text{BIC} = -131,356.68$
- $K=4$: $\text{BIC} = -135,424.85$
- **$K=5$ (Selected): $\text{BIC} = -138,744.39$** (Lowest BIC)

### Anomaly Score Definition
The anomaly score is the negative log-likelihood computed via `GaussianMixture.score_samples()`:
$$S(\mathbf{x}) = -\ln p(\mathbf{x})$$
- Normal frames fall in high-density regions $\implies$ low score (baseline mean: $-7.61$).
- Anomalies fall far in distribution tails $\implies$ high score.

### Decision Boundaries
Calibrated from percentiles of normal baseline training scores:
- **`NORMAL`:** $S(\mathbf{x}) \le -3.8321$ (Below 95th percentile)
- **`SUSPICIOUS`:** $-3.8321 < S(\mathbf{x}) \le +0.0856$ (Between 95th and 99th percentile)
- **`ANOMALOUS`:** $S(\mathbf{x}) > +0.0856$ (Above 99th percentile)

---

## 8. Experimental Results & Benchmarks

Evaluation at the critical anomaly threshold $\tau_{99} = 0.0856$ against ground-truth labels yielded:

| Metric | GMM Anomaly Detector (Primary) | Isolation Forest (Baseline) | Performance Difference |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **99.08%** | 96.48% | **+2.60%** |
| **Precision** | **89.74%** | 85.56% | **+4.18%** |
| **Recall (Sensitivity)** | **100.00%** | 67.70% | **+32.30%** (Zero missed faults) |
| **F1-Score** | **0.9459** | 0.7559 | **+0.1900** |
| **ROC-AUC** | **1.0000** | 0.9908 | Perfect discrimination |
| **PR-AUC** | **1.0000** | 0.8864 | **+0.1136** |
| **True Positives (TP)** | **805 / 805** | 545 / 805 | GMM caught all 805 anomalies |
| **False Negatives (FN)** | **0** | **260** | Isolation Forest missed 260 anomalies |

---

## 9. Visual Gallery & Screenshots

All high-resolution demonstration screenshots are available in `screenshots/`:

| Screen | Description |
| :--- | :--- |
| `01_dashboard_overview.png` | Executive KPI cards, network health donut, and 3-channel rolling time series. |
| `02_live_monitoring.png` | Live packet inspection console with interactive fault injection (`Normal`, `Latency Spike`, `Loss Burst`, `Combined Failure`). |
| `03_anomaly_analysis.png` | Dual percentile threshold tuning sliders and continuous log-likelihood score histogram. |
| `04_gmm_diagnostics.png` | Hyperparameter breakdown, BIC/AIC elbow minimization curve, and viva mathematical formulations. |
| `05_model_evaluation.png` | Confusion matrix heatmap and side-by-side benchmark comparison against Isolation Forest. |

---

## 10. Project Directory Structure

```
voice-trunk-anomaly-scoring/
│
├── app.py                          # Streamlit Multi-Page Interactive Dashboard
├── requirements.txt                # Pinned dependencies
├── README.md                       # Project overview and run guide
│
├── data/
│   └── test_custom_200.csv         # Example user-upload CSV for validation
│
├── models/
│   ├── gmm_model.pkl               # Serialized GMMAnomalyDetector artifact
│   ├── scaler.pkl                  # Serialized VoIPPreprocessor (StandardScaler)
│   ├── isolation_forest.pkl        # Serialized Isolation Forest comparison model
│   └── model_metadata.json         # Training metrics, weights, and thresholds
│
├── src/
│   ├── __init__.py                 # Package init
│   ├── data_generator.py           # Standalone legacy generator; not used by the Streamlit app
│   ├── preprocessing.py            # Data validation, cleaning, and scaling
│   ├── anomaly_detector.py         # GMM scoring and classification engine
│   ├── train_model.py              # BIC/AIC model selection and training orchestrator
│   ├── evaluation.py               # Classification metrics, confusion matrix, ROC-AUC
│   └── visualization.py            # Matplotlib & Seaborn publication plot generator
│
├── outputs/
│   ├── plots/                      # 13 high-resolution diagnostic & EDA PNG plots
│   └── results/                    # evaluation_metrics.json & model_comparison.csv
│
├── docs/
│   ├── project_architecture.md     # In-depth technical architecture breakdown
│   ├── methodology.md              # Mathematical derivation of GMM & EM algorithm
│   ├── results.md                  # Real numerical experimental results and analysis
│   ├── ppt_content.md              # 20-slide structured presentation deck
│   └── report_outline.md           # Formal academic submission report outline
│
└── screenshots/
    ├── 01_dashboard_overview.png
    ├── 02_live_monitoring.png
    ├── 03_anomaly_analysis.png
    ├── 04_gmm_diagnostics.png
    ├── 05_model_evaluation.png
    └── README.md
```

---

## 11. Installation & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train Model from Scratch
```bash
python src/train_model.py
```
*Requires an explicit CSV path, then executes EDA plot rendering, BIC/AIC hyperparameter sweep, GMM and Isolation Forest fitting, and artifact export.*

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
*Access the application in your browser at `http://localhost:8501`.*

---

## 12. Academic Integrity Statement

1. **Simulation Clarification:** This project represents an academic simulation of VoIP/SIP voice trunk telemetry designed according to ITU-T G.114/G.1010 standards. No proprietary telecom carrier infrastructure was tapped or claimed to be used.
2. **Authenticity of Results:** All numerical scores, confusion matrices, BIC values, and evaluation metrics reported in this repository and documentation were generated directly from the execution of the code.
