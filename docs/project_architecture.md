# Project Architecture - Voice Trunk Anomaly Scoring

**Subject:** Pattern Recognition and Anomaly Detection (PRAD)  
**System:** Real-Time VoIP Trunk Quality of Service (QoS) Anomaly Detection Platform  
**Primary Algorithm:** Gaussian Mixture Model (GMM) with Negative Log-Likelihood Scoring  

---

## 1. System Overview

The **Voice Trunk Anomaly Scoring** system is an end-to-end machine learning platform designed to monitor VoIP (Voice over IP) and SIP trunk communication infrastructure in real time. In digital telecommunications, voice packets require low latency ($\le 150\text{ ms}$), minimal jitter ($\le 20\text{ ms}$), and low packet loss ($\le 1\%$) to maintain acceptable Mean Opinion Scores (MOS) under ITU-T G.114 standards.

Rather than relying on static, fragile heuristic rules (e.g. `if latency > 100ms`), this architecture implements an unsupervised **Gaussian Mixture Model (GMM)** to model the continuous multi-dimensional probability distribution of normal network telemetry. Anomalies (bufferbloat, fiber cuts, micro-burst packet loss, jitter surges) are detected as deviations from this learned density distribution.

```mermaid
graph TD
    A[VoIP Telemetry Source<br/>data/voice_trunk_data.csv / Live Generator] --> B[VoIPPreprocessor<br/>src/preprocessing.py]
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

## 2. Directory & Module Structure

```
voice-trunk-anomaly-scoring/
│
├── app.py                          # Streamlit Multi-Page Interactive Dashboard
├── requirements.txt                # Pinned dependencies
├── README.md                       # Academic project overview and run guide
│
├── data/
│   └── voice_trunk_data.csv        # 10,000-sample VoIP network telemetry dataset
│
├── models/
│   ├── gmm_model.pkl               # Serialized GMMAnomalyDetector instance
│   ├── scaler.pkl                  # Serialized VoIPPreprocessor (StandardScaler)
│   ├── isolation_forest.pkl        # Serialized Isolation Forest comparison model
│   └── model_metadata.json         # Training metrics, weights, and thresholds
│
├── src/
│   ├── __init__.py                 # Package marker
│   ├── data_generator.py           # VoIP synthetic telemetry engine (ITU-T compliant)
│   ├── preprocessing.py            # Data validation, cleaning, and scaling
│   ├── anomaly_detector.py         # GMM wrapper for scoring and tri-state classification
│   ├── train_model.py              # BIC/AIC model selection and training orchestrator
│   ├── evaluation.py               # Classification metrics, confusion matrix, ROC-AUC
│   └── visualization.py            # Matplotlib & Seaborn publication plot generator
│
├── outputs/
│   ├── plots/                      # 13 high-resolution diagnostic & EDA PNG plots
│   └── results/                    # evaluation_metrics.json & model_comparison.csv
│
├── docs/
│   ├── project_architecture.md     # This architecture specification document
│   ├── methodology.md              # Mathematical derivation of GMM & EM algorithm
│   ├── results.md                  # Real numerical experimental results and analysis
│   ├── ppt_content.md              # 20-slide structured presentation deck
│   └── report_outline.md           # Formal academic submission report outline
│
└── screenshots/
    ├── 01_dashboard_overview.png   # Main KPI cards and network health donut
    ├── 02_live_monitoring.png      # Interactive real-time fault injection simulation
    ├── 03_anomaly_analysis.png     # Threshold calibration sliders and score histogram
    ├── 04_gmm_diagnostics.png      # BIC/AIC optimization curve and parameter weights
    ├── 05_model_evaluation.png     # Confusion matrix and GMM vs Isolation Forest table
    └── README.md                   # Image gallery index and descriptions
```

---

## 3. Data Dictionary & Feature Engineering

The feature vector $\mathbf{x} \in \mathbb{R}^6$ captures essential aspects of transmission impairment:

| Feature Name | Unit | Type | Physical Range | Normal Operational Range | Anomaly Range | Theoretical Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `packet_loss_pct` | % | Float | $[0.0, 100.0]$ | $0.0 - 2.0\%$ | $5.0 - 35.0\%$ | Key indicator of dropped audio frames leading to robotic voice / silence gaps. |
| `latency_ms` | ms | Float | $[0.0, \infty)$ | $15.0 - 75.0\text{ ms}$ | $150.0 - 500.0\text{ ms}$ | One-way propagation delay; delays $>150\text{ ms}$ cause conversational talk-over. |
| `jitter_ms` | ms | Float | $[0.0, \infty)$ | $1.0 - 12.0\text{ ms}$ | $30.0 - 100.0\text{ ms}$ | Statistical variance in packet arrival time; causes de-jitter buffer underruns. |
| `packet_drop_count` | pkts | Int | $[0, \infty)$ | $0 - 1\text{ pkt/sec}$ | $3 - 18\text{ pkts/sec}$ | Discrete count of dropped frames per second window. |
| `throughput_kbps` | kbps | Float | $[0.0, \infty)$ | $75.0 - 95.0\text{ kbps}$ | $30.0 - 65.0\text{ kbps}$ | Bandwidth utilization for standard G.711 / Opus voice stream. |
| `delay_variation_ms`| ms | Float | $[0.0, \infty)$ | $0.5 - 15.0\text{ ms}$ | $35.0 - 120.0\text{ ms}$ | High-frequency packet delay variation across consecutive voice bursts. |

---

## 4. End-to-End Execution Flow

1. **Generation:** `src/data_generator.py` simulates 10,000 chronological observations with an 8% anomaly ratio covering 5 realistic telecom failure modes.
2. **Preprocessing:** `src/preprocessing.py` standardizes the feature space using `StandardScaler` to prevent high-magnitude features (latency) from dominating low-magnitude features (packet loss).
3. **Model Selection:** `src/train_model.py` sweeps $K \in \{1, 2, 3, 4, 5\}$ using BIC and AIC, selecting $K=5$ components.
4. **Scoring:** Log-likelihood is computed via `GaussianMixture.score_samples()`, inverted to yield $S(\mathbf{x}) = -\ln p(\mathbf{x})$.
5. **Threshold Calibration:** The 95th percentile ($\tau_{95} = -3.83$) defines the suspicious cutoff, and the 99th percentile ($\tau_{99} = 0.09$) defines the critical anomaly cutoff.
6. **Inference & UI:** `app.py` loads the pre-trained artifacts to provide interactive real-time simulation, manual fault injection, dynamic threshold tuning, and diagnostic explanations.
