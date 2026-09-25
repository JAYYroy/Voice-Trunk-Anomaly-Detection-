# Voice Trunk Anomaly Scoring — Master Project Documentation

**Project Title:** Voice Trunk Anomaly Scoring: GMM-Based Anomaly Detection for Real-Time Audio Communication Infrastructure  
**Subject / Domain:** Pattern Recognition and Anomaly Detection (PRAD) / VoIP Telecommunications QoS  
**Primary Algorithm:** Unsupervised Gaussian Mixture Model (GMM) with Negative Log-Likelihood Density Scoring  
**Secondary Baseline:** Isolation Forest  
**Frontend Framework:** Streamlit (Python-based interactive reactive web UI)  

---

## 1. Executive Summary & Project Overview

### 1.1 What is the Project About?
Real-time conversational audio protocols (VoIP, SIP Trunking, WebRTC) rely on **UDP/RTP** transmission. Unlike standard web browsing (HTTP/TCP), which can seamlessly buffer or retransmit lost packets, real-time voice packets cannot be retransmitted without causing audible interruptions, broken words, or call dropouts.

Under the **ITU-T G.114 and G.1010** international standards, high-quality interactive voice requires:
- **One-Way Latency:** $\le 150\text{ ms}$
- **Jitter (delay variation):** $\le 20\text{ ms}$
- **Packet Loss:** $\le 1\%$

When network congestion, route flapping, bufferbloat, or hardware degradation occurs, these Quality of Service (QoS) parameters deviate from normal baselines.

### 1.2 The Problem with Traditional Monitoring
Traditional Network Operations Centers (NOCs) rely on **static scalar thresholds** (e.g., alert if `latency > 100ms`). This approach fails in two major ways:
1. **High False Positive Rate:** Temporary surges during peak business hours trigger false alarms (alarm fatigue).
2. **High False Negative Rate on Compound Degradations:** If latency is $85\text{ ms}$ (below $100\text{ ms}$) and packet loss is $1.8\%$ (below $2\%$), neither threshold trips, yet the combined impairment destroys audio intelligibility.
3. **Absence of Labeled Data:** In real networks, true anomaly labels are rarely available, rendering supervised machine learning unfeasible.

### 1.3 The Solution: Unsupervised Gaussian Mixture Modeling (GMM)
This project implements an **unsupervised statistical learning pipeline** using **Gaussian Mixture Models (GMM)**. 
- It fits a multi-component probability distribution over normal multi-dimensional network telemetry.
- Each incoming telemetry sample is evaluated against the learned distribution to compute its **exact log-likelihood density** $\ln p(\mathbf{x})$.
- The **Anomaly Score** is formulated as the negative log-likelihood:
  $$S(\mathbf{x}) = -\ln p(\mathbf{x})$$
- Observations falling in low-density tail regions receive high anomaly scores.
- Percentile-based thresholds ($\tau_{95}$ and $\tau_{99}$) systematically categorize network health into **`NORMAL`**, **`SUSPICIOUS`**, and **`ANOMALOUS`**.

---

## 2. Technology Stack & Tools

| Layer | Component / Tool | Version / Library | Purpose in this Project |
| :--- | :--- | :--- | :--- |
| **Programming Language** | Python | 3.10+ / 3.11 | Core programming language for ML, data processing, and web server. |
| **Machine Learning** | Scikit-Learn | `sklearn.mixture.GaussianMixture`<br>`sklearn.preprocessing.StandardScaler`<br>`sklearn.ensemble.IsolationForest` | Density estimation (GMM), expectation-maximization, feature standardization, and baseline model comparison. |
| **Numerical Computation** | NumPy & SciPy | `numpy>=1.24.0`<br>`scipy>=1.11.0` | Vectorized linear algebra, multi-dimensional array operations, and percentile math. |
| **Data Manipulation** | Pandas | `pandas>=2.0.0` | CSV ingestion, data schema validation, cleaning, missing-value imputation, and tabular filtering. |
| **Model Persistence** | Joblib | `joblib>=1.3.0` | Serialization and deserialization of fitted preprocessors and trained model artifacts (`.pkl`). |
| **Frontend UI** | Streamlit | `streamlit>=1.28.0` | Reactive web application, file uploader, session state management, KPI cards, and dynamic UI rendering. |
| **Interactive Visualizations** | Plotly | `plotly>=5.15.0`<br>(`express`, `graph_objects`) | Interactive time-series charts, dual-threshold scatter plots, and network health distribution donut charts. |
| **Static Visualizations** | Matplotlib & Seaborn | `matplotlib>=3.7.0`<br>`seaborn>=0.12.0` | Diagnostic EDA plots, BIC/AIC elbow curve generation, and confusion matrix heatmaps. |
| **Domain Standards** | ITU-T Recommendations | G.114, G.1010, RFC 3550 | Telecommunications industry QoS thresholds and telephony transmission performance baselines. |

---

## 3. End-to-End System Architecture

```mermaid
flowchart TD
    A["CSV Upload (User Interface)"] --> B["Data Validation Engine"]
    B -->|Check Required 7 Columns| C{"Valid Schema?"}
    C -->|No| D["Display Column Error Alert<br/>Halt Execution"]
    C -->|Yes| E["Data Preprocessing (VoIPPreprocessor)"]
    
    subgraph Data Pipeline
        E -->|Median Imputation & Bounds Enforcement| F["Cleaned Data Matrix"]
        F -->|StandardScaler Transformation| G["Normalized Feature Matrix X_scaled"]
    end

    subgraph Statistical Modeling & Scoring
        G --> H["Gaussian Mixture Model (K Components)"]
        H -->|Expectation-Maximization| I["Learned Joint Density p(x)"]
        I -->|Log-Likelihood score_samples| J["ln p(x)"]
        J -->|Negation S(x) = -ln p(x)| K["Continuous Anomaly Score"]
    end

    subgraph Decision Classification
        K --> L["Dual-Threshold Calibration<br/>tau_95: 95th % | tau_99: 99th %"]
        L --> M{"Tri-State Evaluator"}
        M -->|S(x) <= tau_95| N["NORMAL (Green)"]
        M -->|tau_95 < S(x) <= tau_99| O["SUSPICIOUS (Amber)"]
        M -->|S(x) > tau_99| P["ANOMALOUS (Red)"]
    end

    subgraph Results & Root Cause Presentation
        N & O & P --> Q["Executive KPI Cards"]
        N & O & P --> R["Interactive Timeline & Distribution Charts"]
        N & O & P --> S["Records Table with Anomalies Highlighted"]
        N & O & P --> T["Root-Cause Diagnostics<br/>(High Latency / Loss / Jitter / Drops)"]
        U["Reset / Clear Dataset Button"] -.->|Clears All State| A
    end
```

---

## 4. Frontend & Backend Breakdown

### 4.1 Frontend Architecture (`app.py`)
- **Framework:** Streamlit web application.
- **Styling:** Custom CSS injected via `st.markdown`, providing a modern dark slate theme, typography (`Inter`), styled metric cards, status badges, and responsive containers.
- **State Management:** Uses `st.session_state` to store:
  - `uploaded_df`: Raw user uploaded data.
  - `processed_df`: Cleaned dataset with computed anomaly scores, log-likelihood, and tri-state classifications.
  - `model_context`: Fitted preprocessor, GMM model instance, thresholds, and statistical bounds.
- **Workflow & Views:**
  1. **Home / Initial State:** Starts with **"No dataset loaded"**. No dummy stats, no hardcoded charts. Shows clear instructions and an active drag-and-drop CSV uploader.
  2. **Data & Validation View:** Displays total row count, preview of the first 5-10 records, and comprehensive column statistics (`mean`, `std`, `min`, `max`, `quantiles`).
  3. **Results & Visualizations View:**
     - 8 Key Performance Indicator (KPI) cards: Total records, Normal records, Suspicious records, Anomalous records, Anomaly %, Average Latency, Average Packet Loss, Average Jitter.
     - **Anomaly Score Chart:** Scatter/line chart showing individual sample scores with horizontal threshold lines for Suspicious ($\tau_{95}$) and Critical ($\tau_{99}$).
     - **Multi-channel QoS Chart:** Synchronized time-series subplots for Latency, Packet Loss, and Jitter.
     - **Distribution Donut Chart:** Proportional split of Normal vs Suspicious vs Anomalous traffic.
  4. **Anomaly Details & Diagnostics View:**
     - An interactive, color-highlighted table of records with status filtering (`All`, `Anomalous`, `Suspicious`, `Normal`).
     - **Root-Cause Inspection:** Evaluates anomalous records and flags specific unusual telemetry values:
       - *High Latency*
       - *High Packet Loss*
       - *High Jitter*
       - *High Packet Drops*
       - *High Delay Variation*
  5. **Reset / Clear Action:** "Clear Dataset" button that purges session state and returns the UI to "No dataset loaded".

### 4.2 Backend Modules (`src/`)

#### 1. Preprocessing Engine (`src/preprocessing.py`)
- **Class:** `VoIPPreprocessor`
- **Responsibilities:**
  - Validates presence of the 7 mandatory columns (`timestamp`, `packet_loss_pct`, `latency_ms`, `jitter_ms`, `packet_drop_count`, `throughput_kbps`, `delay_variation_ms`).
  - Imputes missing / NaN values using the column **median** (robust against outliers).
  - Enforces physical network constraints: clips non-negative values ($x \ge 0$) and limits `packet_loss_pct` to $[0, 100]\%$.
  - Fits and applies `StandardScaler` to ensure mean $= 0$ and variance $= 1$ across all features, preventing high-magnitude features (e.g. throughput in kbps) from dominating the covariance matrix.

#### 2. Anomaly Detection Engine (`src/anomaly_detector.py`)
- **Class:** `GMMAnomalyDetector`
- **Responsibilities:**
  - Fits `sklearn.mixture.GaussianMixture` with full covariance matrices.
  - Computes per-sample log-likelihood density $\ln p(\mathbf{x})$.
  - Inverts to negative log-likelihood: $S(\mathbf{x}) = -\ln p(\mathbf{x})$.
  - Calibrates empirical thresholds:
    - $\tau_{95} = \text{percentile}(S(\mathbf{X}), 95.0)$
    - $\tau_{99} = \text{percentile}(S(\mathbf{X}), 99.0)$
  - Performs tri-state classification:
    - $S(\mathbf{x}) \le \tau_{95} \implies \text{NORMAL}$
    - $\tau_{95} < S(\mathbf{x}) \le \tau_{99} \implies \text{SUSPICIOUS}$
    - $S(\mathbf{x}) > \tau_{99} \implies \text{ANOMALOUS}$

#### 3. Evaluation & Benchmarking (`src/evaluation.py`)
- Compares the primary GMM detector against a secondary unsupervised baseline (**Isolation Forest**).
- Computes formal classification metrics: Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC, PR-AUC, and full confusion matrix (TP, FP, TN, FN).

---

## 5. Mathematical & Algorithmic Formulations

### 5.1 Probability Density of GMM
A Gaussian Mixture Model models the probability density function of an observation vector $\mathbf{x} \in \mathbb{R}^D$ ($D=6$) as a convex combination of $K$ multivariate Gaussian distributions:

$$p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

Where:
- $\pi_k$ is the mixture weight of component $k$, with $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \ge 0$.
- $\boldsymbol{\mu}_k \in \mathbb{R}^6$ is the centroid mean vector of component $k$.
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{6 \times 6}$ is the symmetric positive-definite covariance matrix of component $k$.
- The multivariate normal distribution is:
  $$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}_k|^{1/2}} \exp \left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) \right)$$

### 5.2 Expectation-Maximization (EM) Parameter Estimation
The parameters $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K$ are estimated iteratively:
1. **E-Step (Expectation):** Compute the posterior probability (responsibility) that component $k$ generated sample $\mathbf{x}_n$:
   $$\gamma_{nk} = \frac{\pi_k \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$
2. **M-Step (Maximization):** Update parameters using the responsibilities:
   $$N_k = \sum_{n=1}^N \gamma_{nk}, \quad \pi_k = \frac{N_k}{N}, \quad \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} \mathbf{x}_n$$
   $$\boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$$

### 5.3 Optimal Component Selection (BIC & AIC)
To select the optimal number of components $K \in \{1, 2, 3, 4, 5\}$ and prevent overfitting:
- **Bayesian Information Criterion (BIC):**
  $$\text{BIC} = -2 \ln \hat{L} + p \ln(N)$$
- **Akaike Information Criterion (AIC):**
  $$\text{AIC} = -2 \ln \hat{L} + 2p$$
Where $p = (K-1) + K \cdot D + K \frac{D(D+1)}{2}$ is the number of free parameters. The model with the minimum BIC is chosen.

### 5.4 Anomaly Scoring Metric
Because likelihood $p(\mathbf{x}) \in [0, 1]$, log-likelihood $\ln p(\mathbf{x}) \le 0$. Anomaly score is defined as:
$$S(\mathbf{x}) = -\ln p(\mathbf{x})$$
- Healthy, frequent traffic patterns $\implies$ high $p(\mathbf{x}) \implies$ **low score $S(\mathbf{x})$**.
- Irregular spikes and compound degradation $\implies$ near-zero $p(\mathbf{x}) \implies$ **high score $S(\mathbf{x})$**.

---

## 6. Dataset Schema & Telemetry Features

The system expects a CSV file containing real or simulated VoIP trunk telemetry:

| Column Name | Data Type | Physical Unit | Normal Operational Range | Anomaly Threshold Condition |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | String / DateTime | ISO-8601 / Standard | Chronological timestamps | Temporal indexing |
| `packet_loss_pct` | Float | Percentage ($\%$) | $0.0 - 1.95\%$ (mean: $0.41\%$) | Loss surges ($> 5\%$) |
| `latency_ms` | Float | Milliseconds ($\text{ms}$) | $12.0 - 78.0\text{ ms}$ | Latency spikes ($> 150\text{ ms}$) |
| `jitter_ms` | Float | Milliseconds ($\text{ms}$) | $0.8 - 14.5\text{ ms}$ | Jitter bursts ($> 25\text{ ms}$) |
| `packet_drop_count`| Integer / Float | Packets / interval | $0 - 2$ drops | Sudden burst drops ($> 5$) |
| `throughput_kbps` | Float | Kilobits per sec ($\text{kbps}$) | $65.0 - 95.0\text{ kbps}$ | Severe bandwidth throttling / collapse |
| `delay_variation_ms`| Float | Milliseconds ($\text{ms}$) | $0.5 - 8.0\text{ ms}$ | Route instability / bufferbloat |

---

## 7. Experimental Results & Benchmarks

When benchmarked against a 10,000-sample ITU-T compliant dataset containing 805 true injected anomalies:

| Evaluation Metric | GMM (Primary Proposed) | Isolation Forest (Baseline) | Advantage |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **99.08%** | 96.48% | **+2.60%** |
| **Precision** | **89.74%** | 85.56% | **+4.18%** |
| **Recall (Sensitivity)**| **100.00%** | 67.70% | **+32.30% (Zero Missed Anomalies)** |
| **F1-Score** | **0.9459** | 0.7559 | **+0.1900** |
| **ROC-AUC** | **1.0000** | 0.9908 | Perfect discrimination |
| **PR-AUC** | **1.0000** | 0.8864 | **+0.1136** |
| **True Positives (TP)**| **805 / 805** | 545 / 805 | GMM captured all 805 faults |
| **False Negatives (FN)**| **0** | **260** | Baseline missed 260 real anomalies |

---

## 8. Complete 20-Slide PPT Presentation Deck Outline

Use this exact structure when preparing your Microsoft PowerPoint or Google Slides presentation:

### Slide 1: Title Slide
- **Title:** Voice Trunk Anomaly Scoring
- **Subtitle:** GMM-Based Anomaly Detection for Real-Time Audio Communication Infrastructure
- **Domain:** Pattern Recognition and Anomaly Detection (PRAD)
- **Presenter Name:** [Your Name / Roll No]
- **Supervisor / Department:** Department of Computer Science & Engineering / AI & Data Science

### Slide 2: Introduction & Background
- Digital voice transmission (VoIP, SIP, WebRTC) uses UDP/RTP packets.
- Real-time conversational constraints: unlike video streaming, conversational audio cannot tolerate buffering delays without collision.
- Stringent ITU-T standards: latency $< 150\text{ ms}$, packet loss $< 1\%$.

### Slide 3: Problem Statement
- Traditional monitoring uses fixed thresholds (`latency > 100ms`).
- Inability to detect correlated/multi-variate network faults (e.g. moderate jitter + moderate loss).
- Class imbalance and lack of ground-truth anomaly labels prevent supervised training.
- Need for an automated, unsupervised statistical pattern recognition solution.

### Slide 4: Project Objectives
- Build an unsupervised anomaly detection engine based on Gaussian Mixture Models.
- Formulate a mathematically grounded anomaly score via negative log-likelihood.
- Establish an automated tri-state classification hierarchy: `NORMAL`, `SUSPICIOUS`, `ANOMALOUS`.
- Develop a clean, interactive real-time web prototype using Streamlit.

### Slide 5: Literature Review & Limitations of Prior Work
- Rule-based SNMP alarms: fragile, cause alarm fatigue.
- K-Means Clustering: assumes spherical clusters; cannot model correlation between latency and jitter.
- Principal Component Analysis (PCA): linear projection; loses non-linear density boundaries.
- Proposed Solution: Full-covariance Gaussian Mixture Models.

### Slide 6: System Architecture & Workflow
- Complete pipeline diagram (Upload $\to$ Validate $\to$ Preprocess $\to$ Scale $\to$ GMM $\to$ Score $\to$ Threshold $\to$ Classify $\to$ Dashboard).
- Separation of concerns between preprocessing, model engine, and UI layer.

### Slide 7: Mathematical Foundation: Gaussian Mixture Models
- Mathematical definition: $p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.
- Explanation of mixture weights $\pi_k$, mean vectors $\boldsymbol{\mu}_k$, and covariance matrices $\boldsymbol{\Sigma}_k$.
- Use of full covariance to capture inter-metric dependencies.

### Slide 8: Parameter Optimization: EM Algorithm
- The Expectation-Maximization (EM) framework.
- E-Step: Computing posterior responsibility $\gamma_{nk}$.
- M-Step: Updating weights, means, and covariance matrices until likelihood convergence.

### Slide 9: Model Selection via Information Criteria (BIC & AIC)
- Formulations of Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC).
- Penalty for free parameters: $p \ln(N)$.
- Evaluation across $K \in \{1, 2, 3, 4, 5\}$ and finding the minimum BIC curve.

### Slide 10: Anomaly Scoring & Threshold Derivation
- Formulation: $S(\mathbf{x}) = -\ln p(\mathbf{x})$.
- Log-likelihood density interpretation: tail events yield high anomaly scores.
- Threshold calibration: $\tau_{95}$ (95th percentile) for Suspicious and $\tau_{99}$ (99th percentile) for Anomalous alerts.

### Slide 11: Technology Stack
- Language: Python 3.11
- Machine Learning: Scikit-Learn, SciPy, NumPy
- Data Processing: Pandas
- Frontend: Streamlit
- Visualization: Plotly, Matplotlib, Seaborn

### Slide 12: Telemetry Dataset & QoS Metrics
- 6 key QoS metrics: `packet_loss_pct`, `latency_ms`, `jitter_ms`, `packet_drop_count`, `throughput_kbps`, `delay_variation_ms`.
- Synthetic modeling adhering to ITU-T G.114 & G.1010 standards.
- Types of injected faults: latency spikes, drop bursts, jitter storms, combined failures.

### Slide 13: Data Preprocessing & Validation Pipeline
- Schema verification (ensuring all 7 columns exist).
- Robust median imputation for handling missing/corrupted values.
- Non-negative clipping and domain constraints.
- `StandardScaler` normalization to zero mean and unit variance.

### Slide 14: Frontend Architecture & User Interface
- Clean, data-driven design: Starts with "No dataset loaded".
- Drag-and-drop CSV uploader with real-time feedback.
- Instant feedback with row counts, schema verification, and statistical summary.

### Slide 15: Interactive Visualizations & Dashboard
- Metric cards for immediate executive overview.
- Interactive Plotly timeline of anomaly scores with threshold boundaries.
- Multi-metric synchronized timelines for Latency, Packet Loss, and Jitter.
- Donut chart depicting normal vs suspicious vs anomalous ratio.

### Slide 16: Root Cause Diagnostics & Inspection
- Filterable anomaly records table.
- Identification of specific contributing factors (High Latency, High Jitter, High Loss, High Drops) based on baseline distributions.
- Zero invented explanations; completely grounded in uploaded telemetry.

### Slide 17: Experimental Results & Benchmarking
- Comparison table: GMM vs Isolation Forest.
- GMM achieves 99.08% accuracy and 100% recall (zero missed anomalies).
- Isolation Forest misses 260 anomalies (recall 67.70%).

### Slide 18: Confusion Matrix & ROC-AUC Analysis
- Presentation of confusion matrix: 805 True Positives, 0 False Negatives.
- ROC-AUC $= 1.0000$ and PR-AUC $= 1.0000$.
- Mathematical justification for GMM's superior density boundary discrimination.

### Slide 19: Future Enhancements
- Extension to semi-supervised learning with active operator feedback.
- Deployment on live SIP proxy mirrors via eBPF or packet capture (PCAP).
- Containerization with Docker and integration into Prometheus / Grafana alerts.

### Slide 20: Conclusion & References
- Summary of project achievements.
- Key takeaways: Unsupervised GMM is highly effective for telecom QoS anomaly detection.
- Core references: ITU-T G.114, Bishop's PRML, Scikit-learn documentation.
- Q&A / Discussion.

---

## 9. Academic Project Report Blueprint

When formatting your final documentation, use this chapter-by-chapter blueprint:

- **Cover Page:** Project Title, Candidate Name, Roll Number, Guide Name, Institution Emblem.
- **Certificate of Authenticity & Declaration.**
- **Acknowledgements & Abstract.**
- **Chapter 1: Introduction**
  - Background of VoIP and SIP Trunking Infrastructure.
  - Challenges in Real-Time Quality of Service (QoS).
  - Motivation for Machine Learning in Telecom Monitoring.
  - Problem Statement and Project Objectives.
- **Chapter 2: Literature Survey & Theoretical Background**
  - Traditional SNMP & Rule-Based Monitoring.
  - Distance-Based vs. Density-Based Outlier Detection.
  - Overview of Gaussian Mixture Models and Multivariate Normal Distributions.
  - The Expectation-Maximization (EM) Algorithm.
  - Model Selection Criteria: BIC and AIC.
- **Chapter 3: System Requirements & Architecture**
  - Functional and Non-Functional Requirements.
  - Hardware and Software Specifications.
  - High-Level System Architecture and Pipeline Flowchart.
- **Chapter 4: Methodology & Implementation**
  - Data Validation and Schema Enforcement.
  - Preprocessing and Feature Standardization (`StandardScaler`).
  - GMM Fitting and Component Selection.
  - Negative Log-Likelihood Anomaly Scoring ($S(\mathbf{x}) = -\ln p(\mathbf{x})$).
  - Percentile Thresholding and Tri-State Classification.
  - Frontend Implementation in Streamlit with Reactive State Management.
- **Chapter 5: Experimental Evaluation & Comparative Study**
  - Experimental Setup and Dataset Characteristics.
  - Hyperparameter Sweep Results ($K \in \{1..5\}$).
  - Performance Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC.
  - Comparative Analysis: GMM vs Isolation Forest.
  - Confusion Matrix Analysis.
- **Chapter 6: Software Demonstration & Walkthrough**
  - "No Dataset Loaded" Initial View.
  - CSV Ingestion and Schema Validation.
  - Dashboard Metrics and Interactive Plotly Visualizations.
  - Root Cause Diagnostic Inspection.
  - Dataset Clearing and Reset Behavior.
- **Chapter 7: Conclusion & Future Scope**
  - Summary of Accomplishments.
  - Limitations and Assumptions.
  - Future Enhancements (eBPF packet sniffing, real-time alerting).
- **References & Bibliography (IEEE format).**
