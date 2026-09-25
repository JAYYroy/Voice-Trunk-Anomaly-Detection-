# Academic Project Report: Voice Trunk Anomaly Scoring

**Subject:** Pattern Recognition and Anomaly Detection (PRAD)  
**Project Title:** Voice Trunk Anomaly Scoring: GMM-Based Anomaly Detection for Real-Time Audio Communication Infrastructure  
**Author:** [Candidate Name / Roll Number]  
**Academic Year:** 2025–2026  

---

## Abstract
Real-time conversational audio applications such as Voice over IP (VoIP) and SIP trunking are critically sensitive to transmission impairments. While static thresholding mechanisms frequently miss subtle, multi-dimensional QoS degradations, supervised learning approaches suffer from the severe scarcity of labeled telecommunications fault data. In this paper, we propose and implement an unsupervised statistical anomaly scoring system based on Gaussian Mixture Models (GMM). By fitting a multi-component Gaussian mixture over continuous telemetry metrics (packet loss, one-way latency, jitter, packet drop counts, throughput, and delay variation), our system models the normal operational probability density of voice communication infrastructure. Incoming observations are assigned an anomaly score derived from their negative log-likelihood under the trained model. Dual empirical percentile thresholds calibrate classifications into `NORMAL`, `SUSPICIOUS`, and `ANOMALOUS` states. Using a simulated dataset of 10,000 observations adhering to ITU-T G.114/G.1010 standards, our proposed model achieves 99.08% accuracy, 100.00% recall (zero missed SLA degradations), and a 1.0000 ROC-AUC, outperforming a baseline Isolation Forest by 32.30% in recall. An interactive, real-time Streamlit dashboard was developed for live telemetry inspection and fault injection.

---

## 1. Introduction
Modern digital enterprises rely heavily on Voice over IP (VoIP) and SIP trunks for customer service centers, financial trading desks, and unified enterprise collaboration. The Real-Time Transport Protocol (RTP) carrying conversational voice operates over UDP to prioritize minimal delivery latency over packet retransmission. Consequently, unmitigated packet loss results in audible clipping, latency spikes cause conversational collision (talk-over), and severe jitter causes de-jitter buffer exhaustion. 

Proactive anomaly detection on voice trunks is therefore mandatory to identify routing degradation before customer-facing communication fails.

---

## 2. Problem Statement
*"Implement a Gaussian Mixture Model to detect irregular packet drops or latency spikes in real-time audio communication infrastructure."*

Traditional network monitoring systems employ static scalar rules (e.g., alert if latency exceeds 120 ms). Such heuristics fail to detect compound anomalies where latency, jitter, and packet loss degrade simultaneously at moderate levels, or where local and international voice routes exhibit inherently different baseline delays. This project solves this challenge through probabilistic density modeling.

---

## 3. Project Objectives
1. Design and generate a realistic, standard-compliant VoIP network telemetry dataset simulating both healthy routing and diverse failure modes.
2. Implement a robust data preprocessing and scaling pipeline ensuring unified offline training and online inference.
3. Formulate an unsupervised Gaussian Mixture Model pipeline with automated model selection using BIC and AIC.
4. Derive an intuitive, mathematically grounded anomaly scoring function based on negative log-likelihood.
5. Establish statistically justified percentile thresholds for tri-state classification (`NORMAL`, `SUSPICIOUS`, `ANOMALOUS`).
6. Benchmark the GMM against an Isolation Forest baseline.
7. Build an interactive Streamlit application with live streaming simulation and fault injection.

---

## 4. Literature Survey & Theoretical Background
- **VoIP Quality Standards:** The International Telecommunication Union (ITU-T) specifies in G.114 that one-way mouth-to-ear delay must remain below 150 ms for acceptable conversational quality, and G.1010 outlines strict jitter and loss thresholds.
- **Unsupervised Anomaly Detection:** Chandola et al. (2009) categorize anomaly detection into classification-based, clustering-based, and statistical density-based methods. Density estimation techniques, notably Gaussian Mixture Models, provide continuous probability scores without imposing artificial geometric cluster boundaries.
- **Density Estimation vs Isolation Trees:** While tree-based methods like Isolation Forest perform orthogonal partitions, GMM accommodates full covariance matrices that naturally mirror the coupled physics of packet queues.

---

## 5. System Requirements
- **Operating Environment:** Python 3.10+ on Windows, Linux, or macOS.
- **Core Computational Libraries:** `numpy` (v1.24+), `pandas` (v2.0+), `scikit-learn` (v1.3+), `scipy` (v1.11+).
- **Visualization:** `matplotlib` (v3.7+), `seaborn` (v0.12+), `plotly` (v5.15+).
- **Interactive Application:** `streamlit` (v1.28+).
- **Offline Reliability:** Zero external cloud APIs, zero external databases, 100% locally reproducible.

---

## 6. Dataset Description & Simulation Parameters
A chronological synthetic dataset of 10,000 observations was generated adhering to ITU-T VoIP specifications:
- **Baseline Healthy Telemetry (9,195 samples / 91.95%):**
  - Latency: Multi-modal Gaussian distribution (28 ms local trunk, 58 ms interstate trunk, bounds [12, 78] ms).
  - Jitter: Log-normal distribution (mean: 1.2, bounds [0.8, 14.5] ms).
  - Packet Loss: Exponential distribution (mean: 0.35%, bounds [0.0, 1.95]%).
  - Frame Rate: Poisson distribution around 50 packets/sec (20 ms G.711 / Opus framing).
  - Throughput: Nominal 64 kbps payload + 20 kbps RTP/UDP/IP header (~84 kbps).
- **Injected Anomaly Classes (805 samples / 8.05%):**
  - High Packet Loss (5% to 24%), High Latency (150 ms to 520 ms), High Jitter (32 ms to 92 ms), Combined Degradation, and Drop Bursts.

---

## 7. Data Preprocessing Pipeline
1. **Schema & Integrity Validation:** Ensures expected columns are present, non-empty, and numerical.
2. **Missing Value Imputation:** Utilizes column median imputation to maintain distribution stability without outlier distortion.
3. **Bound Enforcement:** Domain physical validation clips negative network latencies and limits packet loss to $[0, 100]\%$.
4. **Standardization (`StandardScaler`):**
   $$z_i = \frac{x_i - \mu_i}{\sigma_i}$$
   Normalizes disparate feature dimensions to zero mean and unit variance.

---

## 8. Gaussian Mixture Model Formulation
The continuous probability distribution of VoIP telemetry $\mathbf{x} \in \mathbb{R}^6$ is modeled as:
$$p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

Where $\pi_k$ are component weights, $\boldsymbol{\mu}_k$ are component mean vectors, and $\boldsymbol{\Sigma}_k$ are full covariance matrices capturing cross-feature covariance. Parameters are iteratively optimized using the Expectation-Maximization (EM) algorithm:
- **E-Step:** Compute posterior probabilities $\gamma_{nk} = P(z_n = k \mid \mathbf{x}_n, \boldsymbol{\theta})$.
- **M-Step:** Update mixture weights $\pi_k$, centroids $\boldsymbol{\mu}_k$, and covariance matrices $\boldsymbol{\Sigma}_k$.

Model selection was executed by evaluating $K \in \{1, 2, 3, 4, 5\}$ using the Bayesian Information Criterion (BIC):
$$\text{BIC} = -2 \ln \hat{L} + p \ln(N)$$
$K=5$ yielded the minimum BIC ($-138,744.39$), effectively capturing multi-route telemetry variations without overfitting.

---

## 9. Anomaly Scoring & Threshold Derivation
For any incoming telemetry frame $\mathbf{x}$, the model computes its log-likelihood $\ln p(\mathbf{x})$. The anomaly score is defined as the negative log-likelihood:
$$S(\mathbf{x}) = -\ln p(\mathbf{x})$$
Observations consistent with normal baseline traffic yield high likelihoods and low anomaly scores (baseline mean: $-7.61$). Rare, SLA-violating conditions fall far into distribution tails, generating high positive anomaly scores.

Decision thresholds are calibrated using percentiles of baseline training scores:
- **`NORMAL`:** $S(\mathbf{x}) \le \tau_{95} = -3.8321$
- **`SUSPICIOUS`:** $-3.8321 < S(\mathbf{x}) \le \tau_{99} = +0.0856$
- **`ANOMALOUS`:** $S(\mathbf{x}) > +0.0856$

---

## 10. System Architecture & Implementation
The system is partitioned into clean, modular Python components:
- `src/data_generator.py`: Telemetry generator and live frame generator.
- `src/preprocessing.py`: Validation, median imputation, and `StandardScaler` wrapping.
- `src/anomaly_detector.py`: GMM scoring and classification engine.
- `src/train_model.py`: End-to-end model selection and artifact exporter.
- `src/evaluation.py`: Metric calculation and baseline comparison.
- `src/visualization.py`: 13 publication-quality static visualization generators.
- `app.py`: Full-featured Streamlit user interface.

---

## 11. Streamlit Dashboard Implementation
The graphical interface contains 7 structured navigation modules:
1. **Dashboard Overview:** Displays KPI metric cards, network health breakdown, and 3-channel time-series charts.
2. **Live Monitoring:** Real-time simulated telemetry feed with manual fault injection buttons (`Normal`, `Latency Spike`, `Loss Burst`, `Combined Failure`).
3. **Anomaly Analysis:** Interactive threshold sliders with dynamic histogram re-binning and 2D scatter projections.
4. **Dataset Explorer:** Searchable raw data table with filter sliders and custom CSV upload validation.
5. **GMM Model Diagnostics:** Mathematical formulation, BIC/AIC elbow plot, and component mixture weights.
6. **Model Evaluation:** Confusion matrix, ROC curve, and side-by-side benchmark table with Isolation Forest.
7. **About & Viva Guide:** Comprehensive answers to typical academic defense questions.

---

## 12. Experimental Results & Discussion
Quantitative evaluation using the critical anomaly threshold $\tau_{99} = 0.0856$ against ground-truth labels yielded:
- **Accuracy:** 99.08%
- **Precision:** 89.74%
- **Recall (Sensitivity):** 100.00%
- **F1-Score:** 0.9459
- **Specificity:** 99.00%
- **ROC-AUC:** 1.0000
- **Confusion Matrix:** True Positives = 805, False Positives = 92, False Negatives = 0, True Negatives = 9,103.

### Comparison with Isolation Forest
| Metric | GMM (Primary) | Isolation Forest (Baseline) | Difference |
| :--- | :---: | :---: | :---: |
| Accuracy | 99.08% | 96.48% | +2.60% |
| Recall | 100.00% | 67.70% | +32.30% |
| F1-Score | 0.9459 | 0.7559 | +0.1900 |
| Missed Faults (FN) | 0 | 260 | -260 |

GMM achieved zero false negatives, detecting all 805 injected faults. Isolation Forest missed 260 faults due to its reliance on orthogonal tree splits that fail to isolate diagonal correlations between latency and jitter.

---

## 13. Applications in Industry
- **Carrier Telecommunication Trunks:** Autonomous routing failover for SIP/PSTN interconnects.
- **Enterprise Contact Centers:** Automated agent quality scoring and MOS degradation alerts.
- **VoIP Service Providers:** Real-time SLA monitoring and automated credit billing validation.
- **Defense & Public Safety Networks:** Immediate detection of denial-of-service or jamming events on mission-critical voice channels.

---

## 14. Limitations & Future Scope
- **Assumptions:** Evaluated on realistic synthetic telemetry adhering to ITU-T standards rather than physical telco taps.
- **Stationarity:** The model assumes steady-state baseline traffic; seasonal traffic shifts require periodic retraining.
- **Future Scope:** Implement incremental online Expectation-Maximization for real-time model adaptation and integrate eBPF kernel packet probes.

---

## 15. Conclusion
This project successfully demonstrated that Gaussian Mixture Models provide a mathematically principled and highly effective solution for voice trunk anomaly scoring. By leveraging negative log-likelihood scoring and empirical percentile thresholding, the system achieves 100% recall on voice quality degradations with sub-millisecond inference latency, making it directly applicable to real-time audio communication infrastructure.

---

## 16. References
1. International Telecommunication Union (ITU-T), *"One-way transmission time,"* Recommendation G.114, 2003.
2. International Telecommunication Union (ITU-T), *"End-user multimedia QoS categories,"* Recommendation G.1010, 2001.
3. C. M. Bishop, *Pattern Recognition and Machine Learning,* Springer, New York, 2006.
4. V. Chandola, A. Banerjee, and V. Kumar, *"Anomaly detection: A survey,"* ACM Computing Surveys (CSUR), vol. 41, no. 3, pp. 1–58, 2009.
5. F. Pedregosa et al., *"Scikit-learn: Machine Learning in Python,"* Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
