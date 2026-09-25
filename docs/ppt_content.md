# Voice Trunk Anomaly Scoring - Academic Presentation Deck

**Subject:** Pattern Recognition and Anomaly Detection (PRAD)  
**Project Title:** Voice Trunk Anomaly Scoring: GMM-Based Anomaly Detection for Real-Time Audio Communication Infrastructure  
**Format:** 20-Slide Formal Presentation Structure  

---

### Slide 1 – Title Slide
- **Title:** Voice Trunk Anomaly Scoring
- **Subtitle:** GMM-Based Anomaly Detection for Real-Time Audio Communication Infrastructure
- **Subject:** Pattern Recognition and Anomaly Detection (PRAD)
- **Primary Algorithm:** Gaussian Mixture Models (Unsupervised Learning)
- **Presenter:** [Student Name / Roll Number]
- **Department:** Computer Science & Engineering / AI & Data Science
- **Institution:** [College / University Name]

---

### Slide 2 – Introduction
- **Context:** Voice over IP (VoIP) and SIP trunking carry critical conversational audio across enterprise and carrier networks.
- **Audio Quality Fragility:** Unlike web traffic (TCP) which tolerates buffering, real-time voice (RTP/UDP) cannot retransmit lost frames without human-perceptible degradation.
- **SLA Requirements:** Low delay ($\le 150\text{ ms}$), minimal jitter ($\le 20\text{ ms}$), and near-zero packet drops ($\le 1\%$).
- **Goal:** Autonomous, statistical anomaly detection to identify degradation before customers drop calls.

---

### Slide 3 – Problem Statement
- **Official Problem Statement:** *"Implement a Gaussian Mixture Model to detect irregular packet drops or latency spikes in real-time audio communication infrastructure."*
- **Key Challenges:**
  - Network anomalies are rare, multi-variate, and unpredictable.
  - Labeled failure datasets do not exist in operational environments.
  - Heuristic thresholds (e.g. `latency > 100ms`) fail when cross-feature correlations degrade voice quality.

---

### Slide 4 – Objectives
- Formulate an **unsupervised statistical learning pipeline** using Gaussian Mixture Models.
- Model multi-modal normal network behavior across 6 QoS metrics.
- Derive a mathematically rigorous **anomaly score based on negative log-likelihood**.
- Establish a **tri-state classification hierarchy** (`NORMAL`, `SUSPICIOUS`, `ANOMALOUS`).
- Deliver an interactive, real-time Streamlit monitoring dashboard for live demonstration.

---

### Slide 5 – Existing Heuristics vs. Statistical Learning
- **Static Thresholding (SNMP/Nagios):** Fixed scalar alarms cause alarm fatigue or miss multi-metric coupled degradation (e.g. moderate jitter + moderate loss = total audio clipping).
- **Supervised Machine Learning:** Unviable due to severe class imbalance ($<5\%$ anomalies) and inability to detect novel, unseen zero-day network faults.
- **Proposed Unsupervised Approach:** Learn the joint probability density $p(\mathbf{x})$ of healthy operational traffic and flag low-probability tail excursions.

---

### Slide 6 – Proposed System Solution
- **Density-Based Outlier Detection:** Learn Gaussian mixture components characterizing diverse voice routes.
- **Continuous Likelihood Scoring:** Compute exact probability density via $p(\mathbf{x}) = \sum \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.
- **Log-Likelihood Score Inversion:** Score $S(\mathbf{x}) = -\ln p(\mathbf{x})$. Higher score = further in distribution tails.
- **Statistical Thresholds:** Empirical 95th and 99th percentiles establish non-arbitrary operational alerts.

---

### Slide 7 – System Architecture & Pipeline
- **End-to-End Pipeline:**
  1. *Telemetry Ingestion* $\to$ 2. *Preprocessing & Standardization* $\to$ 3. *GMM Density Estimation* $\to$ 4. *Log-Likelihood Evaluation* $\to$ 5. *Threshold Comparison* $\to$ 6. *Dashboard Alerting & Fault Injection*.
- **Offline Stage:** Model selection across $K \in \{1..5\}$ via BIC/AIC and StandardScaler calibration.
- **Online Stage:** Sub-millisecond single-frame scoring and rolling live visualizations.

---

### Slide 8 – Dataset & VoIP Simulation Parameters
- **Dataset Size:** 10,000 chronological observations (simulating ~2.7 hours of 1-second VoIP trunk telemetry).
- **ITU-T G.114/G.1010 Normal Operational Baselines:**
  - Packet Loss: $0.0 - 1.95\%$ (mean: $0.41\%$)
  - One-Way Latency: $12.0 - 78.0\text{ ms}$ (multi-modal local vs interstate routes)
  - Jitter: $0.8 - 14.5\text{ ms}$
  - Packets Sent: ~50 packets/sec (20ms G.711 / Opus voice frames)
- **Synthetic Injected Anomalies (8.05% ratio):**
  - High Packet Loss ($6 - 24\%$), High Latency ($150 - 520\text{ ms}$), High Jitter ($32 - 92\text{ ms}$), Combined Degradation, and Drop Bursts.

---

### Slide 9 – Data Preprocessing Pipeline
- **Integrity Validation:** Automatic schema verification and non-negative physical boundary enforcement.
- **Missing Value Handling:** Robust median imputation preserving statistical distributions.
- **Feature Standardization (`StandardScaler`):**
  $$z = \frac{x - \mu}{\sigma}$$
  Prevents latency ($50\text{ ms}$) from mathematically overpowering packet loss ($1\%$) in the Euclidean/Mahalanobis distance space.
- **Consistency:** Exact same fitted scaler artifact serialized and applied during live inference.

---

### Slide 10 – Gaussian Mixture Model Formulation
- **Probability Density Function:**
  $$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
- **Optimization:** Expectation-Maximization (EM) algorithm iterating E-step (responsibilities $\gamma_{nk}$) and M-step (weights, means, covariances).
- **Model Selection via BIC:**
  - Evaluated $K \in \{1, 2, 3, 4, 5\}$.
  - $K=5$ selected with lowest BIC ($-138,744.39$).
  - Full covariance matrices capture joint latency-jitter correlations.

---

### Slide 11 – Anomaly Scoring & Threshold Derivation
- **Mathematical Anomaly Score:**
  $$S(\mathbf{x}) = -\ln p(\mathbf{x}) = -\text{score\_samples}(\mathbf{x})$$
  Maps rare tail events ($p(\mathbf{x}) \to 0$) to large positive anomaly scores.
- **Data-Driven Percentile Cutoffs:**
  - $\tau_{95} = -3.8321$ $\implies$ **NORMAL** ($S \le \tau_{95}$)
  - $\tau_{99} = +0.0856$ $\implies$ **SUSPICIOUS** ($\tau_{95} < S \le \tau_{99}$)
  - Extreme Tail $\implies$ **ANOMALOUS** ($S > \tau_{99}$)
- Avoids arbitrary hardcoded thresholds; backed by training empirical CDF.

---

### Slide 12 – Real-Time Live Monitoring Simulation
- **Classroom Interactive Demonstration:**
  - Live streaming ticker with dynamic sub-second rendering.
  - Interactive fault injection: `🟢 Normal Frame`, `⚡ Latency Spike`, `📉 Loss Burst`, `💥 Combined Failure`.
- **Instantaneous Feedback:** Real-time calculation of $S(\mathbf{x})$, status badge transition, and rolling multi-channel Plotly charts.

---

### Slide 13 – Experimental Results & Model Performance
- **Primary GMM Evaluation Metrics (at $\tau_{99}$):**
  - **Accuracy:** **99.08%**
  - **Precision:** **89.74%**
  - **Recall (Sensitivity):** **100.00%** (Zero missed network failures)
  - **F1-Score:** **0.9459**
  - **Specificity:** **99.00%**
  - **ROC-AUC:** **1.0000** | **PR-AUC:** **1.0000**
- **Zero False Negatives:** All 805 injected SLA-violating events were caught.

---

### Slide 14 – Baseline Comparison: GMM vs. Isolation Forest
- **Head-to-Head Benchmark:**
  - **Accuracy:** GMM **99.08%** vs Isolation Forest 96.48%
  - **Recall:** GMM **100.00%** vs Isolation Forest **67.70%** (+32.3% gain!)
  - **F1-Score:** GMM **0.9459** vs Isolation Forest 0.7559
  - **Missed Incidents (FN):** GMM **0** vs Isolation Forest **260**
- **Takeaway:** GMM's smooth multi-modal ellipsoids preserve voice cluster boundaries significantly better than orthogonal decision tree cuts.

---

### Slide 15 – Application Demonstration & Screenshots
- **Integrated Streamlit Dashboard Pages:**
  - `01_dashboard_overview.png`: High-level network health KPIs and incident history.
  - `02_live_monitoring.png`: Live packet inspection and fault injection console.
  - `03_anomaly_analysis.png`: Interactive threshold sliders and score distribution.
  - `04_gmm_diagnostics.png`: BIC/AIC elbow curve and mathematical formulas.
  - `05_model_evaluation.png`: Confusion matrix and benchmark comparison.

---

### Slide 16 – Real-World Telecom Applications
- **Enterprise Contact Centers:** Real-time voice agent quality assurance.
- **VoIP / SIP Carrier Peering:** Proactive automated trunk failover and route rerouting.
- **Unified Communications (UCaaS):** Automated SLA credit reporting for Zoom/Teams/Webex trunks.
- **Audio Streaming Infrastructure:** Buffer underrun prevention in digital broadcasting.

---

### Slide 17 – Project Limitations & Assumptions
- **Academic Simulation Assumption:** Conducted on realistic synthetic telemetry adhering to ITU-T standards rather than proprietary carrier taps.
- **Stationary Assumption:** Assumes seasonal traffic patterns remain relatively constant; requires scheduled retraining as network infrastructure expands.
- **Computational Cost:** Full covariance GMM requires matrix inversion $\mathcal{O}(D^3)$, though for $D=6$ inference is sub-millisecond.

---

### Slide 18 – Future Scope
- **Online Adaptive EM:** Implement incremental Gaussian Mixture updates to adapt dynamically to seasonal traffic shifts.
- **Automated Root Cause Diagnosis:** Bayesian component attribution to pinpoint whether latency, jitter, or loss drove the anomaly.
- **eBPF Kernel Integration:** Direct kernel-level packet inspection using extended Berkeley Packet Filters for zero-overhead telemetry capture.

---

### Slide 19 – Conclusion
- Successfully designed, implemented, and benchmarked an unsupervised **Gaussian Mixture Model for Voice Trunk Anomaly Scoring**.
- Negative log-likelihood scoring offers an intuitive, mathematically sound measure of voice trunk health.
- Achieved **99.08% accuracy** and **100% recall**, outperforming Isolation Forest by 32.3% recall.
- Delivered a full-stack, classroom-ready Streamlit application with real-time fault injection.

---

### Slide 20 – References & Citations
1. ITU-T Recommendation G.114: *"One-way transmission time."* International Telecommunication Union, 2003.
2. ITU-T Recommendation G.1010: *"End-user multimedia QoS categories."* ITU, 2001.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer, Chapter 9 (Mixture Models and EM).
4. Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly detection: A survey.* ACM Computing Surveys (CSUR).
5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.
