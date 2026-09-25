# Experimental Results and Evaluation Benchmarks

**Subject:** Pattern Recognition and Anomaly Detection (PRAD)  
**Document:** Empirical Model Evaluation & Baseline Comparison Report  
**Dataset:** Simulated VoIP Trunk Telemetry (10,000 observations)  

---

## 1. Experimental Setup & Dataset Profile

The experimental evaluation was conducted on a chronologically simulated VoIP trunk telemetry dataset consisting of **10,000 consecutive 1-second telemetry frames** modeled after ITU-T G.114/G.1010 recommendations.

- **Total Observations:** 10,000
- **Normal Baseline Observations:** 9,195 (91.95%)
- **Ground-Truth Anomalies:** 805 (8.05%)
- **Input Dimension:** 6 numerical features (`packet_loss_pct`, `latency_ms`, `jitter_ms`, `packet_drop_count`, `throughput_kbps`, `delay_variation_ms`)
- **Scaling:** Standardized via `StandardScaler` fitted exclusively on normal baseline traffic.

### Anomaly Breakdown by Category
| Anomaly Category | Count | Percentage of Anomalies | Simulated Impact |
| :--- | :--- | :--- | :--- |
| High Packet Loss | 274 | 34.04% | Loss spikes between $6.0\%$ and $24.0\%$ |
| High Latency | 206 | 25.59% | Latency spikes between $150.0\text{ ms}$ and $520.0\text{ ms}$ |
| Combined Degradation | 155 | 19.25% | Simultaneous packet loss, high latency, and high jitter |
| High Jitter | 110 | 13.66% | Jitter surges between $32.0\text{ ms}$ and $92.0\text{ ms}$ |
| Packet Drop Burst | 60 | 7.45% | Sudden drop spikes between $18.0\%$ and $38.0\%$ |

---

## 2. Hyperparameter Selection: BIC & AIC Analysis

To select the optimal number of Gaussian mixture components $K$, grid search was conducted across $K \in \{1, 2, 3, 4, 5\}$ using a full covariance structure on the $9,195$ baseline normal observations.

| Components ($K$) | Free Parameters ($p$) | Log-Likelihood ($\ln L$) | BIC (Bayesian Info Crit) | AIC (Akaike Info Crit) |
| :---: | :---: | :---: | :---: | :---: |
| $K = 1$ | 27 | 19,238.56 | -38,230.71 | -38,423.13 |
| $K = 2$ | 55 | 65,965.36 | -131,428.77 | -131,820.72 |
| $K = 3$ | 83 | 66,057.08 | -131,356.68 | -131,948.17 |
| $K = 4$ | 111 | 68,218.94 | -135,424.85 | -136,215.88 |
| **$K = 5$** | **139** | **70,006.48** | **-138,744.39** | **-139,734.96** |

**Selection Rationale:**  
$K=5$ achieved the **lowest BIC value ($-138,744.39$)** and the lowest AIC value ($-139,734.96$). The significant BIC reduction from $K=1$ to $K=2$ reflects the multi-modal routing characteristics of network trunks, while $K=5$ captures subtle variations in traffic load and jitter dispersion without penalizing model generalization.

---

## 3. Anomaly Score Distribution & Decision Boundaries

Using the negative log-likelihood formula $S(\mathbf{x}) = -\ln p(\mathbf{x} \mid \boldsymbol{\theta})$:

- **Baseline Score Mean ($\mu_{\text{train}}$):** $-7.6135$
- **Baseline Score Std ($\sigma_{\text{train}}$):** $2.1590$
- **Suspicious Threshold ($\tau_{95}$):** $-3.8321$ (95th percentile of normal baseline)
- **Critical Anomaly Threshold ($\tau_{99}$):** $+0.0856$ (99th percentile of normal baseline)

---

## 4. Quantitative Evaluation Metrics

Evaluating the GMM detector using the critical decision threshold $\tau_{99} = 0.0856$ against ground-truth synthetic labels yielded the following performance:

| Metric | Empirical Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **99.08%** | Extremely high overall correct classification rate across all 10,000 frames. |
| **Precision** | **89.74%** | $89.74\%$ of all flagged critical anomalies were confirmed ground-truth faults. |
| **Recall (Sensitivity)** | **100.00%** | **Zero missed anomalies**: Every single one of the 805 network faults was detected. |
| **F1-Score** | **0.9459** | Harmonic mean of precision and recall demonstrating exceptional balance. |
| **Specificity** | **99.00%** | Accurately preserved $99.00\%$ of normal operational traffic without false alarms. |
| **ROC-AUC** | **1.0000** | Perfect separation across the continuous score spectrum. |
| **PR-AUC** | **1.0000** | Perfect precision-recall area under curve across all threshold variations. |

### Confusion Matrix Breakdown
```
                       Predicted Normal    Predicted Anomaly
Actual Normal (9195)         9,103 (TN)             92 (FP)
Actual Anomaly (805)             0 (FN)            805 (TP)
```
- **True Positives (TP): 805** — All 805 synthetic anomalies were successfully captured.
- **False Positives (FP): 92** — 92 baseline samples fell into the top 1% tail (matching the statistical 99th percentile design).
- **False Negatives (FN): 0** — Zero SLA-impacting incidents bypassed the detector.
- **True Negatives (TN): 9,103** — Normal voice calls experienced uninterrupted transmission.

---

## 5. Comparative Benchmark: GMM vs. Isolation Forest

To benchmark the primary GMM algorithm against an alternative unsupervised anomaly detector, an **Isolation Forest** (150 trees, 8% contamination) was trained on the identical preprocessed baseline data.

| Performance Metric | GMM Anomaly Detector (Primary) | Isolation Forest (Baseline) | Performance Advantage |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **99.08%** | 96.48% | **+2.60%** improvement |
| **Precision** | **89.74%** | 85.56% | **+4.18%** improvement |
| **Recall (Sensitivity)** | **100.00%** | 67.70% | **+32.30%** massive gain |
| **F1-Score** | **0.9459** | 0.7559 | **+0.1900** gain |
| **ROC-AUC** | **1.0000** | 0.9908 | Superior continuous separation |
| **PR-AUC** | **1.0000** | 0.8864 | **+0.1136** gain |
| **True Positives (TP)** | **805 / 805** | 545 / 805 | GMM detected 260 more faults |
| **False Negatives (FN)** | **0** | **260** | Isolation Forest missed 260 anomalies |
| **False Positives (FP)** | 92 | 92 | Identical false alarm rate |
| **True Negatives (TN)** | 9,103 | 9,103 | Identical specificity |

### Why GMM Outperformed Isolation Forest
1. **Density Gradient Continuity:** VoIP network degradations are continuous multi-feature phenomena. Isolation Forest isolates anomalies using axis-aligned orthogonal cuts, which struggles with diagonal joint correlations (e.g. latency scaling with jitter).
2. **Multi-Modal Density Representation:** Normal VoIP traffic naturally clusters into multiple routing clusters. GMM explicitly fits multi-modal Gaussian components with full covariance matrices, effectively enclosing normal multi-hop clusters while leaving anomalies outside in near-zero likelihood territory.
