"""
Synthetic Data Generator for Voice Trunk Telemetry Simulation.

Generates realistic VoIP/SIP network telemetry conforming to ITU-T G.114/G.1010 recommendations.
Simulates normal network conditions along with realistic anomalies (latency spikes, packet loss bursts,
jitter surges, and multi-factor network degradations).
"""

import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_voip_telemetry(
    n_samples: int = 10000,
    anomaly_ratio: float = 0.08,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic VoIP trunk telemetry dataset with normal and anomalous conditions.

    Parameters:
    -----------
    n_samples : int
        Total number of chronological time-series observations.
    anomaly_ratio : float
        Proportion of observations containing synthetic anomalies (0.05 to 0.15).
    random_state : int
        Seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        VoIP network telemetry DataFrame.
    """
    np.random.seed(random_state)
    start_time = datetime(2026, 3, 1, 9, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_samples)]

    # 1. Base Normal Telemetry Generation
    # VoIP typical frame rate: 50 packets/sec (20ms packetization interval, G.711 / Opus)
    packets_sent = np.random.poisson(lam=50, size=n_samples).astype(float)
    packets_sent = np.clip(packets_sent, 42, 60)

    # Normal latency: 10 - 75 ms (multi-modal Gaussian representing local vs interstate routes)
    route_mix = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    latency_normal = np.where(
        route_mix == 0,
        np.random.normal(loc=28.0, scale=6.0, size=n_samples),   # Local/Regional trunk
        np.random.normal(loc=58.0, scale=8.0, size=n_samples)    # Inter-region trunk
    )
    latency = np.clip(latency_normal, 12.0, 78.0)

    # Normal jitter: 1 - 12 ms (log-normal distribution)
    jitter = np.random.lognormal(mean=1.2, sigma=0.45, size=n_samples)
    jitter = np.clip(jitter, 0.8, 14.5)

    # Normal packet loss percentage: 0.0 - 1.8%
    packet_loss_pct = np.random.exponential(scale=0.35, size=n_samples)
    packet_loss_pct = np.clip(packet_loss_pct, 0.0, 1.95)

    # Derived packets received & drops
    packet_drop_count = np.round(packets_sent * (packet_loss_pct / 100.0)).astype(float)
    packets_received = packets_sent - packet_drop_count

    # Nominal throughput (G.711 codec is 64 kbps payload + ~20 kbps RTP/UDP/IP header = ~84 kbps)
    throughput_kbps = packets_received * 1.68 + np.random.normal(0, 1.5, n_samples)
    throughput_kbps = np.clip(throughput_kbps, 40.0, 110.0)

    # Delay variation (PDV - Packet Delay Variation): related to jitter
    delay_variation_ms = jitter * np.random.uniform(0.6, 1.4, n_samples)

    # Anomaly labels initialization
    is_anomaly = np.zeros(n_samples, dtype=int)
    anomaly_type = np.array(["Normal"] * n_samples, dtype=object)

    # 2. Inject Structured Realistic Anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    
    # We create anomalies in realistic continuous clusters/bursts as well as point anomalies
    current_idx = 100
    anomaly_categories = [
        "High Packet Loss",
        "High Latency",
        "High Jitter",
        "Combined Degradation",
        "Packet Drop Burst"
    ]

    while np.sum(is_anomaly) < n_anomalies and current_idx < n_samples - 50:
        burst_length = np.random.choice([3, 5, 8, 12, 20])
        end_idx = min(current_idx + burst_length, n_samples)
        atype = np.random.choice(anomaly_categories, p=[0.25, 0.25, 0.20, 0.20, 0.10])

        if atype == "High Packet Loss":
            loss_spike = np.random.uniform(6.0, 24.0, end_idx - current_idx)
            packet_loss_pct[current_idx:end_idx] = loss_spike
            packet_drop_count[current_idx:end_idx] = np.round(packets_sent[current_idx:end_idx] * (loss_spike / 100.0))
            packets_received[current_idx:end_idx] = packets_sent[current_idx:end_idx] - packet_drop_count[current_idx:end_idx]
            throughput_kbps[current_idx:end_idx] = packets_received[current_idx:end_idx] * 1.68

        elif atype == "High Latency":
            # Bufferbloat / routing loop spike: 160ms - 480ms
            lat_spike = np.random.normal(loc=260.0, scale=60.0, size=end_idx - current_idx)
            lat_spike = np.clip(lat_spike, 150.0, 520.0)
            latency[current_idx:end_idx] = lat_spike
            delay_variation_ms[current_idx:end_idx] = delay_variation_ms[current_idx:end_idx] + np.random.uniform(15.0, 45.0)

        elif atype == "High Jitter":
            # Queue flapping / traffic congestion: 32ms - 95ms
            jit_spike = np.random.uniform(32.0, 92.0, end_idx - current_idx)
            jitter[current_idx:end_idx] = jit_spike
            delay_variation_ms[current_idx:end_idx] = jit_spike * np.random.uniform(1.2, 1.8, end_idx - current_idx)

        elif atype == "Combined Degradation":
            # Fiber micro-cut / congested peering exchange
            loss_spike = np.random.uniform(7.0, 22.0, end_idx - current_idx)
            lat_spike = np.random.uniform(180.0, 420.0, end_idx - current_idx)
            jit_spike = np.random.uniform(35.0, 85.0, end_idx - current_idx)

            packet_loss_pct[current_idx:end_idx] = loss_spike
            latency[current_idx:end_idx] = lat_spike
            jitter[current_idx:end_idx] = jit_spike
            packet_drop_count[current_idx:end_idx] = np.round(packets_sent[current_idx:end_idx] * (loss_spike / 100.0))
            packets_received[current_idx:end_idx] = packets_sent[current_idx:end_idx] - packet_drop_count[current_idx:end_idx]
            throughput_kbps[current_idx:end_idx] = packets_received[current_idx:end_idx] * 1.68
            delay_variation_ms[current_idx:end_idx] = jit_spike * 1.5

        elif atype == "Packet Drop Burst":
            # Sudden router buffer exhaustion drop burst
            loss_spike = np.random.uniform(18.0, 38.0, end_idx - current_idx)
            packet_loss_pct[current_idx:end_idx] = loss_spike
            packet_drop_count[current_idx:end_idx] = np.round(packets_sent[current_idx:end_idx] * (loss_spike / 100.0))
            packets_received[current_idx:end_idx] = packets_sent[current_idx:end_idx] - packet_drop_count[current_idx:end_idx]
            throughput_kbps[current_idx:end_idx] = packets_received[current_idx:end_idx] * 1.68

        is_anomaly[current_idx:end_idx] = 1
        anomaly_type[current_idx:end_idx] = atype

        # Skip next interval to create realistic separation
        current_idx = end_idx + np.random.randint(40, 140)

    # Construct DataFrame
    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "packet_loss_pct": np.round(packet_loss_pct, 2),
        "latency_ms": np.round(latency, 2),
        "jitter_ms": np.round(jitter, 2),
        "packets_sent": packets_sent.astype(int),
        "packets_received": np.clip(packets_received.astype(int), 0, None),
        "packet_drop_count": np.clip(packet_drop_count.astype(int), 0, None),
        "throughput_kbps": np.round(throughput_kbps, 2),
        "delay_variation_ms": np.round(delay_variation_ms, 2),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type
    })

    return df


def generate_single_sample(condition: str = "normal") -> dict:
    """
    Generate a single telemetry packet dict for live real-time simulation.

    Parameters:
    -----------
    condition : str
        'normal', 'latency_spike', 'packet_loss', 'jitter_spike', or 'combined'
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    packets_sent = int(np.random.poisson(lam=50))
    packets_sent = max(45, min(55, packets_sent))

    if condition == "normal":
        loss = round(float(np.random.exponential(scale=0.35)), 2)
        loss = min(loss, 1.8)
        lat = round(float(np.random.normal(35.0, 8.0)), 2)
        lat = max(15.0, min(75.0, lat))
        jit = round(float(np.random.lognormal(1.2, 0.4)), 2)
        jit = max(1.0, min(14.0, jit))
        expected_type = "Normal"
    elif condition == "latency_spike":
        loss = round(float(np.random.uniform(0.5, 2.0)), 2)
        lat = round(float(np.random.uniform(220.0, 450.0)), 2)
        jit = round(float(np.random.uniform(8.0, 22.0)), 2)
        expected_type = "High Latency"
    elif condition == "packet_loss":
        loss = round(float(np.random.uniform(8.0, 28.0)), 2)
        lat = round(float(np.random.uniform(30.0, 70.0)), 2)
        jit = round(float(np.random.uniform(3.0, 15.0)), 2)
        expected_type = "High Packet Loss"
    elif condition == "jitter_spike":
        loss = round(float(np.random.uniform(0.5, 2.5)), 2)
        lat = round(float(np.random.uniform(40.0, 95.0)), 2)
        jit = round(float(np.random.uniform(45.0, 95.0)), 2)
        expected_type = "High Jitter"
    else:  # combined
        loss = round(float(np.random.uniform(10.0, 25.0)), 2)
        lat = round(float(np.random.uniform(240.0, 480.0)), 2)
        jit = round(float(np.random.uniform(40.0, 90.0)), 2)
        expected_type = "Combined Degradation"

    drop_cnt = int(round(packets_sent * (loss / 100.0)))
    recv = max(0, packets_sent - drop_cnt)
    thp = round(float(recv * 1.68 + np.random.normal(0, 1.0)), 2)
    delay_var = round(float(jit * np.random.uniform(0.8, 1.5)), 2)

    return {
        "timestamp": now,
        "packet_loss_pct": loss,
        "latency_ms": lat,
        "jitter_ms": jit,
        "packets_sent": packets_sent,
        "packets_received": recv,
        "packet_drop_count": drop_cnt,
        "throughput_kbps": thp,
        "delay_variation_ms": delay_var,
        "simulated_type": expected_type
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VoIP trunk synthetic telemetry.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of telemetry samples")
    parser.add_argument("--anomaly_ratio", type=float, default=0.08, help="Anomaly fraction (e.g., 0.08 = 8%)")
    parser.add_argument("--output", type=str, default="data/voice_trunk_data.csv", help="Output path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_voip_telemetry(n_samples=args.samples, anomaly_ratio=args.anomaly_ratio)
    df.to_csv(args.output, index=False)
    print(f"[+] Successfully generated {len(df)} samples saved to: {args.output}")
    print(f"[+] Total Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")
    print("[+] Anomaly breakdown:")
    print(df["anomaly_type"].value_counts())
