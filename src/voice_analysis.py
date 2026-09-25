"""
Local voice signal analysis utilities.

This module analyzes a user's actual browser microphone recording. It is separate
from the VoIP telemetry GMM, which only models network CSV features.
"""

import os
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy.fftpack import dct


def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    """Convert integer or stereo PCM audio to mono float samples in [-1, 1]."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_abs = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_abs
    else:
        audio = audio.astype(np.float32)

    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def _frame_audio(audio: np.ndarray, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_length = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, int(sample_rate * hop_ms / 1000.0))

    if len(audio) < frame_length:
        pad_width = frame_length - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")

    n_frames = 1 + int(np.ceil((len(audio) - frame_length) / hop_length))
    pad_length = (n_frames - 1) * hop_length + frame_length
    if pad_length > len(audio):
        audio = np.pad(audio, (0, pad_length - len(audio)), mode="constant")

    indices = (
        np.tile(np.arange(frame_length), (n_frames, 1))
        + np.tile(np.arange(n_frames) * hop_length, (frame_length, 1)).T
    )
    frames = audio[indices]
    return frames * np.hamming(frame_length)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _mel_filter_bank(sample_rate: int, n_fft: int, n_filters: int = 26) -> np.ndarray:
    low_mel = _hz_to_mel(np.array([0.0]))[0]
    high_mel = _hz_to_mel(np.array([sample_rate / 2.0]))[0]
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    bank = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        for k in range(left, min(center, bank.shape[1])):
            bank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, bank.shape[1])):
            bank[m - 1, k] = (right - k) / (right - center)

    return bank


def _mfcc(audio: np.ndarray, sample_rate: int, n_coeffs: int = 13) -> List[float]:
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]) if len(audio) > 1 else audio
    frames = _frame_audio(emphasized, sample_rate)
    n_fft = 512
    power_spectrum = (1.0 / n_fft) * (np.abs(np.fft.rfft(frames, n_fft)) ** 2)

    filter_bank = _mel_filter_bank(sample_rate, n_fft)
    filter_energies = np.dot(power_spectrum, filter_bank.T)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    log_energies = np.log(filter_energies)
    coeffs = dct(log_energies, type=2, axis=1, norm="ortho")[:, :n_coeffs]
    return [float(v) for v in np.mean(coeffs, axis=0)]


def _load_audio(audio_bytes: bytes, filename: str | None = None) -> Tuple[int, np.ndarray]:
    """Decode WAV/FLAC/MP3 bytes into sample rate and mono float waveform."""
    try:
        audio, sample_rate = sf.read(BytesIO(audio_bytes), always_2d=False)
        return int(sample_rate), _to_mono_float(audio)
    except Exception:
        suffix = os.path.splitext(filename or "")[1] or ".audio"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name
            audio, sample_rate = librosa.load(temp_path, sr=None, mono=True)
            return int(sample_rate), _to_mono_float(audio)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def _basic_features(audio: np.ndarray, sample_rate: int) -> Tuple[Dict[str, Any], np.ndarray]:
    duration = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
    rms_energy = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
    zcr = float(len(zero_crossings) / max(1, len(audio) - 1))

    frames = _frame_audio(audio, sample_rate)
    n_fft = 1024
    magnitudes = np.abs(np.fft.rfft(frames, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mag_sums = np.sum(magnitudes, axis=1)
    centroids = np.divide(
        np.dot(magnitudes, freqs),
        mag_sums,
        out=np.zeros_like(mag_sums, dtype=float),
        where=mag_sums > 0,
    )
    spectral_centroid = float(np.mean(centroids))
    centroid_std = float(np.std(centroids))
    bandwidths = np.divide(
        np.sqrt(np.sum(magnitudes * (freqs[None, :] - centroids[:, None]) ** 2, axis=1)),
        np.sqrt(mag_sums),
        out=np.zeros_like(mag_sums, dtype=float),
        where=mag_sums > 0,
    )
    spectral_bandwidth = float(np.mean(bandwidths))

    cumulative = np.cumsum(magnitudes, axis=1)
    rolloff_threshold = 0.85 * cumulative[:, -1]
    rolloff_bins = np.argmax(cumulative >= rolloff_threshold[:, None], axis=1)
    spectral_rolloff = float(np.mean(freqs[np.clip(rolloff_bins, 0, len(freqs) - 1)]))

    features = {
        "duration_sec": float(duration),
        "sample_rate_hz": int(sample_rate),
        "num_samples": int(len(audio)),
        "rms_energy": rms_energy,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_centroid_std_hz": centroid_std,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "spectral_rolloff_hz": spectral_rolloff,
    }
    return features, frames


def _classify_voice(features: Dict[str, Any]) -> Tuple[str, float, List[str]]:
    duration = features["duration_sec"]
    rms = features["rms_energy"]
    zcr = features["zero_crossing_rate"]
    centroid = features["spectral_centroid_hz"]
    centroid_std = features["spectral_centroid_std_hz"]
    bandwidth = features["spectral_bandwidth_hz"]

    score = 0.0
    feedback: List[str] = []

    if duration < 1.0:
        score += 0.35
        feedback.append("Recording is very short, so the voice signal is less stable for analysis.")
    else:
        feedback.append(f"Recording duration is {duration:.2f} seconds, enough for a compact signal summary.")

    if rms < 0.01:
        score += 0.35
        feedback.append("RMS energy is very low, which suggests silence or a very quiet microphone input.")
    elif rms > 0.35:
        score += 0.25
        feedback.append("RMS energy is high, which may indicate clipping or very close microphone input.")
    else:
        feedback.append("RMS energy is within a usable voice capture range.")

    if zcr > 0.25:
        score += 0.20
        feedback.append("Zero crossing rate is high, which can occur with noisy or unstable high frequency content.")
    elif zcr < 0.005 and rms > 0.02:
        score += 0.15
        feedback.append("Zero crossing rate is unusually low for a spoken signal.")
    else:
        feedback.append("Zero crossing rate is consistent with a typical voiced recording.")

    if centroid < 80.0 and rms > 0.01:
        score += 0.15
        feedback.append("Spectral centroid is very low, showing limited high frequency speech detail.")
    elif centroid > 4500.0:
        score += 0.20
        feedback.append("Spectral centroid is high, indicating strong high frequency content or background noise.")
    else:
        feedback.append("Spectral centroid shows balanced speech frequency content.")

    if centroid_std > 2500.0:
        score += 0.15
        feedback.append("Spectral centroid varies strongly over time, so the signal is less steady.")
    else:
        feedback.append("Spectral characteristics are reasonably stable across the recording.")

    if bandwidth > 3500.0:
        score += 0.10
        feedback.append("Spectral bandwidth is broad, which can indicate noisy or highly varied audio content.")
    else:
        feedback.append("Spectral bandwidth is within a compact range for voice signal review.")

    if score >= 0.60:
        status = "ANOMALOUS"
    elif score >= 0.30:
        status = "SUSPICIOUS"
    else:
        status = "NORMAL"

    return status, float(min(score, 1.0)), feedback


def _waveform_preview(audio: np.ndarray, sample_rate: int, max_points: int = 3000) -> Dict[str, List[float]]:
    if len(audio) == 0:
        return {"time_sec": [], "amplitude": []}

    step = max(1, len(audio) // max_points)
    preview = audio[::step]
    times = np.arange(0, len(audio), step)[: len(preview)] / float(sample_rate)
    return {
        "time_sec": [float(v) for v in times],
        "amplitude": [float(v) for v in preview],
    }


def analyze_voice_recording(audio_bytes: bytes, filename: str | None = None) -> Dict[str, Any]:
    """Extract voice signal features and a technical status from actual audio bytes."""
    sample_rate, mono_audio = _load_audio(audio_bytes, filename)

    features, _ = _basic_features(mono_audio, int(sample_rate))
    features["mfcc_mean"] = _mfcc(mono_audio, int(sample_rate))

    status, score, feedback = _classify_voice(features)
    return {
        "status": status,
        "anomaly_score": score,
        "features": features,
        "feedback": feedback,
        "waveform": _waveform_preview(mono_audio, int(sample_rate)),
        "method": "Rule-based technical signal quality scoring from actual microphone audio features.",
    }


def inspect_audio_recording(audio_bytes: bytes, filename: str | None = None) -> Dict[str, Any]:
    """Decode actual audio bytes for upload preview without running voice scoring."""
    sample_rate, mono_audio = _load_audio(audio_bytes, filename)
    duration = len(mono_audio) / float(sample_rate) if sample_rate > 0 else 0.0
    return {
        "duration_sec": float(duration),
        "sample_rate_hz": int(sample_rate),
        "num_samples": int(len(mono_audio)),
        "waveform": _waveform_preview(mono_audio, int(sample_rate)),
    }
