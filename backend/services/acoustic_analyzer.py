"""
Acoustic Analyzer Service for Speech Emotion Recognition.
Extracts psychoacoustic indicators, audio quality diagnostics (SNR, noise floor,
clipping, volume), and multi-segment chronological emotion timelines.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import librosa

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE

logger = logging.getLogger("SER_AcousticAnalyzer")


def compute_acoustic_profile(y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """
    Computes comprehensive acoustic characteristics and audio recording quality indicators:
    - Pitch (F0) dynamics in Hz
    - Energy (RMS) and dynamic range
    - Spectral Centroid (timbral brightness)
    - Zero Crossing Rate (consonant articulation / percussiveness)
    - Signal-to-Noise Ratio (SNR) estimation
    - Clipping distortion ratio
    - Volume level classification
    """
    if len(y) == 0:
        return {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "rms_energy": 0.0,
            "spectral_centroid_hz": 0.0,
            "zero_crossing_rate": 0.0,
            "snr_db": 0.0,
            "clipping_ratio": 0.0,
            "volume_level": "Silent",
            "overall_quality": "Degraded",
            "quality_issues": ["Empty audio signal"]
        }

    # 1. Pitch / Fundamental Frequency (F0) using librosa.yin
    pitch_mean = 0.0
    pitch_std = 0.0
    pitch_min = 0.0
    pitch_max = 0.0
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        valid_f0 = f0[(f0 > 50) & (f0 < 490)]
        if len(valid_f0) > 0:
            pitch_mean = round(float(np.mean(valid_f0)), 1)
            pitch_std = round(float(np.std(valid_f0)), 1)
            pitch_min = round(float(np.min(valid_f0)), 1)
            pitch_max = round(float(np.max(valid_f0)), 1)
    except Exception as e:
        logger.debug(f"Pitch extraction notice: {e}")

    # 2. RMS Energy & Dynamics
    rms_frames = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_max = float(np.max(rms_frames)) if len(rms_frames) > 0 else 0.0

    # Convert RMS to dBFS
    ref_val = 1.0
    rms_db = round(float(20 * np.log10(max(rms_mean, 1e-6) / ref_val)), 1)

    # 3. Spectral Centroid (Brightness in Hz)
    sc_frames = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sc_mean = round(float(np.mean(sc_frames)), 1) if len(sc_frames) > 0 else 0.0

    # 4. Zero Crossing Rate (Articulation / Noisiness)
    zcr_frames = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = round(float(np.mean(zcr_frames)), 4) if len(zcr_frames) > 0 else 0.0

    # 5. Audio Quality Diagnostics
    # Peak amplitude & clipping detection
    peak_abs = float(np.max(np.abs(y)))
    clipping_samples = np.sum(np.abs(y) >= 0.98)
    clipping_ratio = round(float(clipping_samples / len(y)) * 100, 2)

    # Background Noise & SNR estimation
    # Noise floor estimated from the lowest 15% energy frames
    sorted_rms = np.sort(rms_frames) if len(rms_frames) > 0 else np.array([0.001])
    noise_count = max(1, int(len(sorted_rms) * 0.15))
    noise_floor = float(np.mean(sorted_rms[:noise_count]))
    signal_top = float(np.mean(sorted_rms[-noise_count:])) if len(sorted_rms) >= noise_count else rms_mean

    snr_db = round(20 * np.log10(max(signal_top, 1e-5) / max(noise_floor, 1e-6)), 1)
    snr_db = max(0.0, min(60.0, snr_db))

    # Quality Issues Detection
    quality_issues = []
    if clipping_ratio > 0.5:
        quality_issues.append(f"Audio clipping detected ({clipping_ratio}% of samples exceed dynamic threshold)")
    if rms_db < -30.0:
        quality_issues.append("Low recording volume detected (below -30 dBFS)")
    if snr_db < 10.0:
        quality_issues.append(f"High background noise detected (SNR: {snr_db} dB)")
    elif snr_db < 16.0:
        quality_issues.append(f"Moderate background noise present (SNR: {snr_db} dB)")

    # Volume classification
    if rms_db < -32.0:
        volume_level = "Low Volume"
    elif rms_db > -6.0:
        volume_level = "High / Near Peak"
    else:
        volume_level = "Optimal Volume"

    # Overall Quality
    if len(quality_issues) == 0:
        overall_quality = "High Quality"
    elif len(quality_issues) == 1 and "Moderate" in quality_issues[0]:
        overall_quality = "Acceptable"
    else:
        overall_quality = "Degraded"

    return {
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "pitch_min_hz": pitch_min,
        "pitch_max_hz": pitch_max,
        "rms_energy": round(rms_mean, 5),
        "rms_db": rms_db,
        "spectral_centroid_hz": sc_mean,
        "zero_crossing_rate": zcr_mean,
        "snr_db": snr_db,
        "clipping_ratio": clipping_ratio,
        "volume_level": volume_level,
        "overall_quality": overall_quality,
        "quality_issues": quality_issues
    }


def compute_emotion_timeline(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    engine: Any = None,
    num_segments: int = 3
) -> List[Dict[str, Any]]:
    """
    Divides the recording into chronological segments (Initial, Middle, Final)
    and computes the emotional trajectory and acoustic intensity across time.
    """
    total_len = len(y)
    total_duration = total_len / sr
    if total_len == 0 or total_duration < 0.6:
        return []

    num_segments = min(num_segments, max(2, int(total_duration)))
    seg_len = total_len // num_segments
    timeline = []

    names = ["Initial Segment", "Middle Segment", "Final Segment", "Closing Segment"]

    for i in range(num_segments):
        start_idx = i * seg_len
        end_idx = total_len if i == num_segments - 1 else (i + 1) * seg_len
        seg_y = y[start_idx:end_idx]

        start_time = round(start_idx / sr, 1)
        end_time = round(end_idx / sr, 1)
        seg_name = names[i] if i < len(names) else f"Segment {i+1}"

        # Segment Energy & Pitch
        seg_rms = float(np.mean(librosa.feature.rms(y=seg_y)[0])) if len(seg_y) > 0 else 0.0
        seg_energy = "high" if seg_rms > 0.03 else "moderate" if seg_rms > 0.01 else "low"

        seg_pitch = 0.0
        try:
            f0 = librosa.yin(seg_y, fmin=60, fmax=450, sr=sr)
            valid = f0[(f0 > 60) & (f0 < 440)]
            if len(valid) > 0:
                seg_pitch = round(float(np.mean(valid)), 1)
        except Exception:
            pass

        # Predict segment emotion if engine is provided
        pred_emotion = "neutral"
        confidence = 70.0
        if engine is not None and len(seg_y) >= int(sr * 0.4):
            try:
                # Preprocess segment to standard length
                from training.feature_extraction import extract_features_from_audio
                from training.preprocessing import pad_or_truncate_audio
                padded = pad_or_truncate_audio(seg_y)
                ml_feat, _ = extract_features_from_audio(padded, sr=sr)
                if engine.mlp_pipeline is not None:
                    pred_emotion, confidence, _ = engine._predict_mlp(ml_feat)
                elif engine.svm_pipeline is not None:
                    pred_emotion, confidence, _ = engine._predict_svm(ml_feat)
            except Exception as e:
                logger.debug(f"Segment prediction notice: {e}")

        timeline.append({
            "segment_index": i + 1,
            "name": f"{seg_name} ({start_time}s - {end_time}s)",
            "time_range": f"{start_time}s - {end_time}s",
            "start_time": start_time,
            "end_time": end_time,
            "emotion": pred_emotion,
            "confidence": confidence,
            "energy_level": seg_energy,
            "pitch_hz": seg_pitch
        })

    return timeline
