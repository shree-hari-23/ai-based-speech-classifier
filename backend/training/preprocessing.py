"""
Audio Preprocessing Pipeline for Speech Emotion Recognition.
Handles loading, resampling, stereo-to-mono conversion, silence trimming,
amplitude normalization, and uniform duration padding/truncation.
"""

import io
import sys
import logging
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
import librosa
import soundfile as sf

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE, DURATION, TARGET_LENGTH

logger = logging.getLogger(__name__)


def load_audio(
    audio_source: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """
    Loads audio from a file path, raw bytes, or BytesIO buffer.
    Converts stereo to mono and resamples to target_sr.
    """
    try:
        if isinstance(audio_source, (bytes, bytearray)):
            audio_source = io.BytesIO(audio_source)

        if isinstance(audio_source, io.BytesIO):
            # Load from buffer using soundfile, fallback to librosa
            audio_source.seek(0)
            data, sr = sf.read(audio_source, dtype="float32")
            if data.ndim > 1:
                # Convert stereo to mono by averaging channels
                data = np.mean(data, axis=1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            return data.astype(np.float32), target_sr
        else:
            # File path
            path_str = str(audio_source)
            data, sr = librosa.load(path_str, sr=target_sr, mono=True)
            return data.astype(np.float32), target_sr

    except Exception as e:
        logger.error(f"Error loading audio source {audio_source}: {e}")
        raise ValueError(f"Failed to load audio: {e}")


def normalize_amplitude(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalizes audio waveform amplitude to a consistent peak level.
    Prevents clipping and ensures uniform signal loudness.
    """
    peak = np.max(np.abs(y))
    if peak > 1e-7:
        y = (y / peak) * target_peak
    return y.astype(np.float32)


def apply_preemphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Applies a pre-emphasis high-pass filter to amplify high frequencies,
    balancing the speech spectrum for spectral feature extraction.
    """
    if len(y) == 0:
        return y
    return np.append(y[0], y[1:] - coeff * y[:-1]).astype(np.float32)


def trim_silence(y: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    """
    Trims leading and trailing silence from the recording.
    Keeps the core expressive speech segments.
    """
    if len(y) == 0:
        return y
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    # If trimming removed too much (<0.2s remaining), revert to untrimmed
    if len(y_trimmed) < int(SAMPLE_RATE * 0.2):
        return y
    return y_trimmed.astype(np.float32)


def enforce_fixed_length(
    y: np.ndarray,
    target_length: int = TARGET_LENGTH,
    pad_mode: str = "center"
) -> np.ndarray:
    """
    Ensures the audio recording matches target_length exactly.
    - If shorter: center-pads with zeros.
    - If longer: center-crops to retain the central emotional utterance.
    """
    n_samples = len(y)
    if n_samples == target_length:
        return y

    if n_samples < target_length:
        # Pad with zeros
        pad_total = target_length - n_samples
        if pad_mode == "center":
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            y_fixed = np.pad(y, (pad_left, pad_right), mode="constant")
        else:
            y_fixed = np.pad(y, (0, pad_total), mode="constant")
    else:
        # Crop central window
        excess = n_samples - target_length
        start = excess // 2
        y_fixed = y[start : start + target_length]

    return y_fixed.astype(np.float32)


def preprocess_audio(
    audio_source: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    trim_silence_flag: bool = True,
    apply_preemphasis_flag: bool = True
) -> np.ndarray:
    """
    End-to-end audio preprocessing pipeline:
    1. Load audio & convert stereo to mono.
    2. Resample to standard sample rate (22,050 Hz).
    3. Trim leading/trailing background silence.
    4. Normalize audio amplitude.
    5. Apply pre-emphasis filter.
    6. Pad/crop to exact target length (target_sr * duration).

    Returns 1D numpy array of length (target_sr * duration).
    """
    target_length = int(target_sr * duration)

    # 1 & 2. Load and resample
    y, sr = load_audio(audio_source, target_sr=target_sr)

    # 3. Trim silence
    if trim_silence_flag:
        y = trim_silence(y, top_db=28.0)

    # 4. Normalize amplitude
    y = normalize_amplitude(y, target_peak=0.95)

    # 5. Pre-emphasis
    if apply_preemphasis_flag:
        y = apply_preemphasis(y, coeff=0.97)

    # 6. Enforce fixed duration (3.0s = 66,150 samples)
    y = enforce_fixed_length(y, target_length=target_length, pad_mode="center")

    return y

