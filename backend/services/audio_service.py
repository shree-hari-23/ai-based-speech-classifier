"""
Audio Service for Speech Emotion Recognition.
Handles file validation, temporary storage, audio integrity checks,
waveform extraction for UI visualization, and automatic cleanup.
"""

import os
import sys
import uuid
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import soundfile as sf
import librosa
from fastapi import UploadFile, HTTPException

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import UPLOADS_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Allowed audio extensions and MIME types
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
MIN_DURATION = 0.2  # seconds
MAX_DURATION = 60.0  # seconds


def validate_audio_upload(file: UploadFile) -> str:
    """
    Validates uploaded file extension and content type.
    Returns cleaned file extension.
    """
    filename = file.filename or "unknown.wav"
    suffix = Path(filename).suffix.lower()

    if not suffix or suffix not in ALLOWED_EXTENSIONS:
        # Check content type if extension is missing/generic
        content_type = (file.content_type or "").lower()
        if "audio" not in content_type and "video/webm" not in content_type:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{suffix}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        suffix = ".wav" if "wav" in content_type else ".webm"

    return suffix


async def save_temp_upload(file: UploadFile) -> Tuple[Path, int]:
    """
    Safely writes an uploaded file to the temporary uploads directory.
    Validates file size during streaming write.
    Returns (Path to saved file, total bytes written).
    """
    suffix = validate_audio_upload(file)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_filename = f"upload_{uuid.uuid4().hex}{suffix}"
    temp_path = UPLOADS_DIR / temp_filename

    total_size = 0
    with open(temp_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                # Remove oversized file
                buffer.close()
                if temp_path.exists():
                    temp_path.unlink()
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)} MB."
                )
            buffer.write(chunk)

    if total_size == 0:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    return temp_path, total_size


def cleanup_file(file_path: Optional[Path]):
    """Safely removes a temporary file."""
    if file_path and file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove temp file {file_path}: {e}")


def extract_waveform_points(y: np.ndarray, num_points: int = 120) -> List[float]:
    """
    Downsamples audio waveform into num_points peak/RMS amplitudes
    scaled between 0.05 and 1.0 for rendering in frontend Waveform charts.
    """
    if len(y) == 0:
        return [0.1] * num_points

    # Split into chunks
    chunk_size = max(1, len(y) // num_points)
    points = []

    for i in range(num_points):
        chunk = y[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) > 0:
            # RMS amplitude of chunk
            rms = float(np.sqrt(np.mean(chunk**2)))
            points.append(rms)
        else:
            points.append(0.0)

    # Normalize to [0.05, 1.0]
    max_val = max(points) if points and max(points) > 1e-6 else 1.0
    normalized = [round(max(0.05, p / max_val), 3) for p in points]
    return normalized


def extract_audio_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extracts duration, sample rate, channels, and validates duration boundaries.
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        duration = float(len(y) / sr)

        if duration < MIN_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration ({duration:.2f}s) is too short. Minimum required is {MIN_DURATION}s."
            )

        waveform_points = extract_waveform_points(y, num_points=120)

        return {
            "duration": round(duration, 2),
            "sample_rate": sr,
            "samples": len(y),
            "waveform_points": waveform_points
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inspecting audio metadata for {file_path}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Corrupted or unreadable audio format: {str(e)}"
        )

