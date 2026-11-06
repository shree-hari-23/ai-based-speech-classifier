"""
Acoustic Feature Extraction Module for Speech Emotion Recognition.
Extracts 180-dimensional tabular features (40 MFCC + 12 Chroma + 128 Mel) for SVM & MLP,
and fixed-length sequential time-frequency matrices (130 frames x 180 features) for 1D CNN.
"""

import sys
import os
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any

import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SAMPLE_RATE,
    DURATION,
    N_MFCC,
    N_CHROMA,
    N_MELS,
    FEATURE_DIM,
    HOP_LENGTH,
    N_FFT,
    MODELS_DIR,
    LABEL_ENCODER_PATH,
    SUPPORTED_EMOTIONS,
)
from training.preprocessing import preprocess_audio

logger = logging.getLogger(__name__)


def extract_features_from_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts acoustic features from preprocessed audio waveform y:
      1. 40 MFCCs (Mel-Frequency Cepstral Coefficients)
      2. 12 Chroma STFT features (pitch/harmonic content)
      3. 128 Mel Spectrogram bands (log-power scale)

    Returns:
      - ml_features: 1D array of shape (180,) with mean-pooled features across time.
      - cnn_sequence: 2D array of shape (time_frames, 180) representing temporal sequence.
    """
    # 1. 40 MFCCs -> shape: (40, time_frames)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # 2. 12 Chroma STFT -> shape: (12, time_frames)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_chroma=N_CHROMA, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # 3. 128 Mel Spectrogram (log-power / dB) -> shape: (128, time_frames)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # --- Traditional ML Feature Vector (Mean-pooled over time) ---
    mfcc_mean = np.mean(mfcc, axis=1)        # 40
    chroma_mean = np.mean(chroma, axis=1)    # 12
    mel_mean = np.mean(mel_db, axis=1)       # 128

    ml_features = np.hstack([mfcc_mean, chroma_mean, mel_mean]).astype(np.float32)
    assert len(ml_features) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(ml_features)}"

    # --- Deep Learning CNN Sequential Representation ---
    # Stack along feature axis: shape (time_frames, 180)
    # Transpose so time is axis 0 (time_steps) and features is axis 1 (channels)
    cnn_sequence = np.hstack([mfcc.T, chroma.T, mel_db.T]).astype(np.float32)

    return ml_features, cnn_sequence


def extract_features_from_file(
    file_path: Union[str, Path, bytes],
    sr: int = SAMPLE_RATE,
    duration: float = DURATION
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads, preprocesses, and extracts both ML vector and CNN sequence features
    from an audio file path or byte buffer.
    """
    y = preprocess_audio(file_path, target_sr=sr, duration=duration)
    return extract_features_from_audio(y, sr=sr)


def get_feature_names() -> list:
    """Returns descriptive names for the 180 tabular features."""
    names = []
    names.extend([f"mfcc_{i+1}" for i in range(N_MFCC)])
    names.extend([f"chroma_{i+1}" for i in range(N_CHROMA)])
    names.extend([f"mel_{i+1}" for i in range(N_MELS)])
    return names


def prepare_dataset_features(
    train_df,
    test_df,
    save_cache: bool = True,
    cache_dir: Path = MODELS_DIR
) -> Dict[str, Any]:
    """
    Extracts features for all samples in train and test DataFrames,
    fits and saves the LabelEncoder, and caches feature matrices to disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "features_cache.npz"

    # 1. Label Encoder
    encoder = LabelEncoder()
    # Fit encoder on all supported emotions to ensure consistent index mapping (0 to 7)
    encoder.fit(SUPPORTED_EMOTIONS)
    joblib.dump(encoder, LABEL_ENCODER_PATH)
    logger.info(f"LabelEncoder saved to {LABEL_ENCODER_PATH}. Classes: {list(encoder.classes_)}")

    # 2. Extract Training Features
    logger.info(f"Extracting acoustic features from {len(train_df)} training recordings...")
    X_train_ml_list = []
    X_train_cnn_list = []
    y_train_list = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train Features"):
        try:
            ml_feat, cnn_feat = extract_features_from_file(row["file_path"])
            X_train_ml_list.append(ml_feat)
            X_train_cnn_list.append(cnn_feat)
            y_train_list.append(row["emotion"])
        except Exception as e:
            logger.warning(f"Error extracting features from {row['file_path']}: {e}")

    # 3. Extract Test Features
    logger.info(f"Extracting acoustic features from {len(test_df)} test recordings...")
    X_test_ml_list = []
    X_test_cnn_list = []
    y_test_list = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test Features"):
        try:
            ml_feat, cnn_feat = extract_features_from_file(row["file_path"])
            X_test_ml_list.append(ml_feat)
            X_test_cnn_list.append(cnn_feat)
            y_test_list.append(row["emotion"])
        except Exception as e:
            logger.warning(f"Error extracting features from {row['file_path']}: {e}")

    # Convert to numpy arrays
    X_train_ml = np.array(X_train_ml_list, dtype=np.float32)
    X_train_cnn = np.array(X_train_cnn_list, dtype=np.float32)
    y_train = encoder.transform(y_train_list)

    X_test_ml = np.array(X_test_ml_list, dtype=np.float32)
    X_test_cnn = np.array(X_test_cnn_list, dtype=np.float32)
    y_test = encoder.transform(y_test_list)

    logger.info("=" * 60)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info(f"X_train_ml shape:  {X_train_ml.shape}  | y_train: {y_train.shape}")
    logger.info(f"X_test_ml shape:   {X_test_ml.shape}   | y_test:  {y_test.shape}")
    logger.info(f"X_train_cnn shape: {X_train_cnn.shape}")
    logger.info(f"X_test_cnn shape:  {X_test_cnn.shape}")
    logger.info("=" * 60)

    if save_cache:
        logger.info(f"Saving extracted features cache to {cache_file}...")
        np.savez_compressed(
            cache_file,
            X_train_ml=X_train_ml,
            X_test_ml=X_test_ml,
            X_train_cnn=X_train_cnn,
            X_test_cnn=X_test_cnn,
            y_train=y_train,
            y_test=y_test
        )
        logger.info("[SUCCESS] Features successfully cached to disk.")

    return {
        "X_train_ml": X_train_ml,
        "X_test_ml": X_test_ml,
        "X_train_cnn": X_train_cnn,
        "X_test_cnn": X_test_cnn,
        "y_train": y_train,
        "y_test": y_test,
        "encoder": encoder
    }


def load_cached_features(cache_dir: Path = MODELS_DIR) -> Optional[Dict[str, Any]]:
    """Loads previously extracted feature matrices from cache if available."""
    cache_file = cache_dir / "features_cache.npz"
    if not cache_file.exists():
        return None

    try:
        data = np.load(cache_file)
        encoder = joblib.load(LABEL_ENCODER_PATH) if LABEL_ENCODER_PATH.exists() else None
        return {
            "X_train_ml": data["X_train_ml"],
            "X_test_ml": data["X_test_ml"],
            "X_train_cnn": data["X_train_cnn"],
            "X_test_cnn": data["X_test_cnn"],
            "y_train": data["y_train"],
            "y_test": data["y_test"],
            "encoder": encoder
        }
    except Exception as e:
        logger.warning(f"Failed to load cached features: {e}")
        return None


if __name__ == "__main__":
    from dataset_loader import load_combined_dataset, speaker_independent_train_test_split

    print("Running Feature Extraction pipeline test...")
    df = load_combined_dataset()
    if not df.empty:
        # Test on 2 sample files first
        sample_path = df.iloc[0]["file_path"]
        print(f"\nTesting feature extraction on sample file: {Path(sample_path).name}")
        ml_f, cnn_f = extract_features_from_file(sample_path)
        print(f"ML Features shape:  {ml_f.shape} (Expected: (180,))")
        print(f"CNN Sequence shape: {cnn_f.shape} (Expected: (130, 180))")
        print(f"Feature summary: Min={ml_f.min():.3f}, Max={ml_f.max():.3f}, Mean={ml_f.mean():.3f}")

        # Split and extract dataset features
        train_df, test_df = speaker_independent_train_test_split(df)
        data = prepare_dataset_features(train_df, test_df, save_cache=True)
        print("\nPipeline feature extraction test completed successfully!")
    else:
        print("[ERROR] Dataset empty. Please download datasets first.")
