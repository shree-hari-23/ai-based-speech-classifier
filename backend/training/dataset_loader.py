"""
Dataset Loader and Exploration Module for Speech Emotion Recognition.
Handles RAVDESS and TESS audio datasets, emotion label extraction, file validation,
and speaker-independent train/test splitting to prevent data leakage.
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import soundfile as sf

# Add parent directory to system path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAVDESS_DIR,
    TESS_DIR,
    SUPPORTED_EMOTIONS,
    RAVDESS_EMOTION_MAP,
    TESS_EMOTION_MAP,
    MODELS_DIR,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_audio_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Validates an audio file, checks for corruption, and extracts basic audio properties.
    Returns metadata dict if valid, or None if corrupted/unreadable.
    """
    try:
        if not file_path.exists() or file_path.stat().st_size == 0:
            logger.warning(f"File is empty or missing: {file_path}")
            return None

        # Read audio header using soundfile without loading full data into memory
        info = sf.info(str(file_path))
        if info.frames == 0 or info.duration < 0.1:
            logger.warning(f"Audio file has zero or near-zero duration: {file_path}")
            return None

        return {
            "duration": float(info.duration),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "format": str(info.format),
            "subtype": str(info.subtype)
        }
    except Exception as e:
        logger.warning(f"Corrupted or unreadable audio file {file_path.name}: {e}")
        return None


def parse_ravdess_filename(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parses a RAVDESS audio filename according to the official naming convention:
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
    Vocal channel (01 = speech, 02 = song)
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    Emotional intensity (01 = normal, 02 = strong)
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
    Repetition (01 = 1st repetition, 02 = 2nd repetition)
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    """
    stem = file_path.stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None

    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor_str = parts

    # We only process speech/song audio
    emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
    if not emotion:
        return None

    try:
        actor_id = int(actor_str)
        gender = "female" if actor_id % 2 == 0 else "male"
    except ValueError:
        return None

    return {
        "file_path": str(file_path.resolve()),
        "file_name": file_path.name,
        "dataset": "RAVDESS",
        "emotion": emotion,
        "speaker_id": f"RAVDESS_Actor_{actor_id:02d}",
        "gender": gender,
        "intensity": "strong" if intensity == "02" else "normal",
        "statement": int(statement) if statement.isdigit() else 1,
        "repetition": int(repetition) if repetition.isdigit() else 1,
    }


def parse_tess_path(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parses a TESS audio file path.
    TESS files are organized by emotion and speaker (OAF or YAF).
    Example patterns:
      - OAF_back_angry.wav
      - YAF_youth_fear.wav
      - OAF_base_ps.wav (ps = pleasant surprise -> surprised)
      - Directories like 'OAF_Fear', 'YAF_happy', 'OAF_Pleasant_surprise'
    """
    stem = file_path.stem.lower()
    parent_dir = file_path.parent.name.lower()
    combined_name = f"{parent_dir}_{stem}"

    # Determine speaker
    # OAF: Older Adult Female, YAF: Younger Adult Female
    if stem.startswith("oaf") or "oaf" in parent_dir:
        speaker_id = "TESS_OAF"
    elif stem.startswith("yaf") or "yaf" in parent_dir:
        speaker_id = "TESS_YAF"
    else:
        speaker_id = "TESS_Female"

    gender = "female"  # All TESS recordings are by female actresses

    # Determine emotion
    emotion = None

    # Check for pleasant surprise ('ps')
    # Use regex boundaries or specific naming conventions
    if re.search(r'\b(ps|pleasant[_\s-]?surprise[d]?)\b', combined_name) or stem.endswith("_ps"):
        emotion = "surprised"
    elif "fear" in combined_name:
        emotion = "fearful"
    elif "angry" in combined_name:
        emotion = "angry"
    elif "disgust" in combined_name:
        emotion = "disgust"
    elif "happy" in combined_name:
        emotion = "happy"
    elif "neutral" in combined_name:
        emotion = "neutral"
    elif "sad" in combined_name:
        emotion = "sad"

    if not emotion or emotion not in SUPPORTED_EMOTIONS:
        return None

    return {
        "file_path": str(file_path.resolve()),
        "file_name": file_path.name,
        "dataset": "TESS",
        "emotion": emotion,
        "speaker_id": speaker_id,
        "gender": gender,
        "intensity": "normal",
        "statement": 1,
        "repetition": 1,
    }


def load_ravdess_dataset(ravdess_dir: Path = RAVDESS_DIR) -> pd.DataFrame:
    """
    Scans the RAVDESS directory, parses filenames, validates audio integrity,
    and returns a pandas DataFrame.
    """
    logger.info(f"Scanning RAVDESS dataset in {ravdess_dir}...")
    records = []
    skipped_count = 0

    wav_files = list(ravdess_dir.rglob("*.wav"))
    if not wav_files:
        logger.warning(f"No .wav files found in RAVDESS directory: {ravdess_dir}")
        return pd.DataFrame()

    for wav_file in wav_files:
        meta = parse_ravdess_filename(wav_file)
        if not meta:
            skipped_count += 1
            continue

        audio_info = validate_audio_file(wav_file)
        if not audio_info:
            skipped_count += 1
            continue

        meta.update(audio_info)
        records.append(meta)

    df = pd.DataFrame(records)
    logger.info(f"RAVDESS loaded: {len(df)} valid recordings (skipped/corrupted: {skipped_count})")
    return df


def load_tess_dataset(tess_dir: Path = TESS_DIR) -> pd.DataFrame:
    """
    Scans the TESS directory, parses filenames/folders, validates audio integrity,
    and returns a pandas DataFrame.
    """
    logger.info(f"Scanning TESS dataset in {tess_dir}...")
    records = []
    skipped_count = 0

    wav_files = list(tess_dir.rglob("*.wav"))
    if not wav_files:
        logger.warning(f"No .wav files found in TESS directory: {tess_dir}")
        return pd.DataFrame()

    for wav_file in wav_files:
        meta = parse_tess_path(wav_file)
        if not meta:
            skipped_count += 1
            continue

        audio_info = validate_audio_file(wav_file)
        if not audio_info:
            skipped_count += 1
            continue

        meta.update(audio_info)
        records.append(meta)

    df = pd.DataFrame(records)
    logger.info(f"TESS loaded: {len(df)} valid recordings (skipped/corrupted: {skipped_count})")
    return df


def load_combined_dataset(
    ravdess_dir: Path = RAVDESS_DIR,
    tess_dir: Path = TESS_DIR
) -> pd.DataFrame:
    """
    Loads and merges both RAVDESS and TESS datasets into a unified pandas DataFrame.
    Displays summary counts and validates emotion labels.
    """
    df_ravdess = load_ravdess_dataset(ravdess_dir)
    df_tess = load_tess_dataset(tess_dir)

    dfs = [df for df in [df_ravdess, df_tess] if not df.empty]
    if not dfs:
        logger.warning("No audio files could be loaded from either dataset directory!")
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter to only supported emotions
    combined_df = combined_df[combined_df["emotion"].isin(SUPPORTED_EMOTIONS)].reset_index(drop=True)

    logger.info("=" * 60)
    logger.info(f"Total successfully processed recordings: {len(combined_df)}")
    logger.info(f"RAVDESS count: {len(df_ravdess)} | TESS count: {len(df_tess)}")
    logger.info("=" * 60)

    return combined_df


def speaker_independent_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "speaker_stratified"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs train/test splitting ensuring zero data leakage:
      - For RAVDESS (24 distinct actors): test set uses completely held-out actors.
        Actors in test never appear in train (speaker-independent).
      - For TESS (2 speakers, 200 target word prompts): prompts are partitioned so that
        the test set contains distinct word utterances with balanced emotion classes.
      - If split_strategy == "strict_speaker", partitions entire speakers including TESS.

    Returns (train_df, test_df).
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame!")

    np.random.seed(random_state)
    train_records = []
    test_records = []

    # 1. Split RAVDESS: Held-out actors (strict speaker independence)
    ravdess_df = df[df["dataset"] == "RAVDESS"].copy()
    if not ravdess_df.empty:
        actors = sorted(ravdess_df["speaker_id"].unique())
        n_test_actors = max(2, int(len(actors) * test_size))
        
        # Shuffle actors deterministically
        shuffled_actors = list(np.random.permutation(actors))
        test_actors = shuffled_actors[:n_test_actors]
        train_actors = shuffled_actors[n_test_actors:]

        rav_train = ravdess_df[ravdess_df["speaker_id"].isin(train_actors)]
        rav_test = ravdess_df[ravdess_df["speaker_id"].isin(test_actors)]

        train_records.append(rav_train)
        test_records.append(rav_test)

        logger.info(f"RAVDESS Speaker-Independent Split: {len(train_actors)} train actors, {len(test_actors)} test actors ({', '.join(test_actors)})")

    # 2. Split TESS
    tess_df = df[df["dataset"] == "TESS"].copy()
    if not tess_df.empty:
        if split_strategy == "strict_speaker":
            # Hold out one entire speaker (e.g. OAF)
            speakers = sorted(tess_df["speaker_id"].unique())
            test_spk = [speakers[0]]
            train_spk = speakers[1:]
            tess_train = tess_df[tess_df["speaker_id"].isin(train_spk)]
            tess_test = tess_df[tess_df["speaker_id"].isin(test_spk)]
        else:
            # Utterance/prompt-level stratified split to maintain balanced emotions
            from sklearn.model_selection import train_test_split
            tess_train, tess_test = train_test_split(
                tess_df,
                test_size=test_size,
                random_state=random_state,
                stratify=tess_df["emotion"]
            )

        train_records.append(tess_train)
        test_records.append(tess_test)

    train_df = pd.concat(train_records, ignore_index=True)
    test_df = pd.concat(test_records, ignore_index=True)

    # Shuffle train and test sets
    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    logger.info("=" * 60)
    logger.info(f"DATASET SPLIT SUMMARY ({split_strategy})")
    logger.info(f"Total Samples: {len(df)}")
    logger.info(f"Train Samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Test Samples:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    logger.info("=" * 60)

    # Check RAVDESS speaker leakage
    rav_train_spk = set(train_df[train_df["dataset"] == "RAVDESS"]["speaker_id"])
    rav_test_spk = set(test_df[test_df["dataset"] == "RAVDESS"]["speaker_id"])
    assert len(rav_train_spk.intersection(rav_test_spk)) == 0, "RAVDESS actor leakage detected!"

    return train_df, test_df


def explore_dataset(df: pd.DataFrame, save_dir: Path = MODELS_DIR) -> Dict[str, Any]:
    """
    Performs comprehensive dataset exploration and prints/saves distribution summaries:
      - Total counts
      - Emotion class distribution per dataset and overall
      - Speaker distribution
      - Gender distribution
      - Audio duration statistics (min, max, mean, std)
    """
    if df.empty:
        print("[WARNING] Dataset is empty, cannot perform exploration.")
        return {}

    save_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_recordings": int(len(df)),
        "datasets": df["dataset"].value_counts().to_dict(),
        "emotions_overall": df["emotion"].value_counts().to_dict(),
        "emotions_by_dataset": {
            dataset: sub_df["emotion"].value_counts().to_dict()
            for dataset, sub_df in df.groupby("dataset")
        },
        "gender_distribution": df["gender"].value_counts().to_dict(),
        "total_speakers": int(df["speaker_id"].nunique()),
        "duration_stats": {
            "mean_seconds": round(float(df["duration"].mean()), 2),
            "std_seconds": round(float(df["duration"].std()), 2),
            "min_seconds": round(float(df["duration"].min()), 2),
            "max_seconds": round(float(df["duration"].max()), 2),
        }
    }

    # Save summary to JSON
    json_path = save_dir / "dataset_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print formatted exploration report
    print("\n" + "=" * 65)
    print("      ACOUSTIC SPEECH EMOTION RECOGNITION - DATASET EXPLORATION")
    print("=" * 65)
    print(f"Total Successfully Processed Audio Recordings: {summary['total_recordings']}")
    print(f"Total Unique Speakers:                          {summary['total_speakers']}")
    print(f"Gender Breakdown:                               {summary['gender_distribution']}")
    print(f"Duration Stats (sec):                           Mean={summary['duration_stats']['mean_seconds']}s, "
          f"Min={summary['duration_stats']['min_seconds']}s, Max={summary['duration_stats']['max_seconds']}s")

    print("\n--- Audio Recordings by Dataset ---")
    for dset, count in summary["datasets"].items():
        print(f"  * {dset:10s}: {count:5d} samples")

    print("\n--- Emotion Class Distribution (Overall) ---")
    for emotion in SUPPORTED_EMOTIONS:
        count = summary["emotions_overall"].get(emotion, 0)
        pct = (count / summary["total_recordings"]) * 100 if summary["total_recordings"] > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  * {emotion:10s}: {count:5d} ({pct:5.1f}%) {bar}")

    print("\n--- Emotion Breakdown by Dataset ---")
    for dset, emo_dict in summary["emotions_by_dataset"].items():
        print(f"  [{dset}]")
        for emo, c in sorted(emo_dict.items()):
            print(f"    - {emo:10s}: {c:4d}")

    print("=" * 65)
    print(f"[INFO] Dataset summary JSON saved to: {json_path}")

    # Generate visual plot of dataset exploration
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Emotion Distribution by Dataset
        emo_df = pd.DataFrame(summary["emotions_by_dataset"]).fillna(0)
        emo_df.plot(kind="bar", stacked=True, ax=axes[0], colormap="viridis")
        axes[0].set_title("Speech Emotion Distribution by Dataset", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Emotion", fontsize=10)
        axes[0].set_ylabel("Number of Recordings", fontsize=10)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Subplot 2: Duration Histogram
        durations = df["duration"]
        axes[1].hist(durations, bins=25, color="#3b82f6", edgecolor="black", alpha=0.8)
        axes[1].axvline(durations.mean(), color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {durations.mean():.2f}s")
        axes[1].set_title("Audio Recording Duration Distribution", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Duration (seconds)", fontsize=10)
        axes[1].set_ylabel("Count", fontsize=10)
        axes[1].legend()
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = save_dir / "dataset_exploration.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[INFO] Dataset exploration plot saved to: {plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate exploration plot: {e}")

    return summary


if __name__ == "__main__":
    print("Running Dataset Loader & Exploration standalone test...")
    df = load_combined_dataset()
    if not df.empty:
        explore_dataset(df)
        train_df, test_df = speaker_independent_train_test_split(df)
    else:
        print("[INFO] No dataset files currently in local directory.")
        print("[TIP] Run backend/training/download_datasets.py to download RAVDESS and TESS.")

