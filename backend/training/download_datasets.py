"""
Dataset Downloader for Speech Emotion Recognition
Downloads and extracts official RAVDESS and TESS datasets into backend/datasets/
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to allow importing config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAVDESS_DIR, TESS_DIR, DATASETS_DIR

# Direct verified download links
DATASET_URLS = {
    "RAVDESS": {
        "url": "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip",
        "target_dir": RAVDESS_DIR,
        "zip_name": "Audio_Speech_Actors_01-24.zip",
        "description": "RAVDESS Audio Speech (Actors 01-24, 1440 files)"
    },
    "TESS": {
        "url": "https://huggingface.co/datasets/myleslinder/tess/resolve/main/data/tess.zip",
        "target_dir": TESS_DIR,
        "zip_name": "tess.zip",
        "description": "TESS Toronto Emotional Speech Set (2800 files)"
    }
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract(dataset_key: str):
    info = DATASET_URLS[dataset_key]
    target_dir = info["target_dir"]
    zip_path = DATASETS_DIR / info["zip_name"]
    url = info["url"]

    # Check if files already exist
    existing_wavs = list(target_dir.rglob("*.wav"))
    if len(existing_wavs) > 100:
        print(f"[INFO] {dataset_key} dataset already exists with {len(existing_wavs)} .wav files in {target_dir}")
        return len(existing_wavs)

    print(f"\n==========================================")
    print(f"Downloading {info['description']}...")
    print(f"Source URL: {url}")
    print(f"Destination: {zip_path}")
    print(f"==========================================")

    # Download if zip doesn't exist
    if not zip_path.exists():
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=info['zip_name']) as t:
            urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
        print(f"[SUCCESS] Download completed: {zip_path}")
    else:
        print(f"[INFO] Zip archive already exists at {zip_path}")

    # Extract
    print(f"Extracting {zip_path.name} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    extracted_wavs = list(target_dir.rglob("*.wav"))
    print(f"[SUCCESS] Extracted {len(extracted_wavs)} .wav files to {target_dir}")

    # Optional: clean up zip file to save disk space if requested, or keep it
    # os.remove(zip_path)

    return len(extracted_wavs)


def download_all():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print("Checking datasets for Speech Emotion Recognition...")
    results = {}
    for key in DATASET_URLS:
        try:
            count = download_and_extract(key)
            results[key] = count
        except Exception as e:
            print(f"[ERROR] Failed to download/extract {key}: {e}")
            results[key] = 0

    print("\nDataset Download Summary:")
    for k, v in results.items():
        print(f" - {k}: {v} audio files")
    return results


if __name__ == "__main__":
    download_all()

