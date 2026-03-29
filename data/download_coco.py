"""
data/download_coco.py
One-time script to download COCO val2017 annotations.

Run this once after extracting COCO images:
    python data/download_coco.py

What it downloads (~240MB):
  - instances_val2017.json    (object detection annotations)
  - captions_val2017.json     (image captions — useful for context)
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TARGET_DIR = Path(__file__).parent / "coco" / "annotations"
TARGET_FILES = ["instances_val2017.json", "captions_val2017.json"]


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with progress bar."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    zip_path = dest.parent / (dest.name + ".zip")

    with open(zip_path, "wb") as f, tqdm(
        desc=desc, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return zip_path


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    all_present = all((TARGET_DIR / f).exists() for f in TARGET_FILES)
    if all_present:
        print("✅ COCO annotations already present.")
        for f in TARGET_FILES:
            size = (TARGET_DIR / f).stat().st_size / 1e6
            print(f"   {f}: {size:.1f} MB")
        return

    print(f"📥 Downloading COCO annotations from:\n   {ANNOTATIONS_URL}\n")
    zip_path = download_file(
        ANNOTATIONS_URL,
        TARGET_DIR / "annotations",
        desc="COCO Annotations",
    )

    print("\n📦 Extracting selected annotation files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for target_file in TARGET_FILES:
            member = f"annotations/{target_file}"
            if member in zf.namelist():
                zf.extract(member, TARGET_DIR.parent)
                print(f"   ✅ Extracted: {target_file}")
            else:
                print(f"   ⚠️  Not found in zip: {member}")

    # Clean up zip
    zip_path.unlink()
    print(f"\n✅ Done! Annotations saved to: {TARGET_DIR}")


if __name__ == "__main__":
    main()
