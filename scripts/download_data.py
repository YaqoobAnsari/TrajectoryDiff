#!/usr/bin/env python3
"""
Download and setup RadioMapSeer dataset.

Usage:
    python scripts/download_data.py --method gdrive   # Download from Google Drive
    python scripts/download_data.py --method manual   # Show manual instructions
    python scripts/download_data.py --verify          # Verify existing download
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# Dataset information
DATASET_INFO = {
    "name": "RadioMapSeer",
    "gdrive_id": "1_EHrS-Sp25nzmbJKbwnNVzbSR57jb6_8",
    "gdrive_url": "https://drive.google.com/open?id=1_EHrS-Sp25nzmbJKbwnNVzbSR57jb6_8",
    "ieee_url": "https://ieee-dataport.org/documents/dataset-pathloss-and-toa-radio-maps-localization-application",
    "expected_size_gb": 3.0,
    "recommended_variant": "IRT2HighRes.zip",
}


def get_data_dir() -> Path:
    """Get the data directory path."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "raw" / "RadioMapSeer"
    return data_dir


def download_from_gdrive(file_id: str, output_path: Path) -> bool:
    """
    Download file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID
        output_path: Where to save the file

    Returns:
        True if successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive download...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from Google Drive...")
    print(f"File ID: {file_id}")
    print(f"Output: {output_path}")
    print(f"Expected size: ~{DATASET_INFO['expected_size_gb']} GB")
    print()

    try:
        gdown.download(url, str(output_path), quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip file."""
    print(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def verify_dataset(data_dir: Path) -> dict:
    """
    Verify the dataset structure and count files.

    Returns:
        dict with verification results
    """
    results = {
        "exists": data_dir.exists(),
        "png_count": 0,
        "json_count": 0,
        "subdirs": [],
        "total_size_mb": 0,
    }

    if not data_dir.exists():
        return results

    for root, dirs, files in os.walk(data_dir):
        results["subdirs"].extend(dirs)
        for f in files:
            fpath = Path(root) / f
            results["total_size_mb"] += fpath.stat().st_size / (1024 * 1024)
            if f.endswith(".png"):
                results["png_count"] += 1
            elif f.endswith(".json"):
                results["json_count"] += 1

    results["subdirs"] = list(set(results["subdirs"]))[:10]  # First 10 unique
    return results


def print_manual_instructions():
    """Print manual download instructions."""
    print("=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Option 1: Google Drive (Original, 3GB)")
    print("-" * 40)
    print(f"1. Open: {DATASET_INFO['gdrive_url']}")
    print("2. Click 'Download' button")
    print("3. Extract to: data/raw/RadioMapSeer/")
    print()
    print("Option 2: IEEE DataPort (Recommended, has IRT2HighRes)")
    print("-" * 40)
    print(f"1. Open: {DATASET_INFO['ieee_url']}")
    print("2. Create IEEE account if needed")
    print("3. Download 'IRT2HighRes.zip' (~930 MB)")
    print("4. Extract to: data/raw/RadioMapSeer/")
    print()
    print("After download, run:")
    print("  python scripts/download_data.py --verify")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download RadioMapSeer dataset")
    parser.add_argument(
        "--method",
        choices=["gdrive", "manual"],
        default="manual",
        help="Download method (default: manual)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset"
    )
    args = parser.parse_args()

    data_dir = get_data_dir()

    if args.verify:
        print(f"Verifying dataset at: {data_dir}")
        print("-" * 50)
        results = verify_dataset(data_dir)

        if not results["exists"]:
            print("Dataset directory does not exist!")
            print(f"Expected: {data_dir}")
            return 1

        print(f"PNG files found: {results['png_count']}")
        print(f"JSON files found: {results['json_count']}")
        print(f"Total size: {results['total_size_mb']:.1f} MB")
        print(f"Subdirectories: {results['subdirs']}")

        if results["png_count"] > 0:
            print("\n✓ Dataset appears to be present!")
        else:
            print("\n✗ No PNG files found. Dataset may be incomplete.")
        return 0

    if args.method == "manual":
        print_manual_instructions()
        return 0

    elif args.method == "gdrive":
        zip_path = data_dir.parent / "RadioMapSeer.zip"

        if download_from_gdrive(DATASET_INFO["gdrive_id"], zip_path):
            if extract_zip(zip_path, data_dir):
                print("\n✓ Download and extraction complete!")
                print(f"Dataset location: {data_dir}")

                # Verify
                results = verify_dataset(data_dir)
                print(f"PNG files: {results['png_count']}")
                return 0

        print("\n✗ Download failed. Try manual download.")
        print_manual_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())
