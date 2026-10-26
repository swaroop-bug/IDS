"""Utility script to download genuine slices of the 8 CICIDS2017 CSV files from HuggingFace mirror.

Fetches a 5% stratified sample of real flow records across all 8 days/attacks
and saves them as canonical CSVs in data/raw/ for local smoke testing and development.
"""

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning"

FILES = [
    ("Monday-WorkingHours.pcap_ISCX.csv.parquet", "Monday-WorkingHours.pcap_ISCX.csv"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv.parquet", "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("Wednesday-workingHours.pcap_ISCX.csv.parquet", "Wednesday-workingHours.pcap_ISCX.csv"),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet", "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
]

def fetch_sample(sample_frac: float = 0.05):
    print(f"Downloading authentic CICIDS2017 flow samples (sample_frac={sample_frac})...")
    total_downloaded = 0

    for parquet_file, csv_file in FILES:
        target_path = RAW_DIR / csv_file
        if target_path.exists():
            print(f"  [Exists] {csv_file} already in data/raw/. Skipping.")
            continue

        url = f"{BASE_URL}/{parquet_file}"
        print(f"  Fetching {parquet_file} from {url}...")
        df = pd.read_parquet(url)

        # Retain attack flows and sample benign to ensure rare classes survive in smoke test
        if "Label" in df.columns:
            benign_df = df[df["Label"] == "BENIGN"]
            attack_df = df[df["Label"] != "BENIGN"]
            sampled_benign = benign_df.sample(frac=sample_frac, random_state=42)
            # Sample attack flows at max(sample_frac, 0.20) so rare attacks exist
            sampled_attack = attack_df.sample(frac=min(1.0, max(sample_frac, 0.20)), random_state=42)
            sampled = pd.concat([sampled_benign, sampled_attack]).sample(frac=1.0, random_state=42)
        else:
            sampled = df.sample(frac=sample_frac, random_state=42)

        sampled.to_csv(target_path, index=False)
        total_downloaded += len(sampled)
        print(f"  Saved {len(sampled):,} rows to {target_path.name}")

    print(f"Download complete! Total flows available in data/raw/: {total_downloaded:,}")

if __name__ == "__main__":
    fetch_sample(0.05)
