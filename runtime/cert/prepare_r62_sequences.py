from __future__ import annotations

from pathlib import Path

import prepare_r52_sequences as base


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r62"

base.INPUT = PROCESSED / "r6.2_user_day_features.csv"
base.LABELS = PROCESSED / "r6.2_malicious_account_days.csv"
base.OUT_DIR = PROCESSED
base.ARRAY_FILE = PROCESSED / "r6.2_sequence_dataset.npz"
base.META_FILE = PROCESSED / "r6.2_sequence_metadata.json"


if __name__ == "__main__":
    base.main()
