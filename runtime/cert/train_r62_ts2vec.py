from __future__ import annotations

from pathlib import Path

import train_r52_ts2vec as base


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r62"

base.PROCESSED = PROCESSED
base.VERSION = "6.2"
base.DATASET = PROCESSED / "r6.2_sequence_dataset.npz"
base.LABELS = PROCESSED / "r6.2_malicious_account_days.csv"
base.MODEL_FILE = PROCESSED / "r6.2_ts2vec_model.pth"
base.SCORE_FILE = PROCESSED / "r6.2_ts2vec_daily_errors.csv"
base.SUMMARY_FILE = PROCESSED / "r6.2_ts2vec_training.json"
base.TRAINING_CHECKPOINT = (
    PROCESSED / "r6.2_ts2vec_training_checkpoint.json"
)
base.SCORE_ARRAYS = PROCESSED / "r6.2_ts2vec_scores.npz"


if __name__ == "__main__":
    base.main()
