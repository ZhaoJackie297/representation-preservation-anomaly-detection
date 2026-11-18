from __future__ import annotations

import argparse

import numpy as np

import tune_compare_r42 as tuning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["ts2vec", "isolation_forest"],
    )
    parser.add_argument("config")
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    tuning.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(tuning.DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = tuning.pd.to_datetime(
            dataset["dates"].astype(str)
        )
        values = dataset["values"].astype(np.float32)
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])

    if args.model == "ts2vec":
        config = next(
            item
            for item in tuning.TS2VEC_CONFIGS
            if item["name"] == args.config
        )
        print(
            f"Training TS2Vec {args.config} seed={args.seed}",
            flush=True,
        )
        tuning.ts2vec_raw_error(
            values,
            train_end,
            config,
            args.seed,
        )
    else:
        config = next(
            item
            for item in tuning.IFOREST_CONFIGS
            if item["name"] == args.config
        )
        values, _ = tuning.load_iforest_values(users, dates)
        print(
            f"Training Isolation Forest {args.config} seed={args.seed}",
            flush=True,
        )
        tuning.iforest_raw_score(
            values,
            active_mask,
            train_end,
            config,
            args.seed,
        )
    print("Finished", args.model, args.config, args.seed, flush=True)


if __name__ == "__main__":
    main()
