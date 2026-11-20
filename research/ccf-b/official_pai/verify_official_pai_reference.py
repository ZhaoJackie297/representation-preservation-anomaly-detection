from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np


def official_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official PAI score loader on the CERT pilot component "
            "arrays and compare it with the repository implementation."
        )
    )
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_files = sorted(args.score_dir.glob("*_seed_*_scores.npz"))
    if not score_files:
        raise FileNotFoundError("no pilot score arrays found")

    with tempfile.TemporaryDirectory() as temporary:
        uniform_dir = Path(temporary)
        os.environ["PAIAD_UNIFORM_TS2VEC_DIR"] = str(uniform_dir)
        sys.path.insert(0, str(args.official_repo))
        sys.path.insert(0, str(args.official_repo / "aggregators"))
        importlib.import_module("pai_paths")
        official = importlib.import_module("build_full_table")

        checks = []
        for path in score_files:
            with np.load(path, allow_pickle=False) as arrays:
                euclidean = arrays["euclidean"].astype(np.float64)
                mag_g = arrays["mag_g_z"].astype(np.float64)
                t2 = arrays["t2_z"].astype(np.float64)
                expected = arrays["pai_official"].astype(np.float64)
            match = re.search(
                r"(?P<variant>T0_matched|T4_full_soft)_seed_(?P<seed>\d+)",
                path.name,
            )
            if match is None:
                raise ValueError(f"cannot parse score file name: {path.name}")
            fid = f"cert_{match.group('seed')}_{match.group('variant')}"
            np.save(uniform_dir / f"{fid}_eu.npy", euclidean)
            np.save(uniform_dir / f"{fid}_magG.npy", mag_g)
            np.save(uniform_dir / f"{fid}_T2.npy", t2)
            np.savez(
                uniform_dir / f"{fid}_train_stats.npz",
                eu_mean=0.0,
                eu_std=1.0,
                magG_mean=0.0,
                magG_std=1.0,
                T2_mean=0.0,
                T2_std=1.0,
            )
            actual = official._load_score("TS2Vec_ours", fid)
            if actual is None:
                raise RuntimeError(f"official PAI loader skipped {fid}")
            valid = np.isfinite(expected)
            difference = float(
                np.max(np.abs(actual[valid] - expected[valid]))
            )
            inactive_zero = bool(
                np.all(actual[~valid] == 0.0)
                and np.all(np.isfinite(actual))
            )
            checks.append(
                {
                    "file": path.name,
                    "seed": int(match.group("seed")),
                    "variant": match.group("variant"),
                    "max_abs_difference": difference,
                    "inactive_values_zero": inactive_zero,
                    "match": difference < 1e-6 and inactive_zero,
                }
            )

    report = {
        "status": "passed" if all(row["match"] for row in checks) else "failed",
        "official_repository": "pantheon5100/PAI",
        "official_commit": official_commit(args.official_repo),
        "official_formula": "0.6 * z_euclidean + 0.4 * z_magG + 0.2 * z_T2",
        "checks": checks,
        "tested_arrays": len(checks),
        "claim_boundary": (
            "This verifies the official scoring interface on CERT pilot "
            "components. It is not a native TSB-AD benchmark run."
        ),
        "license": "No repository-level license file was found locally.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
