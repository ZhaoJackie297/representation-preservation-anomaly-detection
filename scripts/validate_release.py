from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".npy",
    ".npz",
    ".pth",
    ".pt",
    ".ckpt",
    ".parquet",
}

REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "runtime/cert/train_r42_ts2vec.py",
    "runtime/cert/e0_e1_representation_preservation.py",
    "runtime/cert/e2_softclt_training.py",
    "runtime/cert/e4_joint_instance_pai.py",
    "runtime/cert/e11_iforest_baseline.py",
    "results/canonical/canonical_metric_summary.csv",
    "results/canonical/joint_paired_comparisons.json",
    "paper/manuscript.docx",
    "references/ts2vec_official/ts2vec.py",
    "research/ccf-b/UPGRADE_PREREGISTRATION_20261120.md",
    "research/ccf-b/UPGRADE_PREREGISTRATION_20261120.json",
    "research/ccf-b/deep_baselines/README.md",
    "research/ccf-b/deep_baselines/CERT_DEEP_BASELINE_PILOT.md",
    "research/ccf-b/official_pai/README.md",
    "research/ccf-b/smd_multiseed/README.md",
]


def release_files(root: Path) -> list[Path]:
    """Return tracked and untracked release files, excluding ignored artifacts."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(path for path in root.rglob("*") if path.is_file())

    return sorted(
        root / relative.decode("utf-8")
        for relative in result.stdout.split(b"\0")
        if relative
    )


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    errors: list[str] = []
    files = release_files(root)

    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"missing required path: {relative}")

    python_files = sorted(
        path
        for path in files
        if path.suffix.lower() == ".py"
        if "__pycache__" not in path.parts
    )
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"Python parse error: {path.relative_to(root)}: {error}")

    for path in files:
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"large binary artifact present: {path.relative_to(root)}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"file exceeds 5 MiB: {path.relative_to(root)}")

    if errors:
        print("Release validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Release validation passed: {len(python_files)} Python files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

