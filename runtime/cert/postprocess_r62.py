from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\Lenovo\.conda\envs\ts2vec\python.exe")
ARCHIVE = ROOT / "r6.2.parallel.tar.bz2"
EXPECTED_SIZE = 23_811_144_737
MARKER = ROOT / "r6.2_postprocess.complete.json"


def run(script: str, *arguments: str) -> None:
    command = [str(PYTHON), "-u", str(ROOT / script), *arguments]
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] RUN {script}",
        flush=True,
    )
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def main() -> None:
    print(
        "Waiting for a complete r6.2 archive with the expected size...",
        flush=True,
    )
    while True:
        if ARCHIVE.exists():
            size = ARCHIVE.stat().st_size
            if size == EXPECTED_SIZE:
                break
            if size > EXPECTED_SIZE:
                raise RuntimeError(
                    f"Archive is larger than expected: {size}"
                )
        time.sleep(30)

    print(
        f"Archive size verified: {ARCHIVE.stat().st_size} bytes",
        flush=True,
    )
    run("extract_r62.py")
    run("audit_r62.py")
    run("build_r62_user_day.py", "--no-legacy-csv")
    run("prepare_r62_sequences.py")

    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(ARCHIVE),
        "archive_size": ARCHIVE.stat().st_size,
        "audit": str(ROOT / "r6.2_audit.json"),
        "processed": str(ROOT / "processed_r62"),
    }
    MARKER.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
