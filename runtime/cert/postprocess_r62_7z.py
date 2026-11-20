from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("RPAD_PYTHON", sys.executable))
SOURCE = ROOT / "r6.2_extracted_7z.inprogress"
OUTPUT = ROOT / "r6.2_extracted"
EXTRACTION_LOG = ROOT / "r6.2_7z_cmd2.log"
EXTRACTION_ERROR = ROOT / "r6.2_7z_cmd2.err.log"
MARKER = ROOT / "r6.2_postprocess_7z_cmd.complete.json"


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
    print("Waiting for 7-Zip extraction completion marker...", flush=True)
    while True:
        if EXTRACTION_LOG.exists():
            text = EXTRACTION_LOG.read_text(
                encoding="utf-8", errors="replace"
            )
            if "7Z_EXTRACTION_COMPLETE" in text:
                break
        if EXTRACTION_ERROR.exists():
            error = EXTRACTION_ERROR.read_text(
                encoding="utf-8", errors="replace"
            )
            if "ERROR" in error.upper() or "THROW" in error.upper():
                raise RuntimeError(error[-4000:])
        time.sleep(15)

    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    os.replace(SOURCE, OUTPUT)
    print(f"Extraction directory finalized: {OUTPUT}", flush=True)

    run("audit_r62.py")
    run("build_r62_user_day.py", "--no-legacy-csv")
    run("prepare_r62_sequences.py")

    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(ROOT / "r6.2.parallel.tar.bz2"),
        "extracted": str(OUTPUT),
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
