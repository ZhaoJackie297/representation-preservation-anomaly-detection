from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("RPAD_PYTHON", sys.executable))
MARKER = ROOT / "r6.2_audit_build.complete.json"


def run(script: str, *arguments: str) -> None:
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] RUN {script}",
        flush=True,
    )
    subprocess.run(
        [str(PYTHON), "-u", str(ROOT / script), *arguments],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def main() -> None:
    run("audit_r62_fast.py")
    run("build_r62_user_day.py", "--no-legacy-csv")
    run("prepare_r62_sequences.py")
    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
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
