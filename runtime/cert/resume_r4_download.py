from __future__ import annotations

import time
import urllib.request
from pathlib import Path


URL = "https://api.figshare.com/v2/file/download/24856766"
OUTPUT = Path(__file__).resolve().parent / "r4.2.tar.bz2"
EXPECTED_SIZE = 4_824_287_500


def download_once(offset: int) -> int:
    request = urllib.request.Request(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://kilthub.cmu.edu/",
            "Range": f"bytes={offset}-",
        },
    )
    written = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(
                f"Expected HTTP 206 for resume, received {response.status}"
            )
        content_range = response.headers.get("Content-Range", "")
        if not content_range.endswith(str(EXPECTED_SIZE)):
            raise RuntimeError(f"Unexpected Content-Range: {content_range}")
        with OUTPUT.open("ab") as handle:
            last_report = time.monotonic()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - last_report >= 15:
                    size = offset + written
                    print(
                        f"{size / 1024**2:.1f} MiB / "
                        f"{EXPECTED_SIZE / 1024**2:.1f} MiB "
                        f"({size / EXPECTED_SIZE * 100:.1f}%)",
                        flush=True,
                    )
                    last_report = now
    return written


def main() -> None:
    if not OUTPUT.exists():
        OUTPUT.touch()

    attempts = 0
    while True:
        size = OUTPUT.stat().st_size
        if size == EXPECTED_SIZE:
            print(f"Download complete: {size} bytes", flush=True)
            return
        if size > EXPECTED_SIZE:
            raise RuntimeError(
                f"Partial file is larger than expected: {size} > {EXPECTED_SIZE}"
            )
        attempts += 1
        print(
            f"Resume attempt {attempts} from {size / 1024**2:.1f} MiB",
            flush=True,
        )
        try:
            written = download_once(size)
            if written == 0:
                raise RuntimeError("No bytes received")
        except Exception as error:
            print(f"Download attempt failed: {error}", flush=True)
            if attempts >= 20:
                raise
            time.sleep(5)
            continue
        current = OUTPUT.stat().st_size
        if current < EXPECTED_SIZE:
            print(
                f"Connection ended early at {current / 1024**2:.1f} MiB; resuming.",
                flush=True,
            )
            if attempts >= 20:
                raise RuntimeError("Reached maximum resume attempts")
            time.sleep(2)


if __name__ == "__main__":
    main()
