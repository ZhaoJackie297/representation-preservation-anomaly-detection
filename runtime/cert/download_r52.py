from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path


URL = "https://api.figshare.com/v2/file/download/24849938"
OUTPUT = Path(__file__).resolve().parent / "r5.2.tar.bz2"
EXPECTED_SIZE = 11_137_370_937
EXPECTED_SHA256 = (
    "9a7fadba71482474e8a2a7dddbf3620d450796d0059526db3362d41096d0f5bf"
)
CHUNK_SIZE = 4 * 1024 * 1024


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
    with urllib.request.urlopen(request, timeout=180) as response:
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
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - last_report >= 15:
                    size = offset + written
                    print(
                        f"{size / 1024**3:.2f} GiB / "
                        f"{EXPECTED_SIZE / 1024**3:.2f} GiB "
                        f"({size / EXPECTED_SIZE * 100:.1f}%)",
                        flush=True,
                    )
                    last_report = now
    return written


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not OUTPUT.exists():
        OUTPUT.touch()

    attempts = 0
    while True:
        size = OUTPUT.stat().st_size
        if size == EXPECTED_SIZE:
            break
        if size > EXPECTED_SIZE:
            raise RuntimeError(
                f"Partial file is larger than expected: {size} > {EXPECTED_SIZE}"
            )
        attempts += 1
        print(
            f"Resume attempt {attempts} from {size / 1024**3:.2f} GiB",
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
                f"Connection ended early at {current / 1024**3:.2f} GiB; "
                "resuming.",
                flush=True,
            )
            if attempts >= 20:
                raise RuntimeError("Reached maximum resume attempts")
            time.sleep(2)

    print("Download complete; verifying SHA-256...", flush=True)
    actual_sha256 = sha256(OUTPUT)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch: {actual_sha256} != {EXPECTED_SHA256}"
        )
    print(f"Verified: {actual_sha256}", flush=True)


if __name__ == "__main__":
    main()
