from __future__ import annotations

import hashlib
import math
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FILE_ID = 24844280
URL = (
    "https://s3.dualstack.eu-west-1.amazonaws.com/"
    f"pstorage-cmu-348901238291901/{FILE_ID}/r6.2.tar.bz2"
)
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "r6.2.parallel.tar.bz2"
SEGMENT_DIR = ROOT / "r6.2_segments"
EXPECTED_SIZE = 23_811_144_737
EXPECTED_SHA256 = (
    "6396bc0d6eb1d806ded0a70aaa3f111a5dcbb56269130e854c415e4e1b8eb3e9"
)
CHUNK_SIZE = 64 * 1024 * 1024
READ_SIZE = 1024 * 1024
WORKERS = 24
MAX_ATTEMPTS = 20


print_lock = threading.Lock()
progress_lock = threading.Lock()
completed_bytes = 0
last_report = 0.0


def log(message: str) -> None:
    with print_lock:
        print(message, flush=True)


def report_progress(delta: int, force: bool = False) -> None:
    global completed_bytes, last_report
    with progress_lock:
        completed_bytes += delta
        now = time.monotonic()
        if not force and now - last_report < 15:
            return
        last_report = now
        downloaded = completed_bytes
    print(
        f"{downloaded / 1024**3:.2f} GiB / "
        f"{EXPECTED_SIZE / 1024**3:.2f} GiB "
        f"({downloaded / EXPECTED_SIZE * 100:.2f}%)",
        flush=True,
    )


def segment_bounds(index: int) -> tuple[int, int]:
    start = index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, EXPECTED_SIZE) - 1
    return start, end


def fetch_segment(index: int) -> tuple[int, int]:
    start, end = segment_bounds(index)
    expected = end - start + 1
    path = SEGMENT_DIR / f"{index:05d}.part"

    if path.exists() and path.stat().st_size == expected:
        return index, expected
    if path.exists() and path.stat().st_size > expected:
        raise RuntimeError(
            f"Segment {index} is larger than expected: {path.stat().st_size} > {expected}"
        )

    if not path.exists():
        path.touch()

    attempts = 0
    while path.stat().st_size < expected:
        attempts += 1
        offset = path.stat().st_size
        request_start = start + offset
        expected_content_range = f"bytes {request_start}-{end}/{EXPECTED_SIZE}"
        request = urllib.request.Request(
            URL,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://kilthub.cmu.edu/",
                "Range": f"bytes={request_start}-{end}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Segment {index}: expected HTTP 206, got {response.status}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if content_range != expected_content_range:
                    raise RuntimeError(
                        f"Segment {index}: expected {expected_content_range!r}, "
                        f"got {content_range!r}"
                    )
                with path.open("ab") as handle:
                    while True:
                        chunk = response.read(READ_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        report_progress(len(chunk))
        except Exception as error:
            log(f"Segment {index} attempt {attempts} failed: {error}")
            if attempts >= MAX_ATTEMPTS:
                raise
            time.sleep(min(2 * attempts, 15))
            continue

        current = path.stat().st_size
        if current < expected:
            log(
                f"Segment {index} ended early at {current}/{expected} bytes; "
                "resuming."
            )
            if attempts >= MAX_ATTEMPTS:
                raise RuntimeError(f"Segment {index} exceeded retry limit")

    return index, expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(total_segments: int) -> None:
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".assembling")
    with temporary.open("wb") as output_handle:
        for index in range(total_segments):
            path = SEGMENT_DIR / f"{index:05d}.part"
            with path.open("rb") as segment_handle:
                while chunk := segment_handle.read(READ_SIZE):
                    output_handle.write(chunk)
            if index % 50 == 0 or index + 1 == total_segments:
                log(f"Assembled {index + 1}/{total_segments} segments")

    size = temporary.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"Assembled size mismatch: {size} != {EXPECTED_SIZE}")

    log("Assembled file complete; verifying SHA-256...")
    actual_sha256 = sha256(temporary)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch: {actual_sha256} != {EXPECTED_SHA256}"
        )

    os.replace(temporary, OUTPUT)
    log(f"Verified: {actual_sha256}")
    log(f"Output: {OUTPUT}")


def main() -> None:
    global completed_bytes
    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    total_segments = math.ceil(EXPECTED_SIZE / CHUNK_SIZE)

    for index in range(total_segments):
        path = SEGMENT_DIR / f"{index:05d}.part"
        if not path.exists():
            continue
        start, end = segment_bounds(index)
        expected = end - start + 1
        size = path.stat().st_size
        if size > expected:
            raise RuntimeError(
                f"Existing segment {index} is larger than expected: "
                f"{size} > {expected}"
            )
        completed_bytes += min(size, expected)

    log(
        f"Starting with {completed_bytes / 1024**3:.2f} GiB already present "
        f"across {total_segments} segments."
    )
    report_progress(0, force=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(fetch_segment, index)
            for index in range(total_segments)
        ]
        for future in as_completed(futures):
            future.result()

    report_progress(0, force=True)
    assemble(total_segments)


if __name__ == "__main__":
    main()
