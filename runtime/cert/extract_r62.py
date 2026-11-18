from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "r6.2.parallel.tar.bz2"
OUTPUT = ROOT / "r6.2_extracted"
TEMP = ROOT / "r6.2_extracted.inprogress"
MANIFEST = ROOT / "r6.2_extraction_manifest.json"
EXPECTED_SHA256 = (
    "6396bc0d6eb1d806ded0a70aaa3f111a5dcbb56269130e854c415e4e1b8eb3e9"
)
BUFFER_SIZE = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def safe_target(member: tarfile.TarInfo, root: Path) -> Path:
    name = member.name.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe archive path: {member.name}")
    target = (root / Path(*path.parts)).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise RuntimeError(f"Path escapes extraction root: {member.name}")
    return target


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    if TEMP.exists():
        raise FileExistsError(
            f"Temporary extraction directory already exists: {TEMP}"
        )

    print("Verifying SHA-256 before extraction...", flush=True)
    actual = sha256(ARCHIVE)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch: {actual} != {EXPECTED_SHA256}"
        )
    print(f"Verified SHA-256: {actual}", flush=True)

    TEMP.mkdir(parents=True)
    files = 0
    total_bytes = 0
    started = time.monotonic()
    last_report = started
    manifest_entries = []

    with tarfile.open(ARCHIVE, "r:bz2") as archive:
        for member in archive:
            target = safe_target(member, TEMP)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"Unsupported archive member: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {member.name}")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(
                    source, destination, length=BUFFER_SIZE
                )
            files += 1
            total_bytes += member.size
            manifest_entries.append(
                {
                    "path": member.name,
                    "size": member.size,
                }
            )
            now = time.monotonic()
            if now - last_report >= 15:
                last_report = now
                elapsed = max(now - started, 1e-6)
                print(
                    f"files={files:,} "
                    f"extracted_GiB={total_bytes / 1024**3:.2f} "
                    f"rate_MiBps={total_bytes / 1024**2 / elapsed:.2f}",
                    flush=True,
                )

    os.replace(TEMP, OUTPUT)
    manifest = {
        "archive": str(ARCHIVE),
        "archive_sha256": actual,
        "output": str(OUTPUT),
        "files": files,
        "uncompressed_bytes": total_bytes,
        "uncompressed_gib": total_bytes / 1024**3,
        "members": manifest_entries,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Extraction complete: {files:,} files, "
        f"{total_bytes / 1024**3:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
