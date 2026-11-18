from __future__ import annotations

import csv
import io
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
R1_ARCHIVE = ROOT / "r1.tar.bz2"
ANSWERS_ARCHIVE = ROOT / "answers.tar.bz2"
RESULT_JSON = ROOT / "r1_audit.json"


def parse_date(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%m/%d/%Y %H:%M:%S")


def audit_event_file(
    archive: tarfile.TarFile,
    member_name: str,
) -> dict:
    member = archive.getmember(member_name)
    raw = archive.extractfile(member)
    if raw is None:
        raise RuntimeError(f"Cannot read {member_name}")
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    is_http = member_name.endswith("http.csv")
    if is_http:
        columns = ["id", "date", "user", "pc", "url"]
        csv_reader = csv.reader(text)
        row_iterator = (
            dict(zip(columns, row))
            for row in csv_reader
            if len(row) >= len(columns)
        )
    else:
        dict_reader = csv.DictReader(text)
        columns = dict_reader.fieldnames or []
        row_iterator = iter(dict_reader)
    rows = 0
    users: set[str] = set()
    pcs: set[str] = set()
    domains: set[str] = set()
    missing = Counter({column: 0 for column in columns})
    activities = Counter()
    min_date: datetime | None = None
    max_date: datetime | None = None

    for row in row_iterator:
        rows += 1
        for column in columns:
            if not (row.get(column) or "").strip():
                missing[column] += 1
        user = (row.get("user") or "").strip()
        pc = (row.get("pc") or "").strip()
        if user:
            users.add(user)
        if pc:
            pcs.add(pc)
        activity = (row.get("activity") or "").strip()
        if activity:
            activities[activity] += 1
        if member_name.endswith("http.csv"):
            url = (row.get("url") or "").strip()
            if url:
                domain = urlparse(url).netloc.lower()
                if domain:
                    domains.add(domain)
        date_text = (row.get("date") or "").strip()
        if date_text:
            parsed = parse_date(date_text)
            min_date = parsed if min_date is None or parsed < min_date else min_date
            max_date = parsed if max_date is None or parsed > max_date else max_date
        if rows % 500000 == 0:
            print(f"  {member_name}: {rows:,} rows", flush=True)

    result = {
        "rows": rows,
        "columns": columns,
        "unique_users": len(users),
        "unique_pcs": len(pcs),
        "date_min": min_date.isoformat(sep=" ") if min_date else None,
        "date_max": max_date.isoformat(sep=" ") if max_date else None,
        "missing": {key: value for key, value in missing.items() if value},
        "activities": dict(activities.most_common()),
    }
    if member_name.endswith("http.csv"):
        result["unique_domains"] = len(domains)
    return result


def audit_ldap(archive: tarfile.TarFile, member_names: list[str]) -> dict:
    per_file = {}
    latest_roles = Counter()
    latest_users: set[str] = set()
    all_domains: set[str] = set()
    latest_name = ""

    for member_name in sorted(member_names):
        raw = archive.extractfile(archive.getmember(member_name))
        if raw is None:
            raise RuntimeError(f"Cannot read {member_name}")
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        rows = 0
        users: set[str] = set()
        roles = Counter()
        domains: set[str] = set()
        missing = Counter()
        for row in reader:
            rows += 1
            user = (row.get("user_id") or "").strip()
            role = (row.get("Role") or "").strip()
            domain = (row.get("Domain") or row.get("domain") or "").strip()
            if user:
                users.add(user)
            if role:
                roles[role] += 1
            if domain:
                domains.add(domain)
            for column, value in row.items():
                if not (value or "").strip():
                    missing[column] += 1
        all_domains.update(domains)
        label = Path(member_name).stem
        per_file[label] = {
            "rows": rows,
            "unique_users": len(users),
            "roles": dict(roles.most_common()),
            "domains": sorted(domains),
            "missing": dict(missing),
        }
        if label > latest_name:
            latest_name = label
            latest_roles = roles
            latest_users = users

    return {
        "files": per_file,
        "file_count": len(per_file),
        "latest_file": latest_name,
        "latest_user_count": len(latest_users),
        "latest_roles": dict(latest_roles.most_common()),
        "all_domains": sorted(all_domains),
    }


def audit_answers() -> dict:
    if not ANSWERS_ARCHIVE.is_file():
        return {"available": False}
    with tarfile.open(ANSWERS_ARCHIVE, "r:bz2") as archive:
        insiders = archive.extractfile("answers/insiders.csv")
        if insiders is None:
            return {"available": False}
        text = io.TextIOWrapper(insiders, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        rows = list(reader)
        by_dataset = Counter((row.get("dataset") or "").strip() for row in rows)
        r1_rows = [row for row in rows if (row.get("dataset") or "").strip() == "1"]
        return {
            "available": True,
            "datasets": dict(sorted(by_dataset.items())),
            "r1_labels_present": bool(r1_rows),
            "r1_label_rows": r1_rows,
        }


def main() -> None:
    if not R1_ARCHIVE.is_file():
        raise FileNotFoundError(R1_ARCHIVE)

    summary = {"archive": str(R1_ARCHIVE), "files": {}}
    with tarfile.open(R1_ARCHIVE, "r:bz2") as archive:
        names = archive.getnames()
        print("Auditing event files...", flush=True)
        for member_name in ("r1/logon.csv", "r1/device.csv", "r1/http.csv"):
            print(f"Reading {member_name}", flush=True)
            summary["files"][Path(member_name).name] = audit_event_file(
                archive, member_name
            )

        ldap_names = [name for name in names if name.startswith("r1/LDAP/") and name.endswith(".csv")]
        print(f"Auditing {len(ldap_names)} LDAP files...", flush=True)
        summary["ldap"] = audit_ldap(archive, ldap_names)

    print("Auditing answer key...", flush=True)
    summary["answers"] = audit_answers()
    summary["label_status"] = (
        "r1 has no malicious scenario labels in the official answer key"
        if not summary["answers"].get("r1_labels_present")
        else "r1 labels present"
    )

    RESULT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULT_JSON}", flush=True)


if __name__ == "__main__":
    main()
