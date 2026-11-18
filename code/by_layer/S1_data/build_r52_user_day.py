from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

import build_r42_user_day as base


VERSION = "5.2"
PROCESSED = base.ROOT / "processed_r52"
EMAIL_SEND_ROWS = 0
FILE_WRITE_ROWS = 0

base.DATA = base.ROOT / "r5.2_extracted" / "r5.2"
base.OUT_DIR = PROCESSED
base.CACHE_DIR = PROCESSED / "cache"
base.FEATURE_FILE = PROCESSED / "r5.2_user_day_features.csv"
base.FEATURE_PARQUET = PROCESSED / "r5.2_user_day_features.parquet"
base.LABEL_FILE = PROCESSED / "r5.2_malicious_account_days.csv"
base.SAMPLE_FILE = PROCESSED / "r5.2_user_day_features_sample.csv"
base.SUMMARY_FILE = PROCESSED / "r5.2_user_day_summary.json"
base.LDAP_FILE = PROCESSED / "r5.2_ldap_missingness.csv"
base.PRIMARY_PC_END_DATE = pd.Timestamp("2010-04-15")


def load_labels_r52() -> pd.DataFrame:
    with tarfile.open(base.ANSWERS, "r:bz2") as archive:
        names = set(archive.getnames())
        master_file = archive.extractfile("answers/insiders.csv")
        if master_file is None:
            raise RuntimeError("Missing answers/insiders.csv")
        master = [
            row
            for row in csv.DictReader(
                io.TextIOWrapper(
                    master_file, encoding="utf-8-sig", newline=""
                )
            )
            if row["dataset"].strip() == VERSION
        ]

        labels: dict[tuple[str, str], dict] = defaultdict(
            lambda: {
                "scenarios": set(),
                "insider_ids": set(),
                "event_types": Counter(),
                "record_count": 0,
            }
        )
        for incident in master:
            detail = incident["details"].strip()
            scenario = incident["scenario"].strip()
            insider_id = incident["user"].strip()
            candidates = [
                f"answers/{detail}",
                f"answers/r{VERSION}-{scenario}/{detail}",
            ]
            path = next((item for item in candidates if item in names), None)
            if path is None:
                raise FileNotFoundError(detail)
            raw = archive.extractfile(path)
            if raw is None:
                raise RuntimeError(f"Cannot read {path}")
            detail_text = io.TextIOWrapper(
                raw, encoding="utf-8-sig", newline=""
            )
            for row in csv.reader(detail_text):
                if len(row) < 4:
                    continue
                timestamp = datetime.strptime(
                    row[2].strip(), "%m/%d/%Y %H:%M:%S"
                )
                account = row[3].strip()
                label = labels[(account, str(timestamp.date()))]
                label["scenarios"].add(scenario)
                label["insider_ids"].add(insider_id)
                label["event_types"][row[0].strip()] += 1
                label["record_count"] += 1

    rows = []
    for (account, date), label in sorted(labels.items()):
        rows.append(
            {
                "user_id": account,
                "date": date,
                "scenarios": "|".join(sorted(label["scenarios"])),
                "insider_ids": "|".join(sorted(label["insider_ids"])),
                "malicious_event_records": label["record_count"],
                "event_types": "|".join(
                    f"{event_type}:{count}"
                    for event_type, count in sorted(
                        label["event_types"].items()
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_email_send_events() -> pd.DataFrame:
    global EMAIL_SEND_ROWS
    frame = base.load_email_duckdb(
        activity_filter="send",
        attachments_as_list=True,
    )
    EMAIL_SEND_ROWS = int(frame["email_count"].sum())
    return frame


def load_file_writes() -> pd.DataFrame:
    global FILE_WRITE_ROWS
    frame = pd.read_csv(
        base.DATA / "file.csv",
        usecols=[
            "date",
            "user",
            "filename",
            "activity",
            "to_removable_media",
        ],
        dtype={
            "user": "string",
            "filename": "string",
            "activity": "string",
            "to_removable_media": "string",
        },
        low_memory=False,
    )
    frame = frame[
        frame["activity"].str.strip().str.lower().eq("file write")
        & frame["to_removable_media"]
        .str.strip()
        .str.lower()
        .eq("true")
    ].copy()
    FILE_WRITE_ROWS = len(frame)
    frame["timestamp"] = base.parse_date(frame["date"])
    frame["date"] = frame["timestamp"].dt.normalize()
    extension = (
        frame["filename"]
        .str.extract(r"\.([A-Za-z0-9]{1,10})$", expand=False)
        .str.lower()
        .fillna("none")
    )
    frame["is_archive"] = extension.isin(
        ["zip", "rar", "7z", "tar", "gz"]
    )
    frame["is_executable"] = extension.isin(
        ["exe", "dll", "bat", "cmd", "sh"]
    )
    frame["extension"] = extension
    return (
        frame.groupby(["user", "date"], sort=False)
        .agg(
            file_copy_count=("filename", "size"),
            archive_copy_count=("is_archive", "sum"),
            executable_copy_count=("is_executable", "sum"),
            unique_file_extensions=("extension", "nunique"),
        )
        .reset_index()
        .rename(columns={"user": "user_id"})
    )


base.load_labels = load_labels_r52
base.load_email = load_email_send_events
base.load_file_copies = load_file_writes


def main(
    *,
    force_refresh: bool = False,
    write_legacy_csv: bool = True,
) -> None:
    base.main(
        force_refresh=force_refresh,
        write_legacy_csv=write_legacy_csv,
    )
    summary = json.loads(
        base.SUMMARY_FILE.read_text(encoding="utf-8")
    )
    audit_file = base.ROOT / "r5.2_audit.json"
    if not audit_file.exists():
        raise FileNotFoundError(
            "Run audit_r52.py before building r5.2 user-day features."
        )
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    ldap_fields = set(audit["ldap"]["fields"])
    required_ldap_fields = {"role", "functional_unit"}
    missing_ldap_fields = sorted(required_ldap_fields - ldap_fields)
    if missing_ldap_fields:
        raise RuntimeError(
            f"Required LDAP fields are missing: {missing_ldap_fields}"
        )
    raw_totals = {
        name.removesuffix(".csv"): details["rows"]
        for name, details in audit["files"].items()
    }
    feature_totals = summary["feature_totals"]
    email_activities = audit["files"]["email.csv"]["activities"]
    email_send_rows = sum(
        count
        for activity, count in email_activities.items()
        if activity.strip().lower() == "send"
    )
    file_activities = audit["files"]["file.csv"]["activities"]
    file_write_rows = sum(
        count
        for activity, count in file_activities.items()
        if activity.strip().lower() == "file write"
    )
    count_checks = {
        "logon": (
            raw_totals["logon"],
            feature_totals["logon"] + feature_totals["logoff"],
        ),
        "device": (
            raw_totals["device"],
            feature_totals["usb_connect"]
            + feature_totals["usb_disconnect"],
        ),
        "file": (file_write_rows, feature_totals["file_copy"]),
        "email": (
            email_send_rows,
            feature_totals["email"],
        ),
        "http": (raw_totals["http"], feature_totals["http"]),
    }
    failed_checks = {
        name: {"raw": raw, "features": features}
        for name, (raw, features) in count_checks.items()
        if raw != features
    }
    if failed_checks:
        raise RuntimeError(
            f"Raw/feature event count mismatch: {failed_checks}"
        )
    if summary["label_keys_missing_from_features"] != 0:
        raise RuntimeError("Some malicious account-days are absent from features.")

    labels = pd.read_csv(base.LABEL_FILE, dtype={"user_id": "string"})
    feature_users = set(
        base.load_user_day_features(base.FEATURE_FILE)["user_id"].astype(str)
    )
    missing_accounts = sorted(
        set(labels["user_id"]) - feature_users
    )
    if missing_accounts:
        raise RuntimeError(
            f"Malicious accounts absent from feature users: {missing_accounts}"
        )

    summary["raw_event_totals"] = raw_totals
    summary["raw_feature_count_validation"] = {
        name: {"raw": raw, "feature_sum": features}
        for name, (raw, features) in count_checks.items()
    }
    summary["malicious_accounts_missing_from_features"] = 0
    summary["required_ldap_fields"] = sorted(required_ldap_fields)
    summary["primary_pc_source_end"] = str(
        base.PRIMARY_PC_END_DATE.date()
    )
    summary["primary_pc_uses_future_events"] = False
    summary["email_feature_scope"] = (
        "Send events only, matching the r4.2 email behavior semantics."
    )
    summary["email_activity_counts"] = email_activities
    summary["email_send_rows_used"] = email_send_rows
    summary["file_write_rows_used"] = file_write_rows
    summary["file_feature_scope"] = (
        "File Write to removable media only, matching r4.2 file-copy "
        "semantics."
    )
    summary["dataset_version"] = VERSION
    summary["label_source"] = "answers/insiders.csv dataset=5.2"
    base.SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CERT r5.2 user-day features."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore all per-source caches and rebuild from raw CSV.",
    )
    parser.add_argument(
        "--no-legacy-csv",
        action="store_true",
        help="Write Parquet but skip the large compatibility CSV.",
    )
    arguments = parser.parse_args()
    main(
        force_refresh=arguments.force_refresh,
        write_legacy_csv=not arguments.no_legacy_csv,
    )

