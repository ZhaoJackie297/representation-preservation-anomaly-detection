from __future__ import annotations

import argparse
import json

import pandas as pd

import build_r42_user_day as base
import build_r52_user_day as compat


VERSION = "6.2"
PROCESSED = base.ROOT / "processed_r62"

compat.VERSION = VERSION
base.DATA = base.ROOT / "r6.2_extracted" / "r6.2"
base.OUT_DIR = PROCESSED
base.CACHE_DIR = PROCESSED / "cache"
base.FEATURE_FILE = PROCESSED / "r6.2_user_day_features.csv"
base.FEATURE_PARQUET = PROCESSED / "r6.2_user_day_features.parquet"
base.LABEL_FILE = PROCESSED / "r6.2_malicious_account_days.csv"
base.SAMPLE_FILE = PROCESSED / "r6.2_user_day_features_sample.csv"
base.SUMMARY_FILE = PROCESSED / "r6.2_user_day_summary.json"
base.LDAP_FILE = PROCESSED / "r6.2_ldap_missingness.csv"
base.PRIMARY_PC_END_DATE = pd.Timestamp("2010-04-15")
base.load_labels = compat.load_labels_r52
base.load_email = compat.load_email_send_events
base.load_file_copies = compat.load_file_writes


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
    audit_file = base.ROOT / "r6.2_audit.json"
    if not audit_file.exists():
        raise FileNotFoundError(
            "Run audit_r62.py before building r6.2 user-day features."
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
        "email": (email_send_rows, feature_totals["email"]),
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
        raise RuntimeError(
            "Some malicious account-days are absent from features."
        )

    labels = pd.read_csv(
        base.LABEL_FILE, dtype={"user_id": "string"}
    )
    feature_users = set(
        base.load_user_day_features(
            base.FEATURE_FILE
        )["user_id"].astype(str)
    )
    missing_accounts = sorted(
        set(labels["user_id"]) - feature_users
    )
    if missing_accounts:
        raise RuntimeError(
            f"Malicious accounts absent from feature users: "
            f"{missing_accounts}"
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
        "Send events only, matching r4.2/r5.2 email semantics."
    )
    summary["email_activity_counts"] = email_activities
    summary["email_send_rows_used"] = email_send_rows
    summary["file_write_rows_used"] = file_write_rows
    summary["file_feature_scope"] = (
        "File Write to removable media only, matching r4.2/r5.2 "
        "file-copy semantics."
    )
    summary["dataset_version"] = VERSION
    summary["label_source"] = "answers/insiders.csv dataset=6.2"
    base.SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CERT r6.2 user-day features."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore per-source caches and rebuild from raw CSV.",
    )
    parser.add_argument(
        "--no-legacy-csv",
        action="store_true",
        help="Write Parquet but skip the legacy CSV.",
    )
    arguments = parser.parse_args()
    main(
        force_refresh=arguments.force_refresh,
        write_legacy_csv=not arguments.no_legacy_csv,
    )
