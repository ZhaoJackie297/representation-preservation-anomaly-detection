from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "r4.2_extracted" / "r4.2"
ANSWERS = ROOT / "answers.tar.bz2"
OUT_DIR = ROOT / "processed_r42"
CACHE_DIR = OUT_DIR / "cache"
FEATURE_FILE = OUT_DIR / "r4.2_user_day_features.csv"
FEATURE_PARQUET = OUT_DIR / "r4.2_user_day_features.parquet"
LABEL_FILE = OUT_DIR / "r4.2_malicious_account_days.csv"
SAMPLE_FILE = OUT_DIR / "r4.2_user_day_features_sample.csv"
SUMMARY_FILE = OUT_DIR / "r4.2_user_day_summary.json"
LDAP_FILE = OUT_DIR / "r4.2_ldap_missingness.csv"

WORK_START_HOUR = 8
WORK_END_HOUR = 18
PRIMARY_PC_END_DATE: pd.Timestamp | None = None
DUCKDB_THREADS = int(
    os.environ.get("CERT_S1_DUCKDB_THREADS", min(8, os.cpu_count() or 1))
)


def value_or_unknown(value) -> str:
    return "Unknown" if pd.isna(value) else str(value)


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%m/%d/%Y %H:%M:%S", errors="raise")


def minute_of_day(timestamps: pd.Series) -> pd.Series:
    return (
        timestamps.dt.hour * 60
        + timestamps.dt.minute
        + timestamps.dt.second / 60.0
    )


def load_logon() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    frame = pd.read_csv(
        DATA / "logon.csv",
        usecols=["date", "user", "pc", "activity"],
        dtype={"user": "string", "pc": "string", "activity": "string"},
    )
    frame["timestamp"] = parse_date(frame["date"])
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["minute"] = minute_of_day(frame["timestamp"])
    frame["is_logon"] = frame["activity"].eq("Logon")
    frame["is_logoff"] = frame["activity"].eq("Logoff")
    frame["after_hours"] = (
        (frame["timestamp"].dt.hour < WORK_START_HOUR)
        | (frame["timestamp"].dt.hour >= WORK_END_HOUR)
    )
    frame["after_hours_logon"] = frame["is_logon"] & frame["after_hours"]

    primary_pc_source = frame
    if PRIMARY_PC_END_DATE is not None:
        primary_pc_source = frame[
            frame["date"] <= PRIMARY_PC_END_DATE.normalize()
        ]
    primary_pc = (
        primary_pc_source.loc[primary_pc_source["is_logon"]]
        .groupby("user")["pc"]
        .agg(lambda series: series.value_counts().index[0])
        .to_dict()
    )
    frame["primary_pc"] = frame["user"].map(primary_pc)
    frame["other_pc_logon"] = frame["is_logon"] & frame["pc"].ne(frame["primary_pc"])
    frame["logon_minute_value"] = np.where(frame["is_logon"], frame["minute"], np.nan)
    frame["logoff_minute_value"] = np.where(frame["is_logoff"], frame["minute"], np.nan)

    grouped = (
        frame.groupby(["user", "date"], sort=False)
        .agg(
            logon_count=("is_logon", "sum"),
            logoff_count=("is_logoff", "sum"),
            after_hours_logon_count=("after_hours_logon", "sum"),
            other_pc_logon_count=("other_pc_logon", "sum"),
            first_logon_minute=("logon_minute_value", "min"),
            last_logoff_minute=("logoff_minute_value", "max"),
        )
        .reset_index()
        .rename(columns={"user": "user_id"})
    )
    ranges = (
        frame.groupby("user", sort=True)["date"]
        .agg(["min", "max"])
        .rename(columns={"min": "first_date", "max": "last_date"})
        .reset_index()
        .rename(columns={"user": "user_id"})
    )
    return grouped, ranges, primary_pc


def load_device() -> pd.DataFrame:
    frame = pd.read_csv(
        DATA / "device.csv",
        usecols=["date", "user", "activity"],
        dtype={"user": "string", "activity": "string"},
    )
    frame["timestamp"] = parse_date(frame["date"])
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["connect"] = frame["activity"].eq("Connect")
    frame["disconnect"] = frame["activity"].eq("Disconnect")
    return (
        frame.groupby(["user", "date"], sort=False)
        .agg(
            usb_connect_count=("connect", "sum"),
            usb_disconnect_count=("disconnect", "sum"),
        )
        .reset_index()
        .rename(columns={"user": "user_id"})
    )


def load_file_copies() -> pd.DataFrame:
    frame = pd.read_csv(
        DATA / "file.csv",
        usecols=["date", "user", "filename"],
        dtype={"user": "string", "filename": "string"},
    )
    frame["timestamp"] = parse_date(frame["date"])
    frame["date"] = frame["timestamp"].dt.normalize()
    extension = (
        frame["filename"]
        .str.extract(r"\.([A-Za-z0-9]{1,10})$", expand=False)
        .str.lower()
        .fillna("none")
    )
    frame["is_archive"] = extension.isin(["zip", "rar", "7z", "tar", "gz"])
    frame["is_executable"] = extension.isin(["exe", "dll", "bat", "cmd", "sh"])
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


def recipient_fields(row: dict) -> tuple[int, int]:
    recipients = []
    for field in ("to", "cc", "bcc"):
        value = (row.get(field) or "").strip()
        if value:
            recipients.extend(
                part.strip().lower()
                for part in value.split(";")
                if part.strip()
            )
    external = sum(
        1
        for address in recipients
        if not address.endswith("@dtaa.com")
    )
    return len(recipients), external


def load_email_python() -> pd.DataFrame:
    """Reference implementation kept for exact A/B verification."""
    records = []
    path = DATA / "email.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = datetime.strptime(
                row["date"].strip(), "%m/%d/%Y %H:%M:%S"
            )
            recipient_count, external_count = recipient_fields(row)
            records.append(
                (
                    row["user"].strip(),
                    timestamp.date(),
                    recipient_count,
                    external_count,
                    int(row["attachments"] or 0),
                    int(row["size"] or 0),
                    int(timestamp.hour < WORK_START_HOUR or timestamp.hour >= WORK_END_HOUR),
                )
            )
    frame = pd.DataFrame(
        records,
        columns=[
            "user_id",
            "date",
            "email_recipient_count",
            "external_recipient_count",
            "email_attachment_count",
            "email_size_bytes",
            "after_hours_email",
        ],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["email_count"] = 1
    return (
        frame.groupby(["user_id", "date"], sort=False)
        .agg(
            email_count=("email_count", "sum"),
            email_recipient_count=("email_recipient_count", "sum"),
            external_recipient_count=("external_recipient_count", "sum"),
            email_attachment_count=("email_attachment_count", "sum"),
            email_size_bytes=("email_size_bytes", "sum"),
            after_hours_email_count=("after_hours_email", "sum"),
        )
        .reset_index()
    )


def duckdb_connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(f"SET threads = {DUCKDB_THREADS}")
    return connection


def load_email_duckdb(
    *,
    activity_filter: str | None = None,
    attachments_as_list: bool = False,
) -> pd.DataFrame:
    """Aggregate email events without materializing millions of Python rows.

    ``attachment_count`` has two historical meanings in this project:

    * r4.2 stores a numeric count in the ``attachments`` column;
    * r5.2 stores a semicolon-separated list and counts non-empty items.

    The flag keeps both versions exact instead of silently changing the
    feature definition.
    """

    if attachments_as_list:
        attachment_sql = """
            list_count(
                list_filter(
                    string_split(
                        coalesce(trim("attachments"::VARCHAR), ''),
                        ';'
                    ),
                    item -> len(trim(item)) > 0
                )
            )
        """
    else:
        attachment_sql = """
            coalesce(
                try_cast(trim("attachments"::VARCHAR) AS BIGINT),
                0
            )
        """

    if activity_filter is None:
        activity_where = ""
    else:
        escaped = activity_filter.replace("'", "''").lower()
        activity_where = (
            "WHERE lower(trim(\"activity\"::VARCHAR)) "
            f"= '{escaped}'"
        )

    query = f"""
        WITH raw AS (
            SELECT
                trim("user"::VARCHAR) AS user_id,
                strptime(
                    "date"::VARCHAR,
                    '%m/%d/%Y %H:%M:%S'
                ) AS timestamp,
                concat_ws(
                    ';',
                    coalesce("to"::VARCHAR, ''),
                    coalesce("cc"::VARCHAR, ''),
                    coalesce("bcc"::VARCHAR, '')
                ) AS recipient_text,
                {attachment_sql} AS attachment_count,
                coalesce(
                    try_cast(trim("size"::VARCHAR) AS BIGINT),
                    0
                ) AS email_size_bytes
            FROM read_csv(
                ?,
                header = true,
                all_varchar = true
            )
            {activity_where}
        ),
        parsed AS (
            SELECT
                user_id,
                CAST(timestamp AS DATE) AS date,
                list_filter(
                    string_split(recipient_text, ';'),
                    item -> len(trim(item)) > 0
                ) AS recipients,
                attachment_count,
                email_size_bytes,
                date_part('hour', timestamp) AS hour
            FROM raw
        )
        SELECT
            user_id,
            date,
            count(*)::BIGINT AS email_count,
            sum(len(recipients))::BIGINT AS email_recipient_count,
            sum(
                list_count(
                    list_filter(
                        recipients,
                        item -> NOT ends_with(
                            lower(trim(item)),
                            '@dtaa.com'
                        )
                    )
                )
            )::BIGINT AS external_recipient_count,
            sum(attachment_count)::BIGINT AS email_attachment_count,
            sum(email_size_bytes)::BIGINT AS email_size_bytes,
            sum(
                CASE
                    WHEN hour < ?
                      OR hour >= ?
                    THEN 1
                    ELSE 0
                END
            )::BIGINT AS after_hours_email_count
        FROM parsed
        GROUP BY user_id, date
    """

    connection = duckdb_connection()
    try:
        frame = connection.execute(
            query,
            [
                str(DATA / "email.csv"),
                WORK_START_HOUR,
                WORK_END_HOUR,
            ],
        ).fetchdf()
    finally:
        connection.close()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_email() -> pd.DataFrame:
    return load_email_duckdb()


def load_http_python() -> pd.DataFrame:
    """Reference implementation kept for exact A/B verification."""
    aggregates: dict[tuple[str, str], list] = {}
    path = DATA / "http.csv"
    with path.open("rb", buffering=1024 * 1024) as handle:
        handle.readline()
        for row_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.rstrip(b"\r\n").split(b",", 4)
            if len(parts) < 5:
                continue
            date_text = parts[1].decode("ascii", "replace")
            user = parts[2].decode("utf-8", "replace")
            day = (
                date_text[6:10]
                + "-"
                + date_text[0:2]
                + "-"
                + date_text[3:5]
            )
            hour = int(date_text[11:13])
            rest = parts[4]
            if rest.startswith(b'"'):
                end = rest.find(b'"', 1)
                url_bytes = rest[1:end] if end > 1 else b""
            else:
                url_bytes = rest.split(b",", 1)[0]
            try:
                url = url_bytes.decode("utf-8", "replace")
                domain = urlparse(url).netloc.lower()
            except Exception:
                domain = ""
            key = (user, day)
            entry = aggregates.get(key)
            if entry is None:
                entry = [0, 0, 0, set()]
                aggregates[key] = entry
            entry[0] += 1
            if hour < WORK_START_HOUR or hour >= WORK_END_HOUR:
                entry[1] += 1
            if domain and not (domain == "dtaa.com" or domain.endswith(".dtaa.com")):
                entry[2] += 1
            if domain:
                entry[3].add(domain)
            if row_number % 5_000_000 == 0:
                print(f"  http.csv: {row_number:,} rows", flush=True)

    frame = pd.DataFrame(
        [
            (
                user,
                day,
                values[0],
                values[1],
                values[2],
                len(values[3]),
            )
            for (user, day), values in aggregates.items()
        ],
        columns=[
            "user_id",
            "date",
            "http_count",
            "after_hours_http_count",
            "external_http_count",
            "unique_http_domains",
        ],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_http_duckdb() -> pd.DataFrame:
    """Aggregate the 13+ GiB HTTP log in DuckDB's vectorized engine.

    The regular expression mirrors ``urllib.parse.urlparse(url).netloc`` for
    valid absolute URLs: a scheme followed by ``://`` and the authority
    component up to the first path, query, or fragment delimiter.
    """

    query = """
        WITH raw AS (
            SELECT
                "user"::VARCHAR AS user_id,
                strptime(
                    "date"::VARCHAR,
                    '%m/%d/%Y %H:%M:%S'
                ) AS timestamp,
                "url"::VARCHAR AS url
            FROM read_csv(
                ?,
                header = true,
                all_varchar = true
            )
        ),
        parsed AS (
            SELECT
                user_id,
                CAST(timestamp AS DATE) AS date,
                date_part('hour', timestamp) AS hour,
                lower(
                    regexp_extract(
                        url,
                        '^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]*)',
                        1
                    )
                ) AS domain
            FROM raw
        )
        SELECT
            user_id,
            date,
            count(*)::BIGINT AS http_count,
            sum(
                CASE
                    WHEN hour < ?
                      OR hour >= ?
                    THEN 1
                    ELSE 0
                END
            )::BIGINT AS after_hours_http_count,
            sum(
                CASE
                    WHEN len(domain) > 0
                     AND NOT (
                         domain = 'dtaa.com'
                         OR ends_with(domain, '.dtaa.com')
                     )
                    THEN 1
                    ELSE 0
                END
            )::BIGINT AS external_http_count,
            count(
                DISTINCT CASE
                    WHEN len(domain) > 0 THEN domain
                END
            )::BIGINT AS unique_http_domains
        FROM parsed
        GROUP BY user_id, date
    """

    connection = duckdb_connection()
    try:
        frame = connection.execute(
            query,
            [
                str(DATA / "http.csv"),
                WORK_START_HOUR,
                WORK_END_HOUR,
            ],
        ).fetchdf()
    finally:
        connection.close()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_http() -> pd.DataFrame:
    return load_http_duckdb()


def load_ldap() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load all monthly LDAP snapshots as one tidy lookup table."""

    lookup_parts = []
    months = []
    missing_rows = []
    fields = [
        "role",
        "functional_unit",
        "department",
        "team",
        "supervisor",
    ]
    for path in sorted((DATA / "LDAP").glob("*.csv")):
        month = path.stem
        months.append(month)
        frame = pd.read_csv(path, dtype="string")
        missing_rows.append(
            {"month": month, **{field: int(frame[field].isna().sum()) for field in fields}}
        )
        snapshot = frame[["user_id", *fields]].copy()
        snapshot["user_id"] = snapshot["user_id"].fillna("Unknown").astype(str)
        for field in fields:
            snapshot[field] = (
                snapshot[field]
                .map(value_or_unknown)
                .astype(str)
            )
        snapshot["month"] = month
        lookup_parts.append(snapshot)

    if not lookup_parts:
        raise FileNotFoundError(f"No LDAP snapshots found in {DATA / 'LDAP'}")
    lookup = (
        pd.concat(lookup_parts, ignore_index=True)
        .drop_duplicates(subset=["month", "user_id"], keep="last")
        .reset_index(drop=True)
    )
    return lookup, months, pd.DataFrame(missing_rows)


def build_calendar(ranges: pd.DataFrame) -> pd.DataFrame:
    """Expand user date ranges with array operations instead of per-user frames."""

    counts = (
        ranges["last_date"].to_numpy(dtype="datetime64[D]")
        - ranges["first_date"].to_numpy(dtype="datetime64[D]")
    ).astype("timedelta64[D]").astype(np.int64) + 1
    if np.any(counts <= 0):
        raise ValueError("Every user range must contain at least one day")

    nested_offsets = np.arange(int(counts.sum()), dtype=np.int64)
    group_starts = np.repeat(np.cumsum(counts) - counts, counts)
    offsets = nested_offsets - group_starts
    first_days = np.repeat(
        ranges["first_date"].to_numpy(dtype="datetime64[D]"),
        counts,
    )
    dates = first_days + offsets.astype("timedelta64[D]")
    return pd.DataFrame(
        {
            "user_id": np.repeat(ranges["user_id"].to_numpy(), counts),
            "date": pd.to_datetime(dates),
        }
    )


def assign_org_fields(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Assign current-month LDAP fields, then previous-month fallback.

    This replaces a 464k-row Python loop with two keyed pandas merges. The
    fallback is used only when the current snapshot has no row for the user;
    an existing row whose value is ``Unknown`` remains ``Unknown``.
    """

    fields = ["role", "functional_unit", "department", "team", "supervisor"]
    periods = frame["date"].dt.to_period("M")
    user_ids = frame["user_id"].astype(str).to_numpy()
    current_keys = pd.MultiIndex.from_arrays(
        [periods.astype(str).to_numpy(), user_ids],
        names=["month", "user_id"],
    )
    previous_keys = pd.MultiIndex.from_arrays(
        [(periods - 1).astype(str).to_numpy(), user_ids],
        names=["month", "user_id"],
    )
    indexed_lookup = lookup.set_index(["month", "user_id"])
    current_positions = indexed_lookup.index.get_indexer(current_keys)
    previous_positions = indexed_lookup.index.get_indexer(previous_keys)
    use_positions = previous_positions.copy()
    current_present = current_positions >= 0
    use_positions[current_present] = current_positions[current_present]
    safe_positions = np.maximum(use_positions, 0)
    for field in fields:
        lookup_values = indexed_lookup[field].to_numpy(dtype=object)
        values = np.where(
            use_positions >= 0,
            lookup_values[safe_positions],
            "Unknown",
        )
        frame[field] = pd.Series(values, index=frame.index).fillna("Unknown").astype(str)
    return frame


def load_labels() -> pd.DataFrame:
    with tarfile.open(ANSWERS, "r:bz2") as archive:
        names = set(archive.getnames())
        master_file = archive.extractfile("answers/insiders.csv")
        if master_file is None:
            raise RuntimeError("Missing insiders.csv")
        text = io.TextIOWrapper(master_file, encoding="utf-8-sig", newline="")
        master = [
            row
            for row in csv.DictReader(text)
            if row["dataset"].strip() == "4.2"
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
                f"answers/r4.2-{scenario}/{detail}",
            ]
            path = next((item for item in candidates if item in names), None)
            if path is None:
                raise FileNotFoundError(detail)
            raw = archive.extractfile(path)
            if raw is None:
                raise RuntimeError(f"Cannot read {path}")
            detail_text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            for row in csv.reader(detail_text):
                if len(row) < 4:
                    continue
                timestamp = datetime.strptime(
                    row[2].strip(), "%m/%d/%Y %H:%M:%S"
                )
                account = row[3].strip()
                key = (account, str(timestamp.date()))
                label = labels[key]
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
                    for event_type, count in sorted(label["event_types"].items())
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def cached_frame_exists(name: str) -> bool:
    return any(
        (CACHE_DIR / f"{name}.{suffix}").exists()
        for suffix in ("parquet", "pkl")
    )


def read_cached_frame(name: str) -> pd.DataFrame:
    parquet_cache = CACHE_DIR / f"{name}.parquet"
    if parquet_cache.exists():
        return pd.read_parquet(parquet_cache)
    pickle_cache = CACHE_DIR / f"{name}.pkl"
    if pickle_cache.exists():
        return pd.read_pickle(pickle_cache)
    raise FileNotFoundError(f"No cache found for {name!r} in {CACHE_DIR}")


def write_cached_frame(name: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(
        CACHE_DIR / f"{name}.parquet",
        index=False,
        compression="zstd",
    )


def load_user_day_features(
    feature_file: Path = FEATURE_FILE,
) -> pd.DataFrame:
    """Load the user-day table through Parquet when available."""

    parquet_file = feature_file.with_suffix(".parquet")
    if parquet_file.exists():
        return pd.read_parquet(parquet_file)
    return pd.read_csv(feature_file, parse_dates=["date"])


def main(
    *,
    force_refresh: bool = False,
    write_legacy_csv: bool = True,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    primary_pc_cache = CACHE_DIR / "primary_pc.json"
    have_logon_cache = (
        cached_frame_exists("logon")
        and cached_frame_exists("ranges")
        and primary_pc_cache.exists()
    )
    if have_logon_cache and not force_refresh:
        print("Loading cached logon...", flush=True)
        logon = read_cached_frame("logon")
        ranges = read_cached_frame("ranges")
        primary_pc = json.loads(primary_pc_cache.read_text(encoding="utf-8"))
    else:
        print("Loading logon...", flush=True)
        logon, ranges, primary_pc = load_logon()
        write_cached_frame("logon", logon)
        write_cached_frame("ranges", ranges)
        primary_pc_cache.write_text(
            json.dumps(primary_pc, ensure_ascii=False), encoding="utf-8"
        )

    def load_or_build(name: str, builder):
        if cached_frame_exists(name) and not force_refresh:
            print(f"Loading cached {name}...", flush=True)
            return read_cached_frame(name)
        print(f"Loading {name}...", flush=True)
        value = builder()
        write_cached_frame(name, value)
        return value

    device = load_or_build("device", load_device)
    file_copies = load_or_build("file_copies", load_file_copies)
    email = load_or_build("email", load_email)
    http = load_or_build("http", load_http)
    print("Loading LDAP...", flush=True)
    ldap_lookup, months, ldap_missing = load_ldap()
    ldap_missing.to_csv(LDAP_FILE, index=False)

    print("Building calendar...", flush=True)
    frame = build_calendar(ranges)
    for extra in (logon, device, file_copies, email, http):
        frame = frame.merge(extra, on=["user_id", "date"], how="left")

    numeric_counts = [
        "logon_count",
        "logoff_count",
        "after_hours_logon_count",
        "other_pc_logon_count",
        "usb_connect_count",
        "usb_disconnect_count",
        "file_copy_count",
        "archive_copy_count",
        "executable_copy_count",
        "unique_file_extensions",
        "email_count",
        "email_recipient_count",
        "external_recipient_count",
        "email_attachment_count",
        "email_size_bytes",
        "after_hours_email_count",
        "http_count",
        "after_hours_http_count",
        "external_http_count",
        "unique_http_domains",
    ]
    frame[numeric_counts] = frame[numeric_counts].fillna(0)
    integer_columns = [column for column in numeric_counts if column != "email_size_bytes"]
    frame[integer_columns] = frame[integer_columns].astype("int32")
    frame["email_size_bytes"] = frame["email_size_bytes"].astype("int64")
    for column in ("first_logon_minute", "last_logoff_minute"):
        frame[column] = frame[column].fillna(-1).astype("float32")
    frame["active_span_minutes"] = np.where(
        (frame["first_logon_minute"] >= 0)
        & (frame["last_logoff_minute"] >= frame["first_logon_minute"]),
        frame["last_logoff_minute"] - frame["first_logon_minute"],
        -1,
    ).astype("float32")

    frame["weekday"] = frame["date"].dt.weekday.astype("int8")
    frame["is_weekend"] = (frame["weekday"] >= 5).astype("int8")
    frame["is_workday"] = (frame["weekday"] < 5).astype("int8")
    frame["observed_event_count"] = (
        frame["logon_count"]
        + frame["logoff_count"]
        + frame["usb_connect_count"]
        + frame["usb_disconnect_count"]
        + frame["file_copy_count"]
        + frame["email_count"]
        + frame["http_count"]
    ).astype("int64")
    frame["has_any_event"] = (frame["observed_event_count"] > 0).astype("int8")
    frame = assign_org_fields(frame, ldap_lookup)
    frame["primary_pc"] = frame["user_id"].map(primary_pc).fillna("Unknown")
    frame = frame.sort_values(["user_id", "date"]).reset_index(drop=True)
    frame["day_index"] = frame.groupby("user_id").cumcount().astype("int32")

    column_order = [
        "user_id",
        "date",
        "day_index",
        "primary_pc",
        "role",
        "functional_unit",
        "department",
        "team",
        "supervisor",
        "weekday",
        "is_weekend",
        "is_workday",
        "has_any_event",
        "observed_event_count",
        "logon_count",
        "logoff_count",
        "first_logon_minute",
        "last_logoff_minute",
        "active_span_minutes",
        "after_hours_logon_count",
        "other_pc_logon_count",
        "usb_connect_count",
        "usb_disconnect_count",
        "file_copy_count",
        "archive_copy_count",
        "executable_copy_count",
        "unique_file_extensions",
        "email_count",
        "email_recipient_count",
        "external_recipient_count",
        "email_attachment_count",
        "email_size_bytes",
        "after_hours_email_count",
        "http_count",
        "after_hours_http_count",
        "external_http_count",
        "unique_http_domains",
    ]
    frame = frame[column_order]

    print("Loading malicious account-day labels...", flush=True)
    labels = load_or_build("labels", load_labels)
    feature_keys = set(zip(frame["user_id"], frame["date"].dt.strftime("%Y-%m-%d")))
    label_keys = set(zip(labels["user_id"], labels["date"].dt.strftime("%Y-%m-%d")))
    missing_label_keys = sorted(label_keys - feature_keys)

    frame.to_parquet(
        FEATURE_PARQUET,
        index=False,
        compression="zstd",
    )
    if write_legacy_csv:
        frame.to_csv(FEATURE_FILE, index=False, encoding="utf-8")
    else:
        print("Skipping legacy CSV output.", flush=True)
    labels.to_csv(LABEL_FILE, index=False, encoding="utf-8")
    frame.head(200).to_csv(SAMPLE_FILE, index=False, encoding="utf-8")

    summary = {
        "feature_file": str(FEATURE_FILE),
        "feature_parquet_file": str(FEATURE_PARQUET),
        "legacy_csv_written": bool(write_legacy_csv),
        "label_file": str(LABEL_FILE),
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
        "columns": column_order,
        "label_rows": int(len(labels)),
        "label_keys_missing_from_features": len(missing_label_keys),
        "missing_label_keys_sample": missing_label_keys[:20],
        "raw_event_totals": {
            "logon": 854859,
            "device": 405380,
            "file": 445581,
            "email": 2629979,
            "http": 28434423,
        },
        "feature_totals": {
            "logon": int(frame["logon_count"].sum()),
            "logoff": int(frame["logoff_count"].sum()),
            "usb_connect": int(frame["usb_connect_count"].sum()),
            "usb_disconnect": int(frame["usb_disconnect_count"].sum()),
            "file_copy": int(frame["file_copy_count"].sum()),
            "email": int(frame["email_count"].sum()),
            "http": int(frame["http_count"].sum()),
            "all_events": int(frame["observed_event_count"].sum()),
        },
        "zero_event_rows": int((frame["observed_event_count"] == 0).sum()),
        "weekend_rows": int(frame["is_weekend"].sum()),
        "missing_org": {
            field: int((frame[field] == "Unknown").sum())
            for field in ("functional_unit", "department", "team", "supervisor")
        },
        "psychometric_used": False,
        "assumptions": {
            "work_hours_start": WORK_START_HOUR,
            "work_hours_end": WORK_END_HOUR,
            "calendar": "all dates between each user's first and last logon",
            "ldap": "same-month snapshot, fallback to previous month",
            "labels": "observer-account malicious dates only; labels kept in separate CSV",
        },
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CERT user-day features."
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
