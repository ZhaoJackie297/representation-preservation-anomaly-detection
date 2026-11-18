from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
R1_DIR = ROOT / "r1_extracted" / "r1"
OUT_DIR = ROOT / "processed"
OUT_FILE = OUT_DIR / "r1_user_day.csv"
SAMPLE_FILE = OUT_DIR / "r1_user_day_sample.csv"
SUMMARY_FILE = OUT_DIR / "r1_user_day_summary.json"

WORK_START_HOUR = 8
WORK_END_HOUR = 18


def split_event_user(series: pd.Series) -> pd.Series:
    return series.astype("string").str.rsplit("/", n=1).str[-1]


def parse_event_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%m/%d/%Y %H:%M:%S", errors="raise")


def add_hour_flags(frame: pd.DataFrame) -> pd.DataFrame:
    hour = frame["timestamp"].dt.hour
    frame["is_after_hours"] = (hour < WORK_START_HOUR) | (hour >= WORK_END_HOUR)
    return frame


def load_logon_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = R1_DIR / "logon.csv"
    frame = pd.read_csv(
        path,
        usecols=["date", "user", "activity"],
        dtype={"user": "string", "activity": "string"},
    )
    frame["timestamp"] = parse_event_date(frame["date"])
    frame["user_id"] = split_event_user(frame["user"])
    frame["date"] = frame["timestamp"].dt.normalize()
    frame = add_hour_flags(frame)
    frame["is_logon"] = frame["activity"].eq("Logon")
    frame["is_logoff"] = frame["activity"].eq("Logoff")
    frame["minute_of_day"] = (
        frame["timestamp"].dt.hour * 60
        + frame["timestamp"].dt.minute
        + frame["timestamp"].dt.second / 60.0
    )
    frame["logon_minute_value"] = np.where(
        frame["is_logon"], frame["minute_of_day"], np.nan
    )
    frame["logoff_minute_value"] = np.where(
        frame["is_logoff"], frame["minute_of_day"], np.nan
    )
    frame["after_hours_logon"] = frame["is_logon"] & frame["is_after_hours"]

    grouped = (
        frame.groupby(["user_id", "date"], sort=False)
        .agg(
            logon_count=("is_logon", "sum"),
            logoff_count=("is_logoff", "sum"),
            first_logon_minute=("logon_minute_value", "min"),
            last_logoff_minute=("logoff_minute_value", "max"),
            after_hours_logon_count=("after_hours_logon", "sum"),
        )
        .reset_index()
    )

    ranges = (
        frame.groupby("user_id", sort=True)["date"]
        .agg(["min", "max"])
        .rename(columns={"min": "first_date", "max": "last_date"})
        .reset_index()
    )
    return grouped, ranges


def load_device_features() -> pd.DataFrame:
    path = R1_DIR / "device.csv"
    frame = pd.read_csv(
        path,
        usecols=["date", "user", "activity"],
        dtype={"user": "string", "activity": "string"},
    )
    frame["timestamp"] = parse_event_date(frame["date"])
    frame["user_id"] = split_event_user(frame["user"])
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["usb_connect"] = frame["activity"].eq("Connect")
    frame["usb_disconnect"] = frame["activity"].eq("Disconnect")
    return (
        frame.groupby(["user_id", "date"], sort=False)
        .agg(
            usb_connect_count=("usb_connect", "sum"),
            usb_disconnect_count=("usb_disconnect", "sum"),
        )
        .reset_index()
    )


def load_http_features() -> pd.DataFrame:
    path = R1_DIR / "http.csv"
    columns = ["id", "date", "user", "pc", "url"]
    frame = pd.read_csv(
        path,
        header=None,
        names=columns,
        usecols=["date", "user", "url"],
        dtype={"user": "string", "url": "string"},
    )
    frame["timestamp"] = parse_event_date(frame["date"])
    frame["user_id"] = split_event_user(frame["user"])
    frame["date"] = frame["timestamp"].dt.normalize()
    frame = add_hour_flags(frame)
    frame["http_domain"] = (
        frame["url"]
        .str.extract(r"^[a-zA-Z]+://([^/]+)", expand=False)
        .str.lower()
        .fillna("")
    )
    frame["is_internal_domain"] = frame["http_domain"].eq("dtaa.com") | frame[
        "http_domain"
    ].str.endswith(".dtaa.com")
    frame["external_http"] = ~frame["is_internal_domain"]
    frame["after_hours_http"] = frame["is_after_hours"]

    return (
        frame.groupby(["user_id", "date"], sort=False)
        .agg(
            http_count=("url", "size"),
            unique_http_domains=("http_domain", "nunique"),
            external_http_count=("external_http", "sum"),
            after_hours_http_count=("after_hours_http", "sum"),
        )
        .reset_index()
    )


def load_ldap_roles() -> tuple[dict[tuple[str, str], str], list[str]]:
    role_lookup: dict[tuple[str, str], str] = {}
    months = []
    for path in sorted((R1_DIR / "LDAP").glob("*.csv")):
        month = path.stem
        months.append(month)
        frame = pd.read_csv(path, dtype="string")
        frame["user_id"] = frame["user_id"].str.strip()
        frame["Role"] = frame["Role"].fillna("Unknown").str.strip()
        for user_id, role in frame[["user_id", "Role"]].itertuples(index=False):
            role_lookup[(month, user_id)] = role
    return role_lookup, months


def build_calendar_grid(ranges: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in ranges.itertuples(index=False):
        dates = pd.date_range(row.first_date, row.last_date, freq="D")
        frames.append(
            pd.DataFrame({"user_id": row.user_id, "date": dates})
        )
    return pd.concat(frames, ignore_index=True)


def previous_month(period: pd.Period) -> pd.Period:
    return period - 1


def assign_roles(
    table: pd.DataFrame,
    role_lookup: dict[tuple[str, str], str],
    available_months: list[str],
) -> tuple[pd.DataFrame, int]:
    available = set(available_months)
    periods = table["date"].dt.to_period("M")
    roles = []
    missing = 0
    for user_id, period in zip(table["user_id"], periods, strict=True):
        current_key = (str(period), user_id)
        prev_key = (str(previous_month(period)), user_id)
        role = role_lookup.get(current_key)
        if role is None and str(previous_month(period)) in available:
            role = role_lookup.get(prev_key)
        if role is None:
            role = "Unknown"
            missing += 1
        roles.append(role)
    table["role"] = roles
    return table, missing


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading logon events...", flush=True)
    logon, ranges = load_logon_features()
    print(f"  grouped rows: {len(logon):,}", flush=True)

    print("Reading device events...", flush=True)
    device = load_device_features()
    print(f"  grouped rows: {len(device):,}", flush=True)

    print("Reading HTTP events...", flush=True)
    http = load_http_features()
    print(f"  grouped rows: {len(http):,}", flush=True)

    print("Building complete calendar grid...", flush=True)
    user_day = build_calendar_grid(ranges)
    print(f"  calendar rows: {len(user_day):,}", flush=True)

    for frame in (logon, device, http):
        user_day = user_day.merge(frame, on=["user_id", "date"], how="left")

    numeric_counts = [
        "logon_count",
        "logoff_count",
        "after_hours_logon_count",
        "usb_connect_count",
        "usb_disconnect_count",
        "http_count",
        "unique_http_domains",
        "external_http_count",
        "after_hours_http_count",
    ]
    user_day[numeric_counts] = user_day[numeric_counts].fillna(0).astype("int32")
    for column in ("first_logon_minute", "last_logoff_minute"):
        user_day[column] = user_day[column].fillna(-1).astype("float32")

    user_day["weekday"] = user_day["date"].dt.weekday.astype("int8")
    user_day["is_weekend"] = (user_day["weekday"] >= 5).astype("int8")
    user_day["is_workday"] = (user_day["weekday"] < 5).astype("int8")
    user_day["active_span_minutes"] = np.where(
        (user_day["first_logon_minute"] >= 0)
        & (user_day["last_logoff_minute"] >= user_day["first_logon_minute"]),
        user_day["last_logoff_minute"] - user_day["first_logon_minute"],
        -1,
    ).astype("float32")
    user_day["observed_event_count"] = (
        user_day["logon_count"]
        + user_day["logoff_count"]
        + user_day["usb_connect_count"]
        + user_day["usb_disconnect_count"]
        + user_day["http_count"]
    ).astype("int32")
    user_day["has_any_event"] = (user_day["observed_event_count"] > 0).astype("int8")

    print("Assigning monthly LDAP roles...", flush=True)
    role_lookup, months = load_ldap_roles()
    user_day, missing_roles = assign_roles(user_day, role_lookup, months)

    user_day = user_day.sort_values(["user_id", "date"]).reset_index(drop=True)
    user_day["day_index"] = user_day.groupby("user_id").cumcount().astype("int32")

    column_order = [
        "user_id",
        "date",
        "day_index",
        "role",
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
        "usb_connect_count",
        "usb_disconnect_count",
        "http_count",
        "unique_http_domains",
        "external_http_count",
        "after_hours_http_count",
    ]
    user_day = user_day[column_order]
    user_day.to_csv(OUT_FILE, index=False, encoding="utf-8")
    user_day.head(200).to_csv(SAMPLE_FILE, index=False, encoding="utf-8")

    role_counts = user_day["role"].value_counts().to_dict()
    summary = {
        "output": str(OUT_FILE),
        "rows": int(len(user_day)),
        "users": int(user_day["user_id"].nunique()),
        "date_min": str(user_day["date"].min().date()),
        "date_max": str(user_day["date"].max().date()),
        "weekend_rows": int(user_day["is_weekend"].sum()),
        "zero_event_rows": int((user_day["observed_event_count"] == 0).sum()),
        "missing_role_rows": int(missing_roles),
        "role_counts": role_counts,
        "columns": column_order,
        "assumptions": {
            "work_hours_start": WORK_START_HOUR,
            "work_hours_end": WORK_END_HOUR,
            "after_hours": "before start hour or at/after end hour",
            "date_grid": "all calendar days between each user's first and last logon date",
            "role_mapping": "same monthly LDAP snapshot, fallback to previous month",
        },
        "labels": "r1 has no malicious scenario labels",
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
