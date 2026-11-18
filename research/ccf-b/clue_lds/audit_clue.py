from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


def parse_iso_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iter_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at line {line_number}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(
                    f"line {line_number} is not a JSON object"
                )
            yield payload


@dataclass
class ClueAudit:
    total_events: int = 0
    malformed_events: int = 0
    missing_required_fields: Counter[str] = field(default_factory=Counter)
    uid_type_counts: Counter[str] = field(default_factory=Counter)
    event_type_counts: Counter[str] = field(default_factory=Counter)
    role_counts: Counter[str] = field(default_factory=Counter)
    local_ip_counts: Counter[str] = field(default_factory=Counter)
    event_count_by_user: Counter[str] = field(default_factory=Counter)
    actions_by_user: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    active_days_by_user: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    all_dates: set[str] = field(default_factory=set)
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    top_level_keys: Counter[str] = field(default_factory=Counter)
    params_keys: Counter[str] = field(default_factory=Counter)

    def add(self, event: dict[str, Any]) -> None:
        self.total_events += 1
        self.top_level_keys.update(event.keys())

        required = ("time", "uid", "uidType", "type")
        missing = [field_name for field_name in required if field_name not in event]
        if missing:
            self.missing_required_fields.update(missing)
            self.malformed_events += 1
            return

        try:
            timestamp = parse_iso_timestamp(str(event["time"]))
        except (TypeError, ValueError):
            self.malformed_events += 1
            self.missing_required_fields["invalid_time"] += 1
            return

        uid = str(event["uid"])
        uid_type = str(event["uidType"])
        event_type = str(event["type"])
        date_text = timestamp.date().isoformat()

        self.uid_type_counts[uid_type] += 1
        self.event_type_counts[event_type] += 1
        self.event_count_by_user[uid] += 1
        self.actions_by_user[uid].add(event_type)
        self.active_days_by_user[uid].add(date_text)
        self.all_dates.add(date_text)
        if self.first_timestamp is None or timestamp < self.first_timestamp:
            self.first_timestamp = timestamp
        if self.last_timestamp is None or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp

        role = event.get("role")
        self.role_counts[str(role) if role is not None else "<missing>"] += 1
        local_ip = event.get("isLocalIP")
        self.local_ip_counts[
            str(local_ip) if local_ip is not None else "<missing>"
        ] += 1

        params = event.get("params")
        if isinstance(params, dict):
            self.params_keys.update(params.keys())

    def summary(self) -> dict[str, Any]:
        eligible_user_days = sum(
            len(days) for days in self.active_days_by_user.values()
        )
        return {
            "total_events": self.total_events,
            "malformed_events": self.malformed_events,
            "missing_required_fields": dict(
                self.missing_required_fields
            ),
            "unique_users": len(self.event_count_by_user),
            "unique_dates": len(self.all_dates),
            "first_timestamp": (
                self.first_timestamp.isoformat()
                if self.first_timestamp is not None
                else None
            ),
            "last_timestamp": (
                self.last_timestamp.isoformat()
                if self.last_timestamp is not None
                else None
            ),
            "user_days": eligible_user_days,
            "uid_type_counts": dict(self.uid_type_counts),
            "event_type_counts": dict(
                self.event_type_counts.most_common()
            ),
            "role_counts": dict(self.role_counts),
            "local_ip_counts": dict(self.local_ip_counts),
            "top_level_keys": dict(self.top_level_keys),
            "params_keys": dict(self.params_keys.most_common()),
            "top_users_by_events": [
                {"uid": uid, "events": count}
                for uid, count in self.event_count_by_user.most_common(50)
            ],
            "dates": sorted(self.all_dates),
        }

    def user_stats(self) -> dict[str, Any]:
        return {
            uid: {
                "events": self.event_count_by_user[uid],
                "actions": sorted(self.actions_by_user[uid]),
                "active_days": sorted(self.active_days_by_user[uid]),
            }
            for uid in sorted(self.event_count_by_user)
        }


def audit_file(
    path: Path,
    *,
    max_events: int | None = None,
    progress_every: int = 1_000_000,
) -> ClueAudit:
    audit = ClueAudit()
    for event in iter_json_lines(path):
        audit.add(event)
        if progress_every > 0 and audit.total_events % progress_every == 0:
            print(
                json.dumps(
                    {
                        "processed_events": audit.total_events,
                        "users": len(audit.event_count_by_user),
                    }
                ),
                flush=True,
            )
        if max_events is not None and audit.total_events >= max_events:
            break
    return audit


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-audit the CLUE-LDS JSON-lines file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-stats-output", type=Path)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    args = parser.parse_args()

    audit = audit_file(
        args.input,
        max_events=args.max_events,
        progress_every=args.progress_every,
    )
    write_json(args.output, audit.summary())
    if args.user_stats_output is not None:
        write_json(args.user_stats_output, audit.user_stats())


if __name__ == "__main__":
    main()
