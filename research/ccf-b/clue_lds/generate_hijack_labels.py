from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import random
from typing import Any


@dataclass(frozen=True)
class UserStats:
    uid: str
    events: int
    actions: frozenset[str]
    active_days: tuple[date, ...]

    @property
    def avg_events_per_day(self) -> float:
        return self.events / max(len(self.active_days), 1)


def load_user_stats(path: Path) -> dict[str, UserStats]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    users: dict[str, UserStats] = {}
    for uid, values in payload.items():
        days = tuple(
            sorted(date.fromisoformat(day) for day in values["active_days"])
        )
        users[uid] = UserStats(
            uid=uid,
            events=int(values["events"]),
            actions=frozenset(str(action) for action in values["actions"]),
            active_days=days,
        )
    return users


def action_jaccard(left: UserStats, right: UserStats) -> float:
    union = left.actions | right.actions
    if not union:
        return 1.0
    return len(left.actions & right.actions) / len(union)


def count_similarity(left: UserStats, right: UserStats) -> float:
    low = min(left.avg_events_per_day, right.avg_events_per_day)
    high = max(left.avg_events_per_day, right.avg_events_per_day)
    return low / high if high > 0 else 1.0


def similarity(
    left: UserStats,
    right: UserStats,
    *,
    event_weight: float = 0.7,
    count_weight: float = 0.3,
) -> float:
    total = event_weight + count_weight
    if total <= 0:
        raise ValueError("similarity weights must sum to a positive value")
    return (
        event_weight * action_jaccard(left, right)
        + count_weight * count_similarity(left, right)
    ) / total


def valid_switch_dates(
    left: UserStats,
    right: UserStats,
    *,
    min_days_before: int,
    min_days_after: int,
    start_date: date | None,
    end_date: date | None,
) -> list[date]:
    if len(left.active_days) <= min_days_before + min_days_after:
        return []
    if len(right.active_days) <= min_days_before + min_days_after:
        return []

    left_days = set(left.active_days)
    right_days = set(right.active_days)
    common = sorted(left_days & right_days)
    valid: list[date] = []
    for candidate in common:
        if start_date is not None and candidate < start_date:
            continue
        if end_date is not None and candidate > end_date:
            continue
        left_before = sum(day < candidate for day in left.active_days)
        left_after = sum(day >= candidate for day in left.active_days)
        right_before = sum(day < candidate for day in right.active_days)
        right_after = sum(day >= candidate for day in right.active_days)
        if (
            left_before >= min_days_before
            and right_before >= min_days_before
            and left_after >= min_days_after
            and right_after >= min_days_after
        ):
            valid.append(candidate)
    return valid


def generate_labels(
    users: dict[str, UserStats],
    *,
    seed: int,
    pairs: int,
    min_similarity: float,
    max_similarity: float,
    min_days_before: int,
    min_days_after: int,
    min_events: int,
    min_unique_actions: int,
    profile_cutoff: date | None,
    min_profile_days: int,
    start_date: date | None,
    end_date: date | None,
    event_weight: float,
    count_weight: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    eligible = sorted(
        uid
        for uid, stats in users.items()
        if stats.events >= min_events
        and len(stats.actions) >= min_unique_actions
        and len(stats.active_days) > min_days_before + min_days_after
        and (
            profile_cutoff is None
            or sum(
                day < profile_cutoff for day in stats.active_days
            )
            >= min_profile_days
        )
    )
    available = set(eligible)
    rows: list[dict[str, Any]] = []

    while len(rows) // 2 < pairs:
        if len(available) < 2:
            raise ValueError(
                f"only {len(rows) // 2} eligible pairs found"
            )
        first_uid = rng.choice(sorted(available))
        available.remove(first_uid)
        first = users[first_uid]

        candidates: list[
            tuple[str, float, list[date]]
        ] = []
        for second_uid in sorted(available):
            second = users[second_uid]
            score = similarity(
                first,
                second,
                event_weight=event_weight,
                count_weight=count_weight,
            )
            if not (min_similarity <= score <= max_similarity):
                continue
            switch_dates = valid_switch_dates(
                first,
                second,
                min_days_before=min_days_before,
                min_days_after=min_days_after,
                start_date=start_date,
                end_date=end_date,
            )
            if switch_dates:
                candidates.append((second_uid, score, switch_dates))

        if not candidates:
            continue
        second_uid, score, switch_dates = rng.choice(candidates)
        switch_day = rng.choice(switch_dates)
        available.remove(second_uid)

        for source, replacement in (
            (first_uid, second_uid),
            (second_uid, first_uid),
        ):
            rows.append(
                {
                    "source_uid": source,
                    "replacement_uid": replacement,
                    "change_date": switch_day.isoformat(),
                    "similarity": score,
                    "source_events": users[source].events,
                    "replacement_events": users[replacement].events,
                }
            )
    return rows


def write_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_uid",
                "replacement_uid",
                "change_date",
                "similarity",
                "source_events",
                "replacement_events",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--min-sim", type=float, default=0.1)
    parser.add_argument("--max-sim", type=float, default=0.6)
    parser.add_argument("--days-before", type=int, default=35)
    parser.add_argument("--days-after", type=int, default=30)
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--min-unique", type=int, default=4)
    parser.add_argument("--profile-cutoff")
    parser.add_argument("--min-profile-days", type=int, default=20)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--event-weight", type=float, default=0.7)
    parser.add_argument("--count-weight", type=float, default=0.3)
    args = parser.parse_args()

    labels = generate_labels(
        load_user_stats(args.user_stats),
        seed=args.seed,
        pairs=args.pairs,
        min_similarity=args.min_sim,
        max_similarity=args.max_sim,
        min_days_before=args.days_before,
        min_days_after=args.days_after,
        min_events=args.min_events,
        min_unique_actions=args.min_unique,
        profile_cutoff=date.fromisoformat(args.profile_cutoff)
        if args.profile_cutoff
        else None,
        min_profile_days=args.min_profile_days,
        start_date=date.fromisoformat(args.start_date)
        if args.start_date
        else None,
        end_date=date.fromisoformat(args.end_date)
        if args.end_date
        else None,
        event_weight=args.event_weight,
        count_weight=args.count_weight,
    )
    write_labels(args.output, labels)


if __name__ == "__main__":
    main()
