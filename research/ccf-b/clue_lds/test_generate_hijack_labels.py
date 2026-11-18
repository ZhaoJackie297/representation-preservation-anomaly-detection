from __future__ import annotations

from datetime import date, timedelta
import unittest

from generate_hijack_labels import UserStats, generate_labels


def user(
    uid: str,
    *,
    start: date,
    days: int,
    actions: list[str],
    events: int,
) -> UserStats:
    return UserStats(
        uid=uid,
        events=events,
        actions=frozenset(actions),
        active_days=tuple(start + timedelta(days=index) for index in range(days)),
    )


class GenerateHijackLabelsTest(unittest.TestCase):
    def setUp(self) -> None:
        start = date(2022, 1, 1)
        self.users = {
            "u1": user(
                "u1",
                start=start,
                days=100,
                actions=["login", "file", "share"],
                events=300,
            ),
            "u2": user(
                "u2",
                start=start,
                days=100,
                actions=["login", "file", "group"],
                events=280,
            ),
            "u3": user(
                "u3",
                start=start + timedelta(days=10),
                days=90,
                actions=["login", "http"],
                events=210,
            ),
        }

    def test_generates_exact_pair_count(self) -> None:
        rows = generate_labels(
            self.users,
            seed=1,
            pairs=1,
            min_similarity=0.0,
            max_similarity=1.0,
            min_days_before=20,
            min_days_after=10,
            min_events=100,
            min_unique_actions=2,
            profile_cutoff=None,
            min_profile_days=20,
            start_date=date(2022, 3, 1),
            end_date=date(2022, 4, 10),
            event_weight=0.7,
            count_weight=0.3,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {rows[0]["source_uid"], rows[1]["source_uid"]},
            {rows[0]["replacement_uid"], rows[1]["replacement_uid"]},
        )
        self.assertTrue(
            date(2022, 3, 1)
            <= date.fromisoformat(rows[0]["change_date"])
            <= date(2022, 4, 10)
        )

    def test_time_restriction_can_make_generation_impossible(self) -> None:
        with self.assertRaises(ValueError):
            generate_labels(
                self.users,
                seed=1,
                pairs=1,
                min_similarity=0.0,
                max_similarity=1.0,
                min_days_before=20,
                min_days_after=10,
                min_events=100,
                min_unique_actions=2,
                profile_cutoff=None,
                min_profile_days=20,
                start_date=date(2021, 1, 1),
                end_date=date(2021, 1, 10),
                event_weight=0.7,
                count_weight=0.3,
            )


if __name__ == "__main__":
    unittest.main()
