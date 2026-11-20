from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def record_run_protocol(
    path: Path,
    protocol: Mapping[str, object],
    seeds: Iterable[int],
) -> dict[str, object]:
    """Merge the current invocation into a durable multi-run protocol."""
    current = dict(protocol)
    current_seeds = sorted({int(seed) for seed in seeds})
    previous: dict[str, object] = {}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))

    completed = {
        int(seed) for seed in previous.get("seeds", [])
    } | set(current_seeds)
    invocations = [
        {
            "seeds": sorted({int(seed) for seed in row["seeds"]}),
            **(
                {"source": str(row["source"])}
                if "source" in row
                else {}
            ),
        }
        for row in previous.get("seed_invocations", [])
    ]
    if previous.get("seeds") and "seed_invocations" not in previous:
        legacy_seeds = sorted({int(seed) for seed in previous["seeds"]})
        if legacy_seeds not in [row["seeds"] for row in invocations]:
            invocations.append(
                {
                    "seeds": legacy_seeds,
                    "source": "previous_protocol",
                }
            )
    if current_seeds not in [row["seeds"] for row in invocations]:
        invocations.append({"seeds": current_seeds})

    current["seeds"] = sorted(completed)
    current["seed_invocations"] = invocations
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return current
