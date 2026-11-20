# Deep-baseline result records

CSV and JSON files in this directory are lightweight, publication-oriented
aggregates. Model weights and score arrays remain in the external experiment
workspace.

Older `*_protocol.json` files preserve machine-local source paths as run-time
provenance. Those paths are historical records, not repository defaults.
Current entry points accept explicit `--data-dir`, `--repo`, and
`--output-dir` arguments.
