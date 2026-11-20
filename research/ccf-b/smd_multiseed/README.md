# SMD multi-seed analysis

The frozen extension seeds are `20261201-20261205`.

Raw SMD outputs remain outside the Git release. The lightweight repository
summaries are generated with:

```powershell
python analyze_smd_multiseed.py `
  --t0-root <path-to-t0_matched> `
  --t4-root <path-to-t4_full_soft> `
  --seeds 20261201 20261202 20261203 20261204 20261205 `
  --output-dir results
```

The script reports:

- seed-level macro means;
- seed mean and t interval;
- machine-paired deltas;
- machine-cluster bootstrap intervals.

Machine-cluster bootstrap addresses repeated measurements on the same 28
machines. It does not create new independent machines or datasets.
