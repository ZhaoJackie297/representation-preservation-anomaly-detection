# Exp S5 lightweight delivery

Archive:

```text
supplement_s5_results_light.zip
SHA256 B979503713E14C77ADB7EC40BBA05716B648FE745936D0231CFB459666C01FBF
```

The archive contains 29 lightweight CSV/JSON files plus `RUN_LOG_S5.md`.
It intentionally excludes model weights (`.pth`), score arrays (`.npz`), and
training temporary files.

Contents:

- Task A main 11-arm matrix and the separate `Hard-L2Reg-b0.2` geometry
  alignment run.
- Task B iForest/T0 exact audit and the 10-seed r4.2/r5.2 hard-random norm
  tables.
- Task C CERT and SMD three-seed radial/norm summaries and by-seed/all-machine
  tables.
- Task D CLUE-LDS and r6.2 unified Hard-Rand/SoftInst-Rand summaries,
  by-seed metrics, and manifests.

Important claim boundaries:

- The historical canonical iForest value `0.0074400944975706` was not exactly
  reproduced by the 500-tree audit; both historical and recomputed values are
  retained verbatim in the audit JSON.
- The added `b0.2` arm is kept separately so the original 11-arm aggregate is
  not silently overwritten.
- Task D uses upstream `SoftTS2Vec` with ten random-batch epochs and fixed
  256-day training chunks for long CLUE sequences. The abandoned slower
  `fit_coverage_table` attempt contributed no result.
- r6.2 has only five threat events and remains stress-audit evidence, not a
  confirmatory external result.
