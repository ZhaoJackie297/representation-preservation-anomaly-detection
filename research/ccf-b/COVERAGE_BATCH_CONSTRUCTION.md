# Coverage batch construction

Date: 2026-11-20.

This note records the implementation behind the `A0-matched` and `A1-Cov`
training variants used by the CERT 10-seed matrix.

## Relation graphs

The graph inputs are training-period behavior profiles, not labels. The 47-D
CERT profile is the median and 90th percentile of each behavior feature plus
the training-period active fraction. The existing profile implementation is
in `runtime/cert/e2_instance_distance_validation.py`.

Given standardized profiles, let `d_ij` be the Euclidean distance and let
`h` be the median off-diagonal distance. The profile prior is

```text
P_ij = exp(-d_ij^2 / (2 h^2)),  P_ii = 0,
P_ij = P_ij / sum_k P_ik.
```

The three target graphs are:

- `A3`: the symmetric union of each user's top-`m` neighbors in `P`;
- `A5a`: a degree-preserving random graph obtained by stub matching, so its
  degree sequence matches `A3`;
- `A5b`: the top-`m` graph of `P` after a seeded joint row/column permutation.

The implementation is `clue_lds/build_profile_prior.py` and
`clue_lds/build_prior_graphs.py`. The same graph functions were used for the
CERT profile in the coverage experiment.

## Batch table

`phase0/common_coverage_sampler.py` constructs a seed- and stage-dependent
table with:

- batch size 16;
- 8 designed slots selected from the union of A3/A5a/A5b edges;
- 8 random filler users that are disjoint from the designed slots;
- independent coverage gates of at least 80% for each of A3, A5a, and A5b;
- zero fully unexposed users incident to a target edge;
- training length `max(base_steps, number_of_coverage_batches)`.

The table and all of its diagnostics are deterministic for the same
`(seed, stage, graph set, batch size, target coverage)`.

## r4.2 realization

The fixed 10-seed coverage values are in
`research/ccf-b/r4.2_10seed_coverage_summary.csv`. Across the ten seeds:

```text
batches:              1130-1158
A3 coverage:          0.9010-0.9141
A5a coverage:         0.8000-0.8006
A5b coverage:         0.8888-0.9060
zero-exposure users:  0 in all seeds
minimum user exposure: 17-22
```

`A0-official` uses the unconstrained random 16-user shuffle. `A0-matched`
uses the same coverage table as `A1-Cov` but a hard 0/1 instance target.
`A1-Cov` uses the same table and the profile-derived soft instance target,
with temporal-soft disabled and `lambda=0.5`.
