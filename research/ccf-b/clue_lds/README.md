# Profile graph construction

These scripts build the training-only profile prior and the three graph
controls used by the coverage batch construction:

- `build_profile_prior.py`: profile features, Gaussian profile prior, and
  entropy diagnostics;
- `build_prior_graphs.py`: A3 top-m edges, the A5a degree-preserving random
  graph, and the A5b permuted-profile graph;
- `test_profile_graphs.py`: unit tests for the graph construction.

Run the tests with:

```powershell
python -m unittest -v test_profile_graphs.py
```

The batch-level coverage algorithm is in
`../phase0/common_coverage_sampler.py`.
