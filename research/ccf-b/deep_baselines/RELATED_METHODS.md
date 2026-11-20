# Related method repositories

Date: 2026-11-20.

The following repositories were checked before selecting baselines.

| Method | Repository | License | Intended use |
| --- | --- | --- | --- |
| TS2Vec | <https://github.com/yuezhihan/ts2vec> | MIT | Primary encoder and native control |
| SoftCLT | <https://github.com/seunghan96/softclt> | Not declared in the downloaded snapshot | Direct method family |
| PAI | <https://github.com/pantheon5100/PAI> | Not declared | Amplitude-aware scoring reference |
| Anomaly Transformer | <https://github.com/thuml/Anomaly-Transformer> | MIT | Deep multivariate time-series baseline |
| OmniAnomaly | <https://github.com/NetManAIOps/OmniAnomaly> | MIT | Deep stochastic RNN/VAE baseline; native TensorFlow 1 requires an unavailable legacy environment |
| t-NNCLR | <https://github.com/imics-lab/t-nnclr> | MIT | Nearest-neighbor contrastive representation baseline |
| TS-TCC | <https://github.com/emadeldeen24/TS-TCC> | MIT | Temporal/contextual contrastive representation baseline |
| DCdetector | <https://github.com/ChaoLiZhang/DCdetector> | Not declared | Possible second-wave contrastive deep baseline; audit code before use |

The GitHub repository search also returns several small CERT projects. They
were not selected because most have no license, no reproducible split, or
report implausible headline scores without a usable evaluation protocol.

## Selection

- Use Anomaly Transformer as the first deep detection baseline.
- Use OmniAnomaly as the SMD detection baseline because SMD is part of its
  native benchmark.
- Use t-NNCLR as the first representation-learning baseline because it is the
  closest NNCLR-family alternative to instance-soft.
- Keep TS-TCC and DCdetector as second-wave candidates. They are not required
  for the current result table.
