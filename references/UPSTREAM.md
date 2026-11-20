# 第三方代码来源

## TS2Vec

- Repository: <https://github.com/yuezhihan/ts2vec>
- Snapshot: `b0088e14a99706c05451316dc6db8d3da9351163`
- License: MIT
- Local path: `references/ts2vec_official/`

## SoftCLT

- Repository: <https://github.com/seunghan96/softclt>
- Commit used in paper experiments: `14c638979b129075d7a1111e9f529b9a275ea394`
- Local path after setup: `references/softclt_repo/`
- The upstream repository did not contain a top-level license file in the downloaded snapshot. SoftCLT is therefore downloaded separately instead of being vendored here.

## Anomaly Transformer

- Repository: <https://github.com/thuml/Anomaly-Transformer>
- Supplementary baseline commit: `b0ee470c8012bff857fb600462aec6209c4a18d9`
- License: MIT
- Local checkout used for adaptation:
  `references/external/Anomaly-Transformer`

## OmniAnomaly

- Repository: <https://github.com/NetManAIOps/OmniAnomaly>
- Supplementary baseline commit: `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`
- License: MIT
- The official model depends on TensorFlow 1.12 and Python 3.5/3.6.
- The repository experiment uses a documented PyTorch GRU-VAE port named
  `OmniAnomaly-adapted`.

## t-NNCLR

- Repository: <https://github.com/imics-lab/t-nnclr>
- Inspected commit: `9c6439991202986a2af108798d5092ea34f0e9ed`
- License: MIT
- Candidate representation baseline for a later CERT comparison.

## TS-TCC

- Repository: <https://github.com/emadeldeen24/TS-TCC>
- Inspected commit: `15bdfd6eced6aeb6c638839828effe1b4b2ea391`
- License: MIT
- Candidate representation baseline for a later CERT comparison.

Use `scripts/setup_third_party.ps1` to prepare the primary TS2Vec/SoftCLT
dependencies. Supplementary benchmark checkouts are kept outside the release
when their size or runtime requirements make vendoring impractical.

