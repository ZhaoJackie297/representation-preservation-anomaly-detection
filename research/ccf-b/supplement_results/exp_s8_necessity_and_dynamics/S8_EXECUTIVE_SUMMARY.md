# Exp S8 Executive Summary

Date: 2026-11-23 (Asia/Shanghai). Seeds: `20261116-20261120`.

## 核心答案

### (1) Soft-Sphere 是否使增益消失

在三个温度与三类关键评分器上，Soft-Sphere 相对 SoftInst 的 PR-AUC 平均配对差值为 -0.003272；相对同温度 Hard-Sphere 的平均配对差值为 0.002285。 逐项看，切断未归一化模长后相对 SoftInst 的点级增益均未保留。

| Candidate | Comparison | Scorer | Delta PR-AUC | Wins | p | Holm p |
|---|---|---|---:|---:|---:|---:|
| Soft-Sphere-t0.1 | t0.1_vs_softinst | bank_euclidean | -0.002239 | 0/5 | 0.0245 | 0.049 |
| Soft-Sphere-t0.1 | t0.1_vs_hardsphere | bank_euclidean | -0.000465 | 1/5 | 0.392 | 0.392 |
| Soft-Sphere-t0.2 | t0.2_vs_softinst | bank_euclidean | -0.002045 | 0/5 | 0.0158 | 0.0473 |
| Soft-Sphere-t0.2 | t0.2_vs_hardsphere | bank_euclidean | -0.000611 | 1/5 | 0.289 | 0.289 |
| Soft-Sphere-t0.5 | t0.5_vs_softinst | bank_euclidean | -0.001520 | 0/5 | 0.0408 | 0.049 |
| Soft-Sphere-t0.5 | t0.5_vs_hardsphere | bank_euclidean | 0.003928 | 5/5 | 0.00227 | 0.00227 |
| Soft-Sphere-t0.1 | t0.1_vs_softinst | bank_cosine | -0.006983 | 0/5 | 0.0175 | 0.0349 |
| Soft-Sphere-t0.1 | t0.1_vs_hardsphere | bank_cosine | 0.000346 | 3/5 | 0.818 | 0.818 |
| Soft-Sphere-t0.2 | t0.2_vs_softinst | bank_cosine | -0.005952 | 0/5 | 0.00961 | 0.0288 |
| Soft-Sphere-t0.2 | t0.2_vs_hardsphere | bank_cosine | 0.002703 | 5/5 | 0.0111 | 0.0111 |
| Soft-Sphere-t0.5 | t0.5_vs_softinst | bank_cosine | -0.003774 | 1/5 | 0.116 | 0.116 |
| Soft-Sphere-t0.5 | t0.5_vs_hardsphere | bank_cosine | 0.009829 | 5/5 | 0.000128 | 0.000128 |
| Soft-Sphere-t0.1 | t0.1_vs_softinst | pai_a1 | -0.002946 | 0/5 | 0.00453 | 0.00906 |
| Soft-Sphere-t0.1 | t0.1_vs_hardsphere | pai_a1 | -0.000680 | 1/5 | 0.169 | 0.169 |
| Soft-Sphere-t0.2 | t0.2_vs_softinst | pai_a1 | -0.002221 | 0/5 | 0.0018 | 0.00541 |
| Soft-Sphere-t0.2 | t0.2_vs_hardsphere | pai_a1 | -0.000018 | 2/5 | 0.982 | 0.982 |
| Soft-Sphere-t0.5 | t0.5_vs_softinst | pai_a1 | -0.001762 | 0/5 | 0.0233 | 0.0233 |
| Soft-Sphere-t0.5 | t0.5_vs_hardsphere | pai_a1 | 0.005530 | 5/5 | 0.00366 | 0.00366 |

### (2) 为什么等范数约束在余弦上仍不足

Hard-L2Reg-b0.20 的余弦 PR-AUC 为 0.0143，SoftInst 为 0.0167；二者欧氏 PR-AUC 分别为 0.0140 和 0.0141。Hard-L2Reg 的单位方向有效秩为 12.6814，SoftInst 为 13.8013；单位球平均成对余弦分别为 0.0234 和 0.3745。 这与软权重保留更高方向维数、缓解纯径向收缩后方向坍缩的解释一致。

| Arm | Mean norm | Unit effective rank | Mean pairwise cosine | Cosine NN margin | Cosine PR-AUC | Euclidean PR-AUC | PAI-Mean PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| PermSoft-Rand | 1.2939 | 13.5247 | 0.3958 | 0.0161 | 0.0173 | 0.0132 | 0.0133 |
| UniformSoft-Rand | 1.2965 | 13.5352 | 0.3930 | 0.0161 | 0.0163 | 0.0129 | 0.0131 |
| Hard-L2Reg-b0.20 | 1.3745 | 12.6814 | 0.0234 | 0.0213 | 0.0143 | 0.0140 | 0.0154 |
| SoftInst-Rand | 1.3836 | 13.8013 | 0.3745 | 0.0164 | 0.0167 | 0.0141 | 0.0141 |
| Hard-Rand | 3.4892 | 8.7535 | 0.1315 | 0.0155 | 0.0114 | 0.0113 | 0.0112 |

| Arm | Norm step 0 | Norm step 620 | Unit ER step 0 | Unit ER step 620 | Val cosine PR step 0 | Val cosine PR step 620 |
|---|---:|---:|---:|---:|---:|---:|
| Hard-Rand | 2.7015 | 3.4683 | 6.2293 | 9.0386 | 0.1050 | 0.1060 |
| Hard-L2Reg-b0.20 | 2.7015 | 1.3642 | 6.2293 | 12.8910 | 0.1050 | 0.1061 |
| Hard-L2Reg-b1.00 | 2.7015 | 0.3386 | 6.2293 | 16.3419 | 0.1050 | 0.1003 |
| PermSoft-Rand | 2.7015 | 1.2896 | 6.2293 | 13.8789 | 0.1050 | 0.1047 |
| SoftInst-Rand | 2.7015 | 1.3805 | 6.2293 | 14.0734 | 0.1050 | 0.1057 |
| Soft-Sphere-t0.1 | 2.7015 | 5.4606 | 6.2293 | 14.2134 | 0.1050 | 0.1046 |

### (3) Alpha 与 tau_mult 的剂量-响应形状

- Alpha: norm=`monotonic_decreasing`, entropy=`monotonic_increasing`, unit_rank=`monotonic_increasing`, cosine_PR=`non_monotonic`.
- Tau multiplier: norm=`monotonic_increasing`, entropy=`monotonic_decreasing`, unit_rank=`monotonic_decreasing`, cosine_PR=`inverted_U`.

| Alpha | Target entropy | Offdiag mass ratio | Mean norm | Unit ER | Cosine PR-AUC | Euclidean PR-AUC | PAI-Mean PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.0000 | 0.0000 | 3.4892 | 8.9738 | 0.0114 | 0.0113 | 0.0112 |
| 0.050 | 6.7661 | 0.9638 | 2.2586 | 12.1676 | 0.0139 | 0.0123 | 0.0124 |
| 0.150 | 6.8472 | 0.9876 | 1.8111 | 12.8042 | 0.0151 | 0.0132 | 0.0129 |
| 0.250 | 6.8595 | 0.9926 | 1.6048 | 13.2990 | 0.0164 | 0.0135 | 0.0133 |
| 0.500 | 6.8666 | 0.9963 | 1.3836 | 14.1044 | 0.0167 | 0.0141 | 0.0141 |
| 0.750 | 6.8682 | 0.9975 | 1.2972 | 14.5292 | 0.0165 | 0.0141 | 0.0145 |
| 1.000 | 6.8688 | 0.9981 | 1.2552 | 14.7741 | 0.0167 | 0.0143 | 0.0145 |

| Tau multiplier | Target entropy | Offdiag mass ratio | Mean norm | Unit ER | Cosine PR-AUC | Euclidean PR-AUC | PAI-Mean PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.250 | 6.9055 | 0.9977 | 1.1767 | 15.0483 | 0.0149 | 0.0131 | 0.0129 |
| 0.500 | 6.8988 | 0.9973 | 1.2384 | 14.8367 | 0.0169 | 0.0136 | 0.0139 |
| 1.000 | 6.8666 | 0.9963 | 1.3836 | 14.1044 | 0.0167 | 0.0141 | 0.0141 |
| 2.000 | 6.7101 | 0.9928 | 1.6979 | 12.4424 | 0.0158 | 0.0138 | 0.0138 |
| 4.000 | 6.0703 | 0.9772 | 2.2634 | 11.3683 | 0.0129 | 0.0136 | 0.0126 |

## Reproduction and integrity

- S5a 重合 arm 的逐种子复现审计最大绝对误差为 1.667e-01；无轨迹的同命令直跑也出现同类 GPU 重训数值差异，因此最终基准指标使用冻结 S5a，轨迹 run 仅用于训练动力学。
- Soft-Sphere runs assert exactly 620 soft-loss calls, at least 1,240 tensor-gradient hook calls, nonzero off-diagonal target mass, and post-normalization norm 1.0 within `1e-4`.
- Training-trajectory snapshots save/restore Python, NumPy, CPU Torch, and CUDA RNG states.
- The fast trajectory monitor uses all validation positive active days plus 5,000 deterministic negative active days; final performance uses the full test set.

## Interpretation boundary

These results are causal attribution evidence for CERT r4.2 under the frozen 620-step protocol. They do not establish the same mechanism on CLUE-LDS, SMD, or live bank production data. Watchlist is absent from every encoder, memory bank, and score path.
