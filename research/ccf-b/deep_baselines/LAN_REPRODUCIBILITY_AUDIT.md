# LAN reproducibility audit

Date: 2026-11-20.

Status: not run; protocol boundary recorded.

## Reference

The project bibliography identifies:

> Cai X., Wang Y., Xu S., Li H., et al. LAN: Learning Adaptive Neighbors for
> Real-Time Insider Threat Detection. arXiv:2403.09209, 2024.

LAN is a CERT r4.2/r5.2 insider-threat method and is not a generic
multivariate sensor baseline.

## Availability check

- No LAN implementation was found in the local reference workspace.
- GitHub access failed during this audit with an SSL/TLS handshake error, so a
  remote implementation could not be fetched or pinned.
- No result is fabricated or inferred from the paper's reported numbers.

## Protocol gap

LAN operates on activity-level behavior. The current CERT pipeline uses
27-dimensional daily user features. A fair comparison requires:

1. the same r4.2 users and train/validation/test boundaries;
2. activity logs retained as ordered event sequences rather than collapsed
   only into daily counts;
3. LAN scoring aggregated to a user-day score without using test labels;
4. the same Top-K grid and seven-day cooldown after aggregation;
5. tuning only on training/validation data and the same compute budget.

Without an executable implementation, this cannot be represented as a
same-protocol baseline. Replacing LAN with a daily feature table would create
a new adaptation whose result must not be labelled "LAN".

## Decision

The current paper retains LAN in related work and the unpaired-domain-baseline
limitation. No LAN number is included in result tables.
