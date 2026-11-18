from __future__ import annotations

import argparse
import json
import math

import numpy as np


def ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0,1]")
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    state = 0.0
    for index, value in enumerate(values):
        state = alpha * value + (1.0 - alpha) * state
        result[index] = state
    return result


def cusum(values: np.ndarray, drift: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    state = 0.0
    for index, value in enumerate(values):
        state = max(0.0, state + value - drift)
        result[index] = state
    return result


def ewma_mean_delay(alpha: float, threshold: float, shift: float) -> float:
    """Deterministic mean-crossing approximation for a step shift."""
    if shift <= 0 or threshold <= 0:
        return 0.0
    if shift <= threshold:
        return math.inf
    return math.log(max(1.0 - threshold / shift, 1e-12)) / math.log(
        1.0 - alpha
    )


def cusum_mean_delay(drift: float, threshold: float, shift: float) -> float:
    if shift <= drift or threshold <= 0:
        return 0.0
    return threshold / (shift - drift)


def simulate_method(
    *,
    method: str,
    parameter: float,
    threshold: float,
    shift: float,
    repetitions: int,
    horizon: int,
    seed: int,
    change_index: int,
) -> dict[str, float]:
    if method not in {"ewma", "cusum"}:
        raise ValueError(f"unknown method: {method}")
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(repetitions, horizon))
    shifted = rng.normal(size=(repetitions, horizon))
    if change_index >= horizon:
        raise ValueError("change_index must be smaller than horizon")
    shifted[:, change_index:] += shift

    def statistic(sequence: np.ndarray) -> np.ndarray:
        if method == "ewma":
            return ewma(sequence, parameter)
        return cusum(sequence, parameter)

    normal_stat = np.stack([statistic(row) for row in normal])
    shifted_stat = np.stack([statistic(row) for row in shifted])
    false_alarm = np.any(normal_stat >= threshold, axis=1)
    detection = np.full(repetitions, horizon, dtype=np.float64)
    for index, row in enumerate(shifted_stat[:, change_index:]):
        hits = np.flatnonzero(row >= threshold)
        if len(hits):
            detection[index] = float(hits[0])
    return {
        "false_alarm_rate": float(false_alarm.mean()),
        "mean_detection_delay": float(detection.mean()),
        "median_detection_delay": float(np.median(detection)),
    }


def calibrate_threshold(
    *,
    method: str,
    parameter: float,
    repetitions: int,
    horizon: int,
    seed: int,
    target_false_alarm: float,
) -> float:
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(repetitions, horizon))
    if method == "ewma":
        values = np.stack([ewma(row, parameter) for row in normal]).ravel()
    elif method == "cusum":
        values = np.stack([cusum(row, parameter) for row in normal]).ravel()
    else:
        raise ValueError(f"unknown method: {method}")
    # target_false_alarm is the probability of at least one alert in a
    # normal sequence of length ``horizon``. Convert it to a per-timestep
    # exceedance probability before taking the empirical quantile.
    per_timestep = 1.0 - (1.0 - target_false_alarm) ** (1.0 / horizon)
    return float(np.quantile(values, 1.0 - per_timestep))


def run_grid(
    *,
    repetitions: int = 2000,
    horizon: int = 100,
    change_index: int = 20,
    shift: float = 1.0,
    target_false_alarm: float = 0.01,
    seed: int = 20261118,
) -> dict[str, object]:
    rows = []
    for method, parameters in (
        ("ewma", (0.05, 0.1, 0.2, 0.5)),
        ("cusum", (0.25, 0.5, 1.0)),
    ):
        for parameter in parameters:
            threshold = calibrate_threshold(
                method=method,
                parameter=parameter,
                repetitions=repetitions,
                horizon=horizon,
                seed=seed + int(parameter * 1000),
                target_false_alarm=target_false_alarm,
            )
            empirical = simulate_method(
                method=method,
                parameter=parameter,
                threshold=threshold,
                shift=shift,
                repetitions=repetitions,
                horizon=horizon,
                seed=seed + 1 + int(parameter * 1000),
                change_index=change_index,
            )
            analytic_delay = (
                ewma_mean_delay(parameter, threshold, shift)
                if method == "ewma"
                else cusum_mean_delay(parameter, threshold, shift)
            )
            rows.append(
                {
                    "method": method,
                    "parameter": parameter,
                    "threshold": threshold,
                    "analytic_mean_delay": analytic_delay,
                    **empirical,
                }
            )
    return {
        "assumptions": {
            "normal_distribution": "standard normal",
            "shift": shift,
            "change_index": change_index,
            "target_false_alarm": target_false_alarm,
            "false_alarm_scope": "probability of at least one alert in a normal sequence",
            "repetitions": repetitions,
            "horizon": horizon,
        },
        "rows": rows,
        "interpretation": (
            "Analytic delays are mean-drift approximations. Empirical values "
            "include noise and threshold calibration; real scores are heavy-tailed "
            "and serially dependent, so they must be re-estimated on training data."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--target-false-alarm", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()

    result = run_grid(
        repetitions=args.repetitions,
        horizon=args.horizon,
        shift=args.shift,
        target_false_alarm=args.target_false_alarm,
        seed=args.seed,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
