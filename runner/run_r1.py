from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from contracts.contracts import (
    all_finite,
    dominant_frequency,
    exponential_smoothing,
    is_monotonic_non_decreasing,
    magnitude_spectrum,
    rms,
)
from synthetic.generators import deterministic_noise, mix, sine, transient

TAU = 2.0 * math.pi


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def frames(signal: list[float], size: int, hop: int) -> list[tuple[int, list[float]]]:
    if len(signal) < size:
        return [(0, signal + [0.0] * (size - len(signal)))]
    return [
        (start, signal[start : start + size])
        for start in range(0, len(signal) - size + 1, hop)
    ]


def peak_frequencies(
    signal: list[float], sample_rate: float, window: str, count: int
) -> list[float]:
    magnitudes = magnitude_spectrum(signal, window)
    if len(magnitudes) <= 2 or count <= 0:
        return []
    ordered = sorted(
        range(1, len(magnitudes) - 1), key=magnitudes.__getitem__, reverse=True
    )
    selected: list[int] = []
    for index in ordered:
        if any(abs(index - existing) <= 1 for existing in selected):
            continue
        selected.append(index)
        if len(selected) >= count:
            break
    return [index * sample_rate / len(signal) for index in selected]


def linear_chirp(
    start_hz: float, end_hz: float, sample_rate: float, sample_count: int
) -> list[float]:
    result: list[float] = []
    phase = 0.0
    denominator = max(1, sample_count - 1)
    for index in range(sample_count):
        frequency = start_hz + (end_hz - start_hz) * index / denominator
        phase += TAU * frequency / sample_rate
        result.append(0.5 * math.sin(phase))
    return result


def tone_score(
    config: dict[str, object], fft_size: int, window: str
) -> tuple[float, list[dict[str, float]]]:
    sample_rate = float(config["sample_rate_hz"])
    bin_width = sample_rate / fft_size
    rows: list[dict[str, float]] = []
    scores: list[float] = []
    for frequency in [float(value) for value in config["tone_frequencies_hz"]]:
        signal = sine(frequency, sample_rate, fft_size)
        measured = dominant_frequency(signal, sample_rate, window)
        error = abs(measured - frequency)
        score = max(0.0, 1.0 - error / bin_width)
        rows.append(
            {
                "target_hz": frequency,
                "measured_hz": measured,
                "error_hz": error,
                "score": score,
            }
        )
        scores.append(score)
    return sum(scores) / len(scores), rows


def polyphony_score(
    config: dict[str, object], fft_size: int, window: str
) -> tuple[float, list[dict[str, object]]]:
    sample_rate = float(config["sample_rate_hz"])
    tolerance = sample_rate / fft_size
    rows: list[dict[str, object]] = []
    recalls: list[float] = []
    for raw_targets in config["polyphony_sets_hz"]:
        targets = [float(value) for value in raw_targets]
        signals = [
            sine(
                frequency,
                sample_rate,
                fft_size,
                amplitude=0.6 / len(targets),
            )
            for frequency in targets
        ]
        combined = mix(signals)
        measured = peak_frequencies(combined, sample_rate, window, len(targets))
        unused = list(measured)
        hits = 0
        for target in targets:
            if not unused:
                break
            nearest_index = min(
                range(len(unused)), key=lambda index: abs(unused[index] - target)
            )
            if abs(unused[nearest_index] - target) <= tolerance:
                hits += 1
                unused.pop(nearest_index)
        recall = hits / len(targets)
        rows.append(
            {"targets_hz": targets, "measured_hz": measured, "recall": recall}
        )
        recalls.append(recall)
    return sum(recalls) / len(recalls), rows


def noise_score(
    config: dict[str, object], fft_size: int, window: str
) -> tuple[float, list[dict[str, float]]]:
    sample_rate = float(config["sample_rate_hz"])
    target_hz = 720.0
    bin_width = sample_rate / fft_size
    tone = sine(target_hz, sample_rate, fft_size, amplitude=0.5)
    tone_rms = rms(tone)
    rows: list[dict[str, float]] = []
    scores: list[float] = []
    for offset, snr_db in enumerate(
        float(value) for value in config["noise_snr_db"]
    ):
        noise = deterministic_noise(
            fft_size,
            seed=int(config["noise_seed"]) + offset,
            amplitude=1.0,
        )
        noise_rms = max(rms(noise), 1.0e-12)
        desired_noise_rms = tone_rms / (10.0 ** (snr_db / 20.0))
        scale = desired_noise_rms / noise_rms
        mixed = [
            tone[index] + noise[index] * scale for index in range(fft_size)
        ]
        measured = dominant_frequency(mixed, sample_rate, window)
        error = abs(measured - target_hz)
        score = max(0.0, 1.0 - error / bin_width)
        rows.append(
            {
                "snr_db": snr_db,
                "measured_hz": measured,
                "error_hz": error,
                "score": score,
            }
        )
        scores.append(score)
    return sum(scores) / len(scores), rows


def transient_score(
    config: dict[str, object], fft_size: int, hop: int
) -> tuple[float, dict[str, float]]:
    sample_rate = float(config["sample_rate_hz"])
    onset = fft_size * 3 + hop // 2
    signal = [0.0] * (fft_size * 8)
    pulse = transient(
        sample_rate,
        fft_size * 2,
        attack_samples=2,
        decay_seconds=0.015,
    )
    for index, value in enumerate(pulse):
        position = onset + index
        if position >= len(signal):
            break
        signal[position] = value

    framed = frames(signal, fft_size, hop)
    energies = [rms(values) for _, values in framed]
    maximum = max(energies, default=0.0)
    threshold = maximum * 0.2
    first_active = next(
        (
            start
            for (start, _), energy in zip(framed, energies)
            if energy >= threshold
        ),
        onset,
    )
    error_samples = abs(first_active - onset)
    score = max(0.0, 1.0 - error_samples / max(1, fft_size))
    return score, {
        "onset_sample": float(onset),
        "estimated_sample": float(first_active),
        "error_samples": float(error_samples),
    }


def continuity_score(
    config: dict[str, object], fft_size: int, hop: int, window: str
) -> tuple[float, dict[str, object]]:
    sample_rate = float(config["sample_rate_hz"])
    signal = linear_chirp(180.0, 2880.0, sample_rate, fft_size * 10)
    measured = [
        dominant_frequency(values, sample_rate, window)
        for _, values in frames(signal, fft_size, hop)
    ]
    if len(measured) < 2:
        return 0.0, {"measurements_hz": measured}
    monotonic = sum(
        right + 1.0e-9 >= left for left, right in zip(measured, measured[1:])
    ) / (len(measured) - 1)
    bin_width = sample_rate / fft_size
    large_jumps = sum(
        abs(right - left) > 4.0 * bin_width
        for left, right in zip(measured, measured[1:])
    )
    jump_score = 1.0 - large_jumps / (len(measured) - 1)
    score = 0.65 * monotonic + 0.35 * jump_score
    return score, {
        "monotonic_fraction": monotonic,
        "jump_score": jump_score,
        "measurements_hz": measured,
    }


def smoothing_score(
    config: dict[str, object],
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    passed = 0
    values = [0.0] + [1.0] * 63
    for alpha in [float(value) for value in config["smoothing_alpha"]]:
        smoothed = exponential_smoothing(values, alpha)
        valid = (
            all_finite(smoothed)
            and is_monotonic_non_decreasing(smoothed)
            and 0.0 <= min(smoothed) <= max(smoothed) <= 1.0
        )
        passed += int(valid)
        rows.append({"alpha": alpha, "valid": valid, "last": smoothed[-1]})
    return passed / len(rows), rows


def numerical_contract(fft_size: int, window: str) -> bool:
    cases = [
        [0.0] * fft_size,
        [
            1.0e-30 if index % 2 == 0 else -1.0e-30
            for index in range(fft_size)
        ],
        [1.0 if index % 2 == 0 else -1.0 for index in range(fft_size)],
    ]
    return all(
        all_finite(magnitude_spectrum(case, window)) for case in cases
    )


def evaluate(
    config: dict[str, object], fft_size: int, hop_ratio: float, window: str
) -> dict[str, object]:
    check(
        fft_size > 0 and not (fft_size & (fft_size - 1)),
        "fft_size must be a power of two",
    )
    check(0.0 < hop_ratio <= 1.0, "hop_ratio must be within (0, 1]")
    check(window in {"hann", "rectangular"}, "unsupported window")
    hop = max(1, int(round(fft_size * hop_ratio)))

    started = time.perf_counter()
    tone, tone_rows = tone_score(config, fft_size, window)
    polyphony, polyphony_rows = polyphony_score(config, fft_size, window)
    noise, noise_rows = noise_score(config, fft_size, window)
    transient_value, transient_row = transient_score(config, fft_size, hop)
    continuity, continuity_row = continuity_score(
        config, fft_size, hop, window
    )
    smoothing, smoothing_rows = smoothing_score(config)
    finite = numerical_contract(fft_size, window)

    scores = {
        "tone_accuracy": tone,
        "polyphony_resolution": polyphony,
        "noise_robustness": noise,
        "transient_localisation": transient_value,
        "continuity": continuity,
        "smoothing_contract": smoothing,
        "numerical_stability": 1.0 if finite else 0.0,
    }
    overall = (
        0.20 * tone
        + 0.30 * polyphony
        + 0.15 * noise
        + 0.10 * transient_value
        + 0.15 * continuity
        + 0.05 * smoothing
        + 0.05 * scores["numerical_stability"]
    )
    runtime_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema": "portable-runtime-r1-result.v1",
        "candidate_id": f"fft{fft_size}-hop{hop_ratio:.2f}-{window}",
        "parameters": {
            "fft_size": fft_size,
            "hop_ratio": hop_ratio,
            "hop_samples": hop,
            "window": window,
        },
        "scores": scores,
        "overall_score": overall,
        "contract_passed": finite and smoothing == 1.0,
        "runtime_ms": runtime_ms,
        "details": {
            "tones": tone_rows,
            "polyphony": polyphony_rows,
            "noise": noise_rows,
            "transient": transient_row,
            "continuity": continuity_row,
            "smoothing": smoothing_rows,
        },
    }


def stable_copy(summary: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"runtime_ms", "result_sha256"}
    }


def write_outputs(summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate_id",
                "overall_score",
                "contract_passed",
                "runtime_ms",
                *summary["scores"].keys(),
            ]
        )
        writer.writerow(
            [
                summary["candidate_id"],
                summary["overall_score"],
                summary["contract_passed"],
                summary["runtime_ms"],
                *summary["scores"].values(),
            ]
        )
    lines = [
        "# Runtime Matrix R1 Candidate",
        "",
        f"- Candidate: `{summary['candidate_id']}`",
        f"- Contract passed: **{summary['contract_passed']}**",
        f"- Overall score: **{summary['overall_score']:.6f}**",
        f"- Result SHA-256: `{summary['result_sha256']}`",
        f"- Runtime: **{summary['runtime_ms']:.3f} ms**",
        "",
        "This is a public-safe synthetic research score, not production authority.",
        "",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--fft-size", type=int, required=True)
    parser.add_argument("--hop-ratio", type=float, required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    first = evaluate(config, args.fft_size, args.hop_ratio, args.window)
    second = evaluate(config, args.fft_size, args.hop_ratio, args.window)
    first_hash = canonical_hash(stable_copy(first))
    second_hash = canonical_hash(stable_copy(second))
    check(first_hash == second_hash, "deterministic replay mismatch")
    check(bool(first["contract_passed"]), "numeric contract failed")
    first["result_sha256"] = first_hash
    write_outputs(first, args.out)
    print(
        json.dumps(
            {
                "candidate_id": first["candidate_id"],
                "overall_score": first["overall_score"],
                "scores": first["scores"],
                "result_sha256": first["result_sha256"],
                "runtime_ms": first["runtime_ms"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
