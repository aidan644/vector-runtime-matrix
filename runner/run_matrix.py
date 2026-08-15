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
    rms,
    spectral_centroid,
)
from synthetic.generators import (
    deterministic_noise,
    harmonic_stack,
    impulse,
    mix,
    sine,
    transient,
)


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(config: dict[str, object]) -> dict[str, object]:
    sample_rate = float(config["sample_rate_hz"])
    fft_sizes = [int(value) for value in config["fft_sizes"]]
    windows = [str(value) for value in config["windows"]]
    frequencies = [float(value) for value in config["frequencies_hz"]]
    smoothing_values = [float(value) for value in config["smoothing_alpha"]]
    seed = int(config["noise_seed"])

    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for fft_size in fft_sizes:
        check(fft_size > 0 and not (fft_size & (fft_size - 1)), "FFT sizes must be powers of two")
        for window in windows:
            for frequency in frequencies:
                signal = sine(frequency, sample_rate, fft_size)
                repeat = sine(frequency, sample_rate, fft_size)
                check(signal == repeat, "synthetic generation must be deterministic")
                check(all_finite(signal), "synthetic signal contains non-finite values")

                measured = dominant_frequency(signal, sample_rate, window)
                centroid = spectral_centroid(signal, sample_rate, window)
                bin_width = sample_rate / fft_size
                error = abs(measured - frequency)

                check(math.isfinite(measured) and math.isfinite(centroid), "analysis produced non-finite values")
                check(error <= bin_width + 1.0e-12, "dominant frequency exceeded one-bin tolerance")

                rows.append(
                    {
                        "kind": "tone",
                        "fft_size": fft_size,
                        "window": window,
                        "input_frequency_hz": frequency,
                        "measured_frequency_hz": measured,
                        "frequency_error_hz": error,
                        "bin_width_hz": bin_width,
                        "rms": rms(signal),
                        "spectral_centroid_hz": centroid,
                        "pass": True,
                    }
                )

    polyphony = mix(
        [
            sine(220.0, sample_rate, 1024, 0.25),
            sine(440.0, sample_rate, 1024, 0.25),
            sine(660.0, sample_rate, 1024, 0.25),
        ]
    )
    stack = harmonic_stack(110.0, sample_rate, 1024, [0.5, 0.25, 0.125, 0.0625])
    pulse = impulse(1024, 256)
    envelope = transient(sample_rate, 1024, attack_samples=4, decay_seconds=0.02)
    noise_a = deterministic_noise(1024, seed=seed)
    noise_b = deterministic_noise(1024, seed=seed)

    for name, signal in (
        ("polyphony", polyphony),
        ("harmonic_stack", stack),
        ("impulse", pulse),
        ("transient", envelope),
        ("noise", noise_a),
    ):
        check(all_finite(signal), f"{name} produced non-finite values")
        rows.append(
            {
                "kind": name,
                "fft_size": 1024,
                "window": "hann",
                "rms": rms(signal),
                "spectral_centroid_hz": spectral_centroid(signal, sample_rate, "hann"),
                "pass": True,
            }
        )

    check(noise_a == noise_b, "noise generator must be deterministic for a fixed seed")

    step = [0.0] + [1.0] * 31
    for alpha in smoothing_values:
        smoothed = exponential_smoothing(step, alpha)
        check(all_finite(smoothed), "smoothing produced non-finite values")
        check(is_monotonic_non_decreasing(smoothed), "step response must be monotonic")
        check(0.0 <= min(smoothed) <= max(smoothed) <= 1.0, "step response escaped [0, 1]")
        rows.append(
            {
                "kind": "smoothing",
                "alpha": alpha,
                "first": smoothed[0],
                "last": smoothed[-1],
                "monotonic": True,
                "pass": True,
            }
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    stable_rows = [
        {key: value for key, value in row.items() if key != "runtime_ms"} for row in rows
    ]
    summary = {
        "schema": "portable-runtime-summary.v1",
        "config_sha256": canonical_hash(config),
        "result_sha256": canonical_hash(stable_rows),
        "case_count": len(rows),
        "passed": all(bool(row["pass"]) for row in rows),
        "runtime_ms": elapsed_ms,
        "rows": rows,
    }
    return summary


def write_outputs(summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = list(summary["rows"])
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Runtime Matrix Summary",
        "",
        f"- Passed: **{summary['passed']}**",
        f"- Cases: **{summary['case_count']}**",
        f"- Config SHA-256: `{summary['config_sha256']}`",
        f"- Result SHA-256: `{summary['result_sha256']}`",
        f"- Runtime: **{summary['runtime_ms']:.3f} ms**",
        "",
        "All inputs were generated mathematically during this run.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    first = run(config)
    second = run(config)

    check(first["config_sha256"] == second["config_sha256"], "config hash changed")
    check(first["result_sha256"] == second["result_sha256"], "deterministic replay mismatch")
    check(first["passed"] is True and second["passed"] is True, "matrix gate failed")

    write_outputs(first, args.out)
    print((args.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
