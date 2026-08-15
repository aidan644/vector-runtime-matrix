from __future__ import annotations

import cmath
import math
from typing import Iterable, Sequence


def all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def apply_window(values: Sequence[float], name: str) -> list[float]:
    if name == "rectangular":
        return list(values)
    if name == "hann":
        if len(values) <= 1:
            return list(values)
        denominator = len(values) - 1
        return [
            value * (0.5 - 0.5 * math.cos(2.0 * math.pi * index / denominator))
            for index, value in enumerate(values)
        ]
    raise ValueError(f"unsupported window: {name}")


def fft(values: Sequence[float]) -> list[complex]:
    size = len(values)
    if size == 0 or size & (size - 1):
        raise ValueError("FFT size must be a positive power of two")

    output = [complex(value, 0.0) for value in values]

    target = 0
    for source in range(1, size):
        bit = size >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target ^= bit
        if source < target:
            output[source], output[target] = output[target], output[source]

    length = 2
    while length <= size:
        root = cmath.exp(-2j * math.pi / length)
        half = length // 2
        for start in range(0, size, length):
            factor = 1.0 + 0.0j
            for offset in range(half):
                even = output[start + offset]
                odd = output[start + offset + half] * factor
                output[start + offset] = even + odd
                output[start + offset + half] = even - odd
                factor *= root
        length <<= 1

    return output


def magnitude_spectrum(values: Sequence[float], window: str) -> list[float]:
    transformed = fft(apply_window(values, window))
    return [abs(value) for value in transformed[: len(values) // 2 + 1]]


def dominant_frequency(
    values: Sequence[float],
    sample_rate_hz: float,
    window: str,
) -> float:
    magnitudes = magnitude_spectrum(values, window)
    if len(magnitudes) <= 1:
        return 0.0
    peak_index = max(range(1, len(magnitudes)), key=magnitudes.__getitem__)
    return peak_index * sample_rate_hz / len(values)


def spectral_centroid(
    values: Sequence[float],
    sample_rate_hz: float,
    window: str,
) -> float:
    magnitudes = magnitude_spectrum(values, window)
    total = sum(magnitudes)
    if total == 0.0:
        return 0.0
    bin_width = sample_rate_hz / len(values)
    return sum(index * bin_width * magnitude for index, magnitude in enumerate(magnitudes)) / total


def exponential_smoothing(values: Sequence[float], alpha: float) -> list[float]:
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be within [0, 1)")
    if not values:
        return []
    output = [float(values[0])]
    for value in values[1:]:
        output.append(alpha * output[-1] + (1.0 - alpha) * value)
    return output


def is_monotonic_non_decreasing(values: Sequence[float], tolerance: float = 1.0e-12) -> bool:
    return all(right + tolerance >= left for left, right in zip(values, values[1:]))
