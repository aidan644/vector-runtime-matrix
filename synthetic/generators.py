from __future__ import annotations

import math
from typing import Iterable, Sequence

TAU = 2.0 * math.pi


def sine(
    frequency_hz: float,
    sample_rate_hz: float,
    sample_count: int,
    amplitude: float = 0.5,
    phase_radians: float = 0.0,
) -> list[float]:
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError("amplitude must be within [0, 1]")
    return [
        amplitude * math.sin(TAU * frequency_hz * index / sample_rate_hz + phase_radians)
        for index in range(sample_count)
    ]


def harmonic_stack(
    fundamental_hz: float,
    sample_rate_hz: float,
    sample_count: int,
    amplitudes: Sequence[float],
) -> list[float]:
    result = [0.0] * sample_count
    for harmonic_index, amplitude in enumerate(amplitudes, start=1):
        if amplitude < 0.0:
            raise ValueError("harmonic amplitudes must be non-negative")
        component = sine(
            fundamental_hz * harmonic_index,
            sample_rate_hz,
            sample_count,
            amplitude,
        )
        for index, value in enumerate(component):
            result[index] += value
    return _normalise_if_needed(result)


def mix(signals: Sequence[Sequence[float]]) -> list[float]:
    if not signals:
        return []
    size = len(signals[0])
    if any(len(signal) != size for signal in signals):
        raise ValueError("all signals must have equal length")
    result = [sum(signal[index] for signal in signals) for index in range(size)]
    return _normalise_if_needed(result)


def impulse(sample_count: int, index: int, amplitude: float = 1.0) -> list[float]:
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if not 0 <= index < sample_count:
        raise ValueError("index must address the signal")
    result = [0.0] * sample_count
    result[index] = amplitude
    return result


def deterministic_noise(sample_count: int, seed: int = 1, amplitude: float = 0.25) -> list[float]:
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError("amplitude must be within [0, 1]")
    state = seed & 0xFFFFFFFF
    result: list[float] = []
    for _ in range(sample_count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        unit = state / 0xFFFFFFFF
        result.append((2.0 * unit - 1.0) * amplitude)
    return result


def transient(
    sample_rate_hz: float,
    sample_count: int,
    attack_samples: int = 1,
    decay_seconds: float = 0.02,
) -> list[float]:
    if sample_rate_hz <= 0.0 or sample_count < 0:
        raise ValueError("invalid signal dimensions")
    if attack_samples < 1:
        raise ValueError("attack_samples must be at least one")
    if decay_seconds <= 0.0:
        raise ValueError("decay_seconds must be positive")

    result: list[float] = []
    decay_constant = decay_seconds * sample_rate_hz
    for index in range(sample_count):
        attack = min(1.0, (index + 1) / attack_samples)
        result.append(attack * math.exp(-index / decay_constant))
    return result


def _normalise_if_needed(signal: Iterable[float]) -> list[float]:
    values = list(signal)
    peak = max((abs(value) for value in values), default=0.0)
    if peak <= 1.0 or peak == 0.0:
        return values
    return [value / peak for value in values]
