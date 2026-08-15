# Portable Runtime Matrix

This repository is a public, implementation-neutral test harness for deterministic
signal-processing experiments on mathematically generated inputs.

It contains no application source, model files, recorded audio, training data,
private research material, credentials, or private build artefacts.

## Scope

The initial gate checks only public-safe properties:

- deterministic synthetic signal generation
- finite numeric outputs
- basic spectral invariants
- block-independent pure functions
- smoothing monotonicity
- repeatable machine-readable summaries

All inputs are generated mathematically at runtime.

## Run locally

```bash
python -B runner/run_matrix.py matrix/base.json --out out/base
```

The runner writes `summary.json`, `summary.csv`, and `summary.md`.
