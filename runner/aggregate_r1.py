from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RESULT_SCHEMA = "portable-runtime-r1-result.v1"
LEADERBOARD_SCHEMA = "portable-runtime-r1-leaderboard.v1"


def load_candidates(root: Path) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != RESULT_SCHEMA:
            continue
        candidate_id = str(value.get("candidate_id", ""))
        if not candidate_id:
            raise RuntimeError(f"missing candidate_id: {path}")
        if candidate_id in seen:
            raise RuntimeError(f"duplicate candidate_id: {candidate_id}")
        if value.get("contract_passed") is not True:
            raise RuntimeError(f"candidate contract failed: {candidate_id}")
        seen.add(candidate_id)
        candidates.append(value)
    if not candidates:
        raise RuntimeError("no R1 candidate summaries found")
    return candidates


def rank(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        candidates,
        key=lambda item: (
            -float(item["overall_score"]),
            float(item["runtime_ms"]),
            str(item["candidate_id"]),
        ),
    )


def compact(candidate: dict[str, object], rank_index: int) -> dict[str, object]:
    return {
        "rank": rank_index,
        "candidate_id": candidate["candidate_id"],
        "parameters": candidate["parameters"],
        "overall_score": candidate["overall_score"],
        "scores": candidate["scores"],
        "runtime_ms": candidate["runtime_ms"],
        "result_sha256": candidate["result_sha256"],
    }


def write_outputs(candidates: list[dict[str, object]], output_dir: Path) -> None:
    ranked = rank(candidates)
    rows = [compact(candidate, index) for index, candidate in enumerate(ranked, start=1)]
    output_dir.mkdir(parents=True, exist_ok=True)

    document = {
        "schema": LEADERBOARD_SCHEMA,
        "authority": "RESEARCH_ONLY_NOT_PRODUCTION_AUTHORITY",
        "candidate_count": len(rows),
        "ranking": rows,
    }
    (output_dir / "leaderboard.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    score_names = list(rows[0]["scores"].keys())
    with (output_dir / "leaderboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "candidate_id",
                "fft_size",
                "hop_ratio",
                "hop_samples",
                "window",
                "overall_score",
                "runtime_ms",
                *score_names,
                "result_sha256",
            ]
        )
        for row in rows:
            parameters = row["parameters"]
            writer.writerow(
                [
                    row["rank"],
                    row["candidate_id"],
                    parameters["fft_size"],
                    parameters["hop_ratio"],
                    parameters["hop_samples"],
                    parameters["window"],
                    row["overall_score"],
                    row["runtime_ms"],
                    *[row["scores"][name] for name in score_names],
                    row["result_sha256"],
                ]
            )

    lines = [
        "# Runtime Matrix R1 Leaderboard",
        "",
        f"Candidates: **{len(rows)}**",
        "",
        "Research-only ranking. It is not authority to change production software, models, or DSP.",
        "",
        "| Rank | Candidate | Overall | Tone | Polyphony | Noise | Transient | Continuity | Runtime ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:10]:
        scores = row["scores"]
        lines.append(
            "| {rank} | `{candidate}` | {overall:.4f} | {tone:.4f} | {poly:.4f} | "
            "{noise:.4f} | {transient:.4f} | {continuity:.4f} | {runtime:.2f} |".format(
                rank=row["rank"],
                candidate=row["candidate_id"],
                overall=float(row["overall_score"]),
                tone=float(scores["tone_accuracy"]),
                poly=float(scores["polyphony_resolution"]),
                noise=float(scores["noise_robustness"]),
                transient=float(scores["transient_localisation"]),
                continuity=float(scores["continuity"]),
                runtime=float(row["runtime_ms"]),
            )
        )
    lines.append("")
    (output_dir / "leaderboard.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidates = load_candidates(args.input_root)
    write_outputs(candidates, args.out)
    print((args.out / "leaderboard.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
