"""Freeze a Lichess puzzle export against the official 10k evaluation set.

The output is a small manifest. The compressed source is retained separately
and addressed by its SHA256; Kaggle workers must verify that digest before
reading any row. Only the 9,995 verbatim source/evaluation ID matches are
excluded, matching the user-approved experiment contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import zstandard as zstd

EXPECTED_SOURCE_ROWS = 6_157_341
EXPECTED_OFFICIAL_ROWS = 10_000
EXPECTED_MATCHES = 9_995
SOURCE_COLUMNS = {"PuzzleId", "FEN", "Moves"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def official_rows(path: Path) -> dict[str, tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not SOURCE_COLUMNS <= set(reader.fieldnames or ()):
            raise ValueError("official puzzle CSV lacks PuzzleId/FEN/Moves")
        out = {}
        for row in reader:
            puzzle_id = row["PuzzleId"]
            if puzzle_id in out:
                raise ValueError(f"duplicate official PuzzleId: {puzzle_id}")
            out[puzzle_id] = (row["FEN"], row["Moves"])
    if len(out) != EXPECTED_OFFICIAL_ROWS:
        raise ValueError(f"expected {EXPECTED_OFFICIAL_ROWS} official rows; got {len(out)}")
    return out


def freeze(source: Path, official: Path, expected_sha256: str) -> dict:
    source_sha256 = file_sha256(source)
    if source_sha256 != expected_sha256:
        raise ValueError(f"source SHA256 mismatch: {source_sha256}")
    benchmark = official_rows(official)
    matches = []
    count = 0
    with source.open("rb") as raw:
        stream = zstd.ZstdDecompressor().stream_reader(raw)
        with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            if not SOURCE_COLUMNS <= set(reader.fieldnames or ()):
                raise ValueError("source CSV lacks PuzzleId/FEN/Moves")
            for row in reader:
                count += 1
                puzzle_id = row["PuzzleId"]
                expected = benchmark.get(puzzle_id)
                if expected is None:
                    continue
                if (row["FEN"], row["Moves"]) != expected:
                    raise ValueError(f"official PuzzleId {puzzle_id} changed FEN or Moves")
                matches.append(puzzle_id)
    if count != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"expected {EXPECTED_SOURCE_ROWS} source rows; got {count}")
    if len(matches) != EXPECTED_MATCHES or len(set(matches)) != EXPECTED_MATCHES:
        raise ValueError(f"expected {EXPECTED_MATCHES} unique official matches; got {len(matches)}")
    return {
        "format": "lichess-puzzle-corpus-v1",
        "source_url": "https://database.lichess.org/lichess_db_puzzle.csv.zst",
        "source_sha256": source_sha256,
        "source_bytes": source.stat().st_size,
        "source_rows": count,
        "official_csv_sha256": file_sha256(official),
        "official_rows": len(benchmark),
        "excluded_ids": sorted(matches),
        "excluded_count": len(matches),
        "included_roots": count - len(matches),
        "root_definition": "FEN after applying Moves[0]; one puzzle-player decision per source row",
        "model_source_checkpoint": "ccgavn-5m-seed0-v2/checkpoint-3896658",
        "additional_steps": 2_000_000,
        "batch": 2_048,
        "evaluation_interval_steps": 50_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.source, args.official, args.source_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"frozen roots={manifest['included_roots']:,} "
          f"excluded={manifest['excluded_count']:,} source={manifest['source_sha256']}")


if __name__ == "__main__":
    main()
