"""Build the committed MATE move-selection eval set.

Source: MATE (NAACL 2025) held-out test set — `OutFlankShu/MATE_DATASET`,
downloaded locally (HF redirect: `OutFlankShu/MATE_NAACL2025_...` ->
`OutFlankShu/MATE_DATASET`). The `testset.zip` contains the dataset15
held-out split; we use the STRATEGY subset (expert move explanations),
dedupe by FEN, and take a deterministic 1000-position sample.

Output (committed):
    data/positions/mate-selection-test.json

Each record carries the exact MATE instruction/input/output, the parsed
candidate moves, and the FEN so scoring and reproduction are exact.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SRC = Path("/tmp/mate/testset/data/explain_dataset15_strategy.jsonl")
OUT = Path("data/positions/mate-selection-test.json")
TARGET = 1000
SEED = 2026

FEN_RE = re.compile(r'FEN of the given chess board is "([^"]+)"')
MOVE_RE = re.compile(r"Move([AB]):([a-h][1-8][a-h][1-8](?:[qrbnQRBN])?)")


def parse_record(line: str, idx: int) -> dict | None:
    d = json.loads(line)
    instruction = d["instruction"]
    input_text = d["input"]
    output = d["output"]
    fen_m = FEN_RE.search(input_text)
    if not fen_m:
        return None
    moves = MOVE_RE.findall(input_text)
    cand = {label: uci for label, uci in moves}
    if "A" not in cand or "B" not in cand:
        return None
    truth_m = re.match(r"Move([AB]):", output or "")
    if not truth_m:
        return None
    return {
        "id": f"mate-sel-{idx:05d}",
        "source": "MATE-testset-strategy",
        "n": 8,
        "turn": "w" if fen_m.group(1).split()[1] == "w" else "b",
        "value": "cap",
        "fen": fen_m.group(1),
        "presented_fen": fen_m.group(1),
        "pieces": [],
        "win_moves": [],
        "lose_moves": [],
        "over_budget": False,
        "task_extra": {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "candidate_a": cand["A"],
            "candidate_b": cand["B"],
            "truth_label": truth_m.group(1),
            "theme": "strategy",
        },
    }


def build() -> None:
    records = []
    seen_fens = set()
    with SRC.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = parse_record(line, len(records))
            if rec is None:
                continue
            if rec["fen"] in seen_fens:
                continue
            seen_fens.add(rec["fen"])
            records.append(rec)
    import random

    rng = random.Random(SEED)
    rng.shuffle(records)
    records = records[:TARGET]
    records.sort(key=lambda r: r["id"])
    OUT.write_text(json.dumps(records, indent=1))
    print(f"wrote {len(records)} MATE strategy-test records -> {OUT}")


if __name__ == "__main__":
    build()
