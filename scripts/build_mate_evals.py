"""Build the committed MATE move-selection eval sets.

Source: MATE (NAACL 2025) held-out test set — `OutFlankShu/MATE_DATASET`,
downloaded locally (HF redirect: `OutFlankShu/MATE_NAACL2025_...` ->
`OutFlankShu/MATE_DATASET`). The `testset.zip` contains the dataset15
held-out split in four variants: strategy, noexplain, tactic, both. For
each variant we use the expert move annotations, dedupe by FEN, and take
a deterministic 1000-position sample (SEED = 2026) — the same protocol,
so all four sets are directly comparable and mirror the paper's 1000-per-
subset evaluation.

Outputs (committed):
    data/positions/mate-selection-test.json               (strategy)
    data/positions/mate-selection-test-noexplain.json
    data/positions/mate-selection-test-tactic.json
    data/positions/mate-selection-test-both.json

Each record carries the exact MATE instruction/input/output, the parsed
candidate moves, and the FEN so scoring and reproduction are exact.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SRC_DIR = Path("/tmp/mate/testset/data")
if not SRC_DIR.exists():
    SRC_DIR = Path("/tmp/mate/data")
SRC_FILES = {
    "strategy": "explain_dataset15_strategy.jsonl",
    "noexplain": "explain_dataset15_noexplain.jsonl",
    "tactic": "explain_dataset15_tactic.jsonl",
    "both": "explain_dataset15_both.jsonl",
}
OUT = Path("data/positions")
OUT_FILES = {
    "strategy": "mate-selection-test.json",
    "noexplain": "mate-selection-test-noexplain.json",
    "tactic": "mate-selection-test-tactic.json",
    "both": "mate-selection-test-both.json",
}
TARGET = 1000
SEED = 2026

FEN_RE = re.compile(r'FEN of the given chess board is "([^"]+)"')
MOVE_RE = re.compile(r"Move([AB]):([a-h][1-8][a-h][1-8](?:[qrbnQRBN])?)")


def parse_record(line: str, idx: int, theme: str) -> dict | None:
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
        "source": f"MATE-testset-{theme}",
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
            "theme": theme,
        },
    }


def build() -> None:
    for theme, src_name in SRC_FILES.items():
        records = []
        seen_fens = set()
        with (SRC_DIR / src_name).open() as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = parse_record(line, len(records), theme)
                if rec is None:
                    continue
                if rec["fen"] in seen_fens:
                    continue
                seen_fens.add(rec["fen"])
                records.append(rec)
        rng = random.Random(SEED)
        rng.shuffle(records)
        records = records[:TARGET]
        records.sort(key=lambda r: r["id"])
        out_path = OUT / OUT_FILES[theme]
        out_path.write_text(json.dumps(records, indent=1))
        print(f"wrote {len(records)} MATE {theme}-test records -> {out_path}")


if __name__ == "__main__":
    build()
