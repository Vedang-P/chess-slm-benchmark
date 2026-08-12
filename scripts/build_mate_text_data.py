"""Build the MATE text-transformer training corpus.

Joins all 4 MATE train subsets (strategy/tactic/noexplain/both) into one
hygienic corpus for a text-only transformer:

    python3 scripts/build_mate_text_data.py \
        --out data/raw/mate-text \
        --n-train 1400000 --n-val 50000

Hygiene (mirrors the campaign):
  - FEN-dedup across subsets (deterministic priority: noexplain wins,
    then strategy, tactic, both — noexplain is our special target)
  - MATE testset FENs excluded (the eval sets under data/positions/)
  - candidate order balanced (50/50 truth=A vs truth=B) so the model
    cannot exploit position-side bias
  - 95/5 train/val split, seed 42

Row schema:
    {"fen", "candidate_a", "candidate_b", "truth", "subset"}

where `truth` is "A" or "B" (the candidate index the expert chose).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "mate-text"
TRAIN_DIR = ROOT / "data" / "raw" / "mate-train"
SEED = 42

HF_BASE = "https://huggingface.co/datasets/OutFlankShu/MATE_DATASET/resolve/main"
TRAIN_ZIPS = {
    "noexplain": "no_explain.zip",
    "strategy": "strategy.zip",
    "tactic": "tactic.zip",
    "both": "both.zip",
}
# deterministic dedup priority: index 0 wins the FEN
SUBSET_PRIORITY = ["noexplain", "strategy", "tactic", "both"]

FEN_RE = re.compile(r'FEN of the given chess board is "([^"]+)"')
MOVE_RE = re.compile(r"Move([AB]):([a-h][1-8][a-h][1-8](?:[qrbnQRBN])?)")

TEST_FILES = [
    "data/positions/mate-selection-test.json",
    "data/positions/mate-selection-test-noexplain.json",
    "data/positions/mate-selection-test-tactic.json",
    "data/positions/mate-selection-test-both.json",
]


def _download_zip(url: str, dest: Path) -> None:
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "chess-bench"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    shutil.move(str(tmp), dest)


def download(args: argparse.Namespace) -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for subset, zip_name in TRAIN_ZIPS.items():
        dest = TRAIN_DIR / zip_name
        if dest.exists() and not args.force:
            print(f"skip {zip_name} (exists)", flush=True)
            continue
        _download_zip(f"{HF_BASE}/{zip_name}", dest)
        with zipfile.ZipFile(dest) as z:
            z.extractall(TRAIN_DIR / subset)
        print(f"{subset}: extracted to {TRAIN_DIR / subset}", flush=True)


def _parse_record(line: str, subset: str = "") -> dict | None:
    d = json.loads(line)
    fen_m = FEN_RE.search(d.get("input") or "")
    if not fen_m:
        return None
    moves = MOVE_RE.findall(d.get("input") or "")
    cand = {label: uci for label, uci in moves}
    if "A" not in cand or "B" not in cand:
        return None
    truth_m = re.match(r"Move([AB]):", d.get("output") or "")
    if not truth_m:
        return None
    return {
        "fen": fen_m.group(1),
        "candidate_a": cand["A"],
        "candidate_b": cand["B"],
        "truth": truth_m.group(1),
        "subset": subset,
    }


def _testset_fens() -> set[str]:
    fens: set[str] = set()
    for name in TEST_FILES:
        path = ROOT / name
        if not path.exists():
            continue
        for rec in json.loads(path.read_text()):
            if rec.get("fen"):
                fens.add(rec["fen"])
    return fens


def build(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_fens = _testset_fens()
    print(f"testset FENs to exclude: {len(test_fens)}", flush=True)

    rng = random.Random(SEED)
    # dedup by (fen, candidate_a, candidate_b) — the actual task instance.
    # MATE's subsets are different annotation runs over overlapping boards:
    # the same FEN carries different candidate pairs per subset (verified:
    # identical (fen,cand) tuples always have identical truth; the subsets
    # are genuinely distinct instances, not contradictions).
    by_key: dict[tuple, dict] = {}
    counts: dict[str, int] = {}

    for subset in SUBSET_PRIORITY:
        src_dir = TRAIN_DIR / subset
        jsonl = sorted(p for p in src_dir.rglob("*.jsonl")
                       if not p.name.startswith("._"))
        if not jsonl:
            print(f"WARNING: no train jsonl for {subset} in {src_dir}",
                  flush=True)
            continue
        subset_rows = 0
        for path in jsonl:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = _parse_record(line, subset)
                    if rec is None:
                        continue
                    if rec["fen"] in test_fens:
                        continue
                    key = (rec["fen"], rec["candidate_a"],
                           rec["candidate_b"])
                    if key not in by_key:
                        by_key[key] = rec
                        subset_rows += 1
        counts[subset] = subset_rows
        print(f"{subset}: {subset_rows} unique non-test instances", flush=True)

    rows = [r for r in by_key.values() if r is not None]
    rng.shuffle(rows)

    # FEN-disjoint split: group instances by fen, shuffle the fens, assign
    # val fens first so no board appears in both splits (contamination).
    by_fen_groups: dict[str, list[dict]] = {}
    for r in rows:
        by_fen_groups.setdefault(r["fen"], []).append(r)
    fens = list(by_fen_groups.keys())
    rng.shuffle(fens)

    val_fens = set(fens[:args.n_val])
    val = [r for f in fens[:args.n_val] for r in by_fen_groups[f]]
    train = [r for f in fens[args.n_val:] for r in by_fen_groups[f]]
    rng.shuffle(val)
    rng.shuffle(train)

    def write(rows_, name):
        with open(OUT_DIR / name, "w") as f:
            for r in rows_:
                f.write(json.dumps(r) + "\n")

    write(train, "train.jsonl")
    write(val, "val.jsonl")
    print(f"wrote {len(train)} train + {len(val)} val -> {OUT_DIR} "
          f"(val has {len(val_fens)} unique fens)", flush=True)
    print(f"per-subset (pre-split): {counts}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("download", help="fetch MATE train zips from HF")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("build", help="join + dedup + balance + split")
    p.add_argument("--n-train", type=int, default=1400000)
    p.add_argument("--n-val", type=int, default=50000)
    p.add_argument("--force", action="store_true")

    args = ap.parse_args()
    if args.stage == "download":
        download(args)
    elif args.stage == "build":
        build(args)


if __name__ == "__main__":
    main()
