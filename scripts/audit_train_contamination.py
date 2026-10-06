"""Quantify benchmark contamination and split overlap in the training corpus.

The 2026-10-06 review found that ChessBench training previously ran without
MATE/puzzle exclusions and that development-fold exclusions only applied to the
first shard. The trainer now excludes reflect-orbit position hashes globally;
this audit measures the actual numbers rather than assuming them:

  - MATE: rows, unique exact positions, unique reflection orbits, cross-file
    duplicates (rows are not independent),
  - per shard: sampled rows, fraction excluded (dev fold / MATE / puzzles),
    within-shard duplicate positions, and cross-shard duplicate positions.

Runs on CPU. Downloads only ``train_set.npz`` per shard into a cache dir; use
``--per-shard`` to bound hashing work. Example (Kaggle CPU kernel):

  python3 scripts/audit_train_contamination.py \
      --puzzles searchless_chess/data/puzzles.csv \
      --tags 00000,00001,00002 --per-shard 2000000 --out logs/audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.data_hygiene import load_exclusions, position_hashes  # noqa: E402

HF_REPO = "vedangfake/chess-slm-benchmark"
PREFIX = "chessbench-full-build"
TAGS_FILE = ROOT / "configs" / "ccgavn-2b-shard-tags.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--puzzles", required=True, help="official puzzles.csv (10,000 rows)")
    p.add_argument("--mate-dir", default=str(ROOT / "data/positions"))
    p.add_argument("--tags", default="", help="comma-separated shard tags (default: first --shards of the frozen set)")
    p.add_argument("--shards", type=int, default=3)
    p.add_argument("--per-shard", type=int, default=2_000_000)
    p.add_argument("--dev-mod", type=int, default=100)
    p.add_argument("--dev-fold", type=int, default=0)
    p.add_argument("--cache", default=str(ROOT / "data" / "audit-cache"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "logs" / "audit_train_contamination.json"))
    return p.parse_args()


def shard_tags(args) -> list[str]:
    if args.tags:
        return [t.strip() for t in args.tags.split(",") if t.strip()]
    payload = json.loads(TAGS_FILE.read_text(encoding="utf-8"))
    tags = payload.get("tags") if isinstance(payload, dict) else payload
    return sorted(str(t) for t in tags)[:args.shards]


def mate_stats(mate_dir: Path) -> dict:
    exact, orbits = set(), set()
    per_file = {}
    for path in sorted(Path(mate_dir).glob("mate-selection-test*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        keys = []
        for row in rows:
            keys.append(" ".join((row.get("fen") or row["position"]).split()[:4]))
        per_file[path.name] = {"rows": len(rows), "unique_positions": len(set(keys))}
        exact.update(keys)
    return {"files": per_file, "rows": sum(v["rows"] for v in per_file.values()),
            "unique_positions": len(exact)}


def main() -> None:
    args = parse_args()
    from huggingface_hub import hf_hub_download
    from scripts.kaggle_checkpoint import hf_token

    token = hf_token(ROOT)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    tags = shard_tags(args)
    rng = np.random.default_rng(args.seed)

    excluded, digest = load_exclusions(Path(args.mate_dir), Path(args.puzzles))
    audit = {"exclusion_digest": digest, "mate": mate_stats(Path(args.mate_dir)),
             "dev_mod": args.dev_mod, "dev_fold": args.dev_fold, "shards": {}}
    seen: dict[int, np.ndarray] = {}
    total_rows = total_sampled = 0
    total_excluded = total_dup = total_cross = 0
    for tag in tags:
        remote = f"{PREFIX}/shard-{tag}/train_set.npz"
        try:
            path = hf_hub_download(HF_REPO, remote, repo_type="dataset",
                                   token=token, local_dir=str(cache))
        except Exception as exc:
            print(f"[audit] {tag}: download failed: {exc}", flush=True)
            continue
        data = np.load(path)
        tokens = data["tokens"]
        rows = int(len(tokens))
        sample = min(rows, args.per_shard)
        idx = rng.choice(rows, size=sample, replace=False)
        hashed = position_hashes(np.asarray(tokens[np.sort(idx)]))
        dev = hashed % np.uint64(args.dev_mod) == np.uint64(args.dev_fold)
        excl = np.isin(hashed, excluded)
        unique = np.unique(hashed)
        cross = 0
        for other, other_hashes in seen.items():
            cross += int(np.isin(unique, other_hashes, assume_unique=True).sum())
        rec = {"rows": rows, "sampled": int(sample),
               "excluded_frac": float((excl | dev).mean()),
               "dev_fold_frac": float(dev.mean()),
               "benchmark_frac": float((excl & ~dev).mean()),
               "duplicate_frac": float(1 - len(unique) / sample),
               "cross_shard_duplicate_positions": int(cross)}
        audit["shards"][tag] = rec
        seen[tag] = unique
        total_rows += rows
        total_sampled += int(sample)
        total_excluded += int((excl | dev).sum())
        total_dup += int(sample - len(unique))
        total_cross += cross
        print(f"[audit] {tag}: rows={rows:,} sampled={sample:,} "
              f"excluded={rec['excluded_frac']:.4%} (bench {rec['benchmark_frac']:.4%}, "
              f"dev {rec['dev_fold_frac']:.4%}) dup={rec['duplicate_frac']:.4%}",
              flush=True)
        del tokens, data
    audit["totals"] = {
        "shards": len(audit["shards"]), "rows": total_rows, "sampled": total_sampled,
        "excluded_frac": total_excluded / max(1, total_sampled),
        "duplicate_frac": total_dup / max(1, total_sampled),
        "cross_shard_duplicate_positions": total_cross,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[audit] wrote {out}\n{json.dumps(audit['totals'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
