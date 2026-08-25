"""Merge the caveman trace shards into the canonical file.

Three concurrent Kaggle kernels generate disjoint position slices:
  - caveman-traces/traces.jsonl          (prefix worker, key 1)
  - caveman-traces/shards/shard-2.jsonl  (rows 667-1333, key 2)
  - caveman-traces/shards/shard-3.jsonl  (rows 1334-1999, key 3)

This combines them into the canonical caveman-traces/traces.jsonl,
deduped by fen (first occurrence wins; canonical first, then shards in
order). Rows that overlap slices (the prefix worker runs past row 667
in its 12h window) are absorbed here — no double counting.

    python3 scripts/merge_caveman_shards.py          # HF -> HF
    python3 scripts/merge_caveman_shards.py --local results/caveman/traces-2000.jsonl

Run after every shard completes (idempotent: re-running merges the same
inputs; already-merged rows are deduped by fen).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.hf_push import _api  # noqa: E402

REPO_ID = "vedangfake/chess-slm-benchmark"
SOURCES = [
    "caveman-traces/traces.jsonl",
    "caveman-traces/shards/shard-2.jsonl",
    "caveman-traces/shards/shard-3.jsonl",
]


def _download(api, filename: str) -> list[dict]:
    try:
        p = api.hf_hub_download(repo_id=REPO_ID, filename=filename,
                                repo_type="dataset")
        return [json.loads(l) for l in Path(p).read_text().splitlines()
                if l.strip()]
    except Exception as e:
        print(f"  {filename}: skipped ({type(e).__name__})", flush=True)
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="",
                    help="also write the merged file locally")
    args = ap.parse_args()

    api = _api()
    merged: dict[str, dict] = {}
    src_counts = {}
    for name in SOURCES:
        rows = _download(api, name)
        src_counts[name] = len(rows)
        for r in rows:
            merged.setdefault(r["fen"], r)
    print(f"sources: {src_counts}", flush=True)
    print(f"merged: {len(merged)} unique positions "
          f"({sum(1 for r in merged.values() if r.get('verified'))} verified)",
          flush=True)

    out = "\n".join(json.dumps(r) for r in merged.values()) + "\n"
    api.upload_file(path_or_fileobj=out.encode(),
                    path_in_repo="caveman-traces/traces.jsonl",
                    repo_id=REPO_ID, repo_type="dataset",
                    commit_message=f"caveman merge ({len(merged)} rows, "
                                   f"{sum(1 for r in merged.values() if r.get('verified'))} verified)")
    print("canonical updated: caveman-traces/traces.jsonl", flush=True)
    if args.local:
        Path(args.local).write_text(out)
        print(f"local copy: {args.local}", flush=True)


if __name__ == "__main__":
    main()
