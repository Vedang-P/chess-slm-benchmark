"""Freeze the work plan for the later-ply labeling fleet (run once, then upload).

Writes ``plan.json`` (eligible roots and shard counts per ply, digests of every input)
and ``exclusion-hashes.npy`` (the benchmark position set that labelers drop). The
exclusion set is rebuilt exactly as scripts/build_cont_data.py builds it for the
first-decision pack, and must reproduce that pack's hash count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.build_cont_data import hf_get, mate_hashes  # noqa: E402
from scripts.cont_data import (CSV_FIELDS, ROOTS_PER_SHARD, extra_exclusion_hashes,  # noqa: E402
                               iter_source, read_split_csv)
from scripts.freeze_puzzle_corpus import file_sha256  # noqa: E402
from scripts.ply_data import (ENGINE, FORMAT, PLIES, PREFIX, SECONDS_PER_ACTION,  # noqa: E402
                              count_eligible, read_ids, shard_counts)
from scripts.tactical_config import HF_REPO, SOURCE_PREFIX, SOURCE_SHA256  # noqa: E402

PLAN_NAME, HASHES_NAME = "plan.json", "exclusion-hashes.npy"


def exclusion_hashes(source: Path, official_ids: set[str], val_csv: Path, test_csv: Path,
                     mate_dir: Path) -> np.ndarray:
    """MATE rows, the official lines and the validation/test lines, as in build_cont_data."""
    official = [{k: row[k] for k in CSV_FIELDS} for row in iter_source(source)
                if row["PuzzleId"] in official_ids]
    if len(official) != len(official_ids):
        raise ValueError(f"found {len(official)} of {len(official_ids)} official puzzles")
    return np.unique(np.concatenate([
        mate_hashes(mate_dir), extra_exclusion_hashes(official),
        extra_exclusion_hashes(read_split_csv(val_csv)),
        extra_exclusion_hashes(read_split_csv(test_csv))]))


def build_plan(source: Path, manifest: dict, val_csv: Path, test_csv: Path, mate_dir: Path,
               out: Path, expected_hashes: int | None) -> dict:
    if file_sha256(source) != SOURCE_SHA256 or manifest["source_sha256"] != SOURCE_SHA256:
        raise ValueError("puzzle source digest mismatch")
    official_ids = set(manifest["excluded_ids"])
    skip_ids = read_ids(val_csv, test_csv)
    hashes = exclusion_hashes(source, official_ids, val_csv, test_csv, mate_dir)
    if expected_hashes is not None and len(hashes) != expected_hashes:
        raise ValueError(f"rebuilt {len(hashes):,} exclusion hashes; the cont-v1 pack used "
                         f"{expected_hashes:,}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / HASHES_NAME, hashes)
    eligible = count_eligible(source, official_ids, skip_ids, PLIES)
    shards = shard_counts(eligible)
    plan = {"format": f"{FORMAT}-plan", "source_sha256": SOURCE_SHA256,
            "official_excluded": len(official_ids), "included_roots": manifest["included_roots"],
            "val_csv_sha256": file_sha256(val_csv), "test_csv_sha256": file_sha256(test_csv),
            "exclusion_sha256": file_sha256(out / HASHES_NAME), "exclusion_count": int(len(hashes)),
            "roots_per_shard": ROOTS_PER_SHARD, "engine": ENGINE,
            "seconds_per_action": SECONDS_PER_ACTION,
            "plies": {str(p): {"eligible": eligible[p], "shards": shards[p]} for p in PLIES},
            "jobs": sum(shards.values()), "prefix": PREFIX,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / PLAN_NAME).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--hf-repo", default=HF_REPO)
    p.add_argument("--mate-dir", type=Path, default=ROOT / "data/positions")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--pause", action="store_true",
                   help="upload a PAUSE marker with the plan so no worker starts until it is deleted")
    args = p.parse_args()

    from scripts.kaggle_checkpoint import api as make_api
    api = make_api(ROOT)
    work = args.workdir
    source = hf_get(api, args.hf_repo, f"{SOURCE_PREFIX}/source.csv.zst", work)
    manifest = json.loads(hf_get(api, args.hf_repo, f"{SOURCE_PREFIX}/manifest.json",
                                 work).read_text(encoding="utf-8"))
    cont = {name: hf_get(api, args.hf_repo, f"cont-v1/{name}", work)
            for name in ("val.csv", "test.csv", "manifest.json")}
    cont_manifest = json.loads(cont["manifest.json"].read_text(encoding="utf-8"))
    if (file_sha256(cont["val.csv"]) != cont_manifest["val_csv_sha256"]
            or file_sha256(cont["test.csv"]) != cont_manifest["test_csv_sha256"]):
        raise ValueError("cont-v1 split files do not match their manifest")
    out = work / PREFIX
    plan = build_plan(source, manifest, cont["val.csv"], cont["test.csv"], args.mate_dir, out,
                      cont_manifest["excluded_hashes"])
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.upload:
        from scripts.hf_upload import retry_hf_call
        patterns = [PLAN_NAME, HASHES_NAME]
        if args.pause:
            (out / "PAUSE").write_text("paused until this file is deleted\n")
            patterns.append("PAUSE")
        retry_hf_call(lambda: api.upload_folder(
            folder_path=str(out), path_in_repo=PREFIX, repo_id=args.hf_repo, repo_type="dataset",
            allow_patterns=patterns,
            commit_message="Add the later-ply labeling plan and exclusion set"), "upload ply plan")
        print(f"[plan] uploaded {PREFIX}/{PLAN_NAME}", flush=True)


if __name__ == "__main__":
    main()
