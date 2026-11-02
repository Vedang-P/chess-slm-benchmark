"""Build and (optionally) upload the later-ply training pack: ``cont-v2/ply-pack``.

Reads every finished ply shard on the Hub, checks each against the labeling plan
(identity, digests, hygiene) and writes the tactical-pack layout plus a ``ply`` field.
Streams shard by shard, so peak disk is one batch of shards plus the pack. Artifacts
land under ``<workdir>/cont-v2/ply-pack`` and, with ``--upload``, in the same path in
the Hugging Face dataset. Nothing here touches ``cont-v1``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.build_cont_data import hf_get  # noqa: E402
from scripts.cont_data import ROOTS_PER_SHARD, save_pack  # noqa: E402
from scripts.freeze_puzzle_corpus import file_sha256  # noqa: E402
from scripts.ply_data import (PACK_FORMAT, PACK_PREFIX, PLIES, PREFIX, PlyPackBuilder,  # noqa: E402
                              complete_shards, pack_manifest_digest, read_ids, shard_dir,
                              shard_identity)
from scripts.tactical_config import HF_REPO, SOURCE_PREFIX  # noqa: E402

PACK_PARTS = 8
SHARD_KEYS = ("tokens", "actions", "winprob", "lens", "index", "rating", "puzzle_id", "ply")


def load_shard(api, repo: str, ply: int, shard: int, workdir: Path, plan: dict):
    """Download one finished shard and verify it is the shard the plan describes."""
    remote = shard_dir(PREFIX, ply, shard)
    meta_path = hf_get(api, repo, f"{remote}/meta.json", workdir)
    npz_path = hf_get(api, repo, f"{remote}/train_set.npz", workdir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = shard_identity(
        source_sha256=plan["source_sha256"], ply=ply, shard=shard, max_roots=0,
        val_csv_sha256=plan["val_csv_sha256"], test_csv_sha256=plan["test_csv_sha256"],
        exclusion_sha256=plan["exclusion_sha256"], roots_per_shard=plan["roots_per_shard"])
    if not meta.get("complete") or meta.get("identity") != expected:
        raise ValueError(f"ply{ply} shard {shard:05d} failed the plan identity gate")
    if file_sha256(npz_path) != meta.get("train_set_sha256"):
        raise ValueError(f"ply{ply} shard {shard:05d}: train_set.npz digest mismatch")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in SHARD_KEYS}
    return meta, arrays, npz_path, meta_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--hf-repo", default=HF_REPO)
    p.add_argument("--plies", type=int, nargs="+", default=list(PLIES))
    p.add_argument("--max-shards", type=int, default=0, help="per ply; 0 = every finished shard")
    p.add_argument("--upload", action="store_true")
    args = p.parse_args()

    from scripts.kaggle_checkpoint import api as make_api
    api = make_api(ROOT)
    work = args.workdir
    out = work / PACK_PREFIX
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    plan_path = hf_get(api, args.hf_repo, f"{PREFIX}/plan.json", work)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    hashes_path = hf_get(api, args.hf_repo, f"{PREFIX}/exclusion-hashes.npy", work)
    if file_sha256(hashes_path) != plan["exclusion_sha256"]:
        raise ValueError("exclusion hashes do not match the plan")
    manifest = json.loads(hf_get(api, args.hf_repo, f"{SOURCE_PREFIX}/manifest.json",
                                 work).read_text(encoding="utf-8"))
    val_csv = hf_get(api, args.hf_repo, "cont-v1/val.csv", work)
    test_csv = hf_get(api, args.hf_repo, "cont-v1/test.csv", work)
    if (file_sha256(val_csv) != plan["val_csv_sha256"]
            or file_sha256(test_csv) != plan["test_csv_sha256"]):
        raise ValueError("validation/test files do not match the plan")
    excluded_ids = set(manifest["excluded_ids"]) | read_ids(val_csv, test_csv)
    builder = PlyPackBuilder(np.load(hashes_path, allow_pickle=False), excluded_ids)

    files = set(api.list_repo_files(args.hf_repo, repo_type="dataset"))
    done = complete_shards(files, PREFIX, args.plies)
    jobs = [(ply, shard) for ply in args.plies
            for shard in sorted(done[ply])[: args.max_shards or None]]
    if not jobs:
        raise ValueError("no finished ply shards on the Hub")
    print(f"[ply-pack] {len(jobs)} finished shards "
          f"({', '.join(f'ply{p}: {len(done[p])}' for p in args.plies)})", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {job: pool.submit(load_shard, api, args.hf_repo, *job, work / "shards", plan)
                   for job in jobs[:12]}
        pending = list(jobs[12:])
        for n, job in enumerate(jobs, 1):
            meta, arrays, npz_path, meta_path = futures.pop(job).result()
            builder.add_shard(job[0], job[1], arrays, meta)
            npz_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            if pending:
                nxt = pending.pop(0)
                futures[nxt] = pool.submit(load_shard, api, args.hf_repo, *nxt,
                                           work / "shards", plan)
            if n % 100 == 0:
                print(f"[ply-pack] {n}/{len(jobs)} shards ({time.time() - started:.0f}s) "
                      f"{builder.stats}", flush=True)
    arrays = builder.finish()
    print(f"[ply-pack] pack: {builder.stats}", flush=True)

    bounds = np.linspace(0, len(arrays["lens"]), PACK_PARTS + 1).astype(int)
    offsets = np.concatenate([[0], np.cumsum(arrays["lens"].astype(np.int64))])
    parts = []
    for i in range(PACK_PARTS):
        a, b = bounds[i], bounds[i + 1]
        piece = {k: arrays[k][a:b] for k in ("tokens", "lens", "rating", "index", "is_dev", "ply")}
        piece["actions"] = arrays["actions"][offsets[a]:offsets[b]]
        piece["winprob"] = arrays["winprob"][offsets[a]:offsets[b]]
        path = out / f"part-{i:02d}.npz"
        save_pack(path, piece)
        parts.append({"file": f"part-{i:02d}.npz", "sha256": file_sha256(path), "roots": int(b - a)})
    del arrays

    summary = {
        "format": PACK_FORMAT, "plan_sha256": file_sha256(plan_path),
        "source_sha256": plan["source_sha256"], "exclusion_sha256": plan["exclusion_sha256"],
        "val_csv_sha256": plan["val_csv_sha256"], "test_csv_sha256": plan["test_csv_sha256"],
        "roots_per_shard": ROOTS_PER_SHARD, "plies": args.plies, "pack_stats": builder.stats,
        "pack_parts": parts, "shards": {str(p): sorted(done[p])[: args.max_shards or None]
                                        for p in args.plies},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary["manifest_sha256"] = pack_manifest_digest(summary)
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[ply-pack] done in {time.time() - started:.0f}s", flush=True)

    if args.upload:
        from scripts.hf_upload import retry_hf_call
        retry_hf_call(lambda: api.upload_folder(
            folder_path=str(out), path_in_repo=PACK_PREFIX, repo_id=args.hf_repo,
            repo_type="dataset",
            commit_message=f"Add the later-ply training pack ({builder.stats['roots']:,} roots)"),
            "upload ply pack")
        print(f"[ply-pack] uploaded {PACK_PREFIX}", flush=True)


if __name__ == "__main__":
    main()
