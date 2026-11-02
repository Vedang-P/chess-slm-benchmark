"""Build and (optionally) upload the cont1 data: splits, tactical pack, dev slice.

Streams everything shard by shard, so peak disk is the source CSV (~300 MB)
plus the pack (~1 GB). Artifacts land under ``<workdir>/cont-v1`` and, with
``--upload``, under ``cont-v1/`` in the Hugging Face dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_data import (CSV_FIELDS, DEV_MODULUS, ROOTS_PER_SHARD, PackBuilder,  # noqa: E402
                               build_splits, extra_exclusion_hashes, save_pack,
                               write_split_csv)
from scripts.data_hygiene import position_hashes  # noqa: E402
from scripts.eval_gavn import tokenize_fen  # noqa: E402
from scripts.freeze_puzzle_corpus import file_sha256  # noqa: E402
from scripts.tactical_config import (DATA_PREFIX, HF_REPO, SOURCE_PREFIX,  # noqa: E402
                                     SOURCE_SHA256)

PACK_PARTS = 8
DEV_SHARD = "00000"


def hf_get(api, repo: str, path: str, workdir: Path, attempts: int = 5) -> Path:
    from huggingface_hub import hf_hub_download

    last = None
    for attempt in range(attempts):
        try:
            return Path(hf_hub_download(repo, path, repo_type="dataset",
                                        local_dir=str(workdir), token=api.token))
        except Exception as exc:  # transient Hub failures must not kill a long build
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed for {path}") from last


def mate_hashes(data_dir: Path) -> np.ndarray:
    names = ("mate-selection-test.json", "mate-selection-test-noexplain.json",
             "mate-selection-test-tactic.json", "mate-selection-test-both.json")
    tokens = []
    for name in names:
        rows = json.loads((data_dir / name).read_text(encoding="utf-8"))
        if len(rows) != 1000:
            raise ValueError(f"{name}: expected 1000 MATE rows")
        tokens.extend(tokenize_fen(r.get("fen") or r["position"]) for r in rows)
    return np.unique(position_hashes(np.asarray(tokens)))


def complete_shards(api, repo: str) -> list[int]:
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    out = []
    for tag in range(1230):
        prefix = f"{DATA_PREFIX}/shard-{tag:05d}"
        if f"{prefix}/train_set.npz" in files and f"{prefix}/meta.json" in files:
            out.append(tag)
    return out


def load_shard(api, repo: str, tag: int, workdir: Path):
    meta_path = hf_get(api, repo, f"{DATA_PREFIX}/shard-{tag:05d}/meta.json", workdir)
    npz_path = hf_get(api, repo, f"{DATA_PREFIX}/shard-{tag:05d}/train_set.npz", workdir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    identity = meta.get("identity") or {}
    if (not meta.get("complete") or identity.get("source_sha256") != SOURCE_SHA256
            or identity.get("engine") != "Stockfish 16"
            or identity.get("seconds_per_action") != 0.05
            or identity.get("max_roots") != 0 or identity.get("shard") != tag):
        raise ValueError(f"shard {tag:05d} failed the frozen identity gate")
    if file_sha256(npz_path) != meta.get("train_set_sha256"):
        raise ValueError(f"shard {tag:05d}: train_set.npz digest mismatch")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = (data["tokens"], data["actions"], data["winprob"])
    return meta, arrays, npz_path, meta_path


def build_broad_dev(api, repo: str, tags_file: Path, excluded: np.ndarray,
                    out: Path, rows_wanted: int, scan_rows: int = 3_000_000) -> dict:
    """Cache ~24k held-out broad rows (fold 0, benchmark-free) from the dev shard.

    Reads only a prefix of the shuffled shard via ranged reads, so the 4.3 GB
    teacher matrix is never downloaded in full.
    """
    from huggingface_hub import HfFileSystem
    from scripts.train_ccgavn import development_mask

    fs = HfFileSystem(token=api.token)
    base = f"datasets/{repo}/chessbench-full-build/shard-{DEV_SHARD}"

    def member(zf, name, rows, width=None, dtype=None):
        with zf.open(f"{name}.npy") as fh:
            version = np.lib.format.read_magic(fh)
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, _, dt = reader(fh)
            per = (width or 1) * dt.itemsize
            buf = fh.read(rows * per)
        arr = np.frombuffer(buf, dtype=dt)
        return arr.reshape(rows, width) if width else arr

    with fs.open(f"{base}/train_set.npz", "rb") as raw:
        zf = zipfile.ZipFile(raw)
        tokens = member(zf, "tokens", scan_rows, 77)
        actions = member(zf, "actions", scan_rows)
        winprob = member(zf, "winprob", scan_rows)
    fold = development_mask(tokens, DEV_MODULUS, 0)
    clean = fold & ~np.isin(position_hashes(tokens), excluded)
    rows = np.flatnonzero(clean)[:rows_wanted]
    if len(rows) < rows_wanted // 2:
        raise ValueError("dev slice too small; scan more rows")
    with fs.open(f"{base}/teacher_logp.npy", "rb") as fh:
        version = np.lib.format.read_magic(fh)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, _, dt = reader(fh)
        header = fh.tell()
        fh.seek(header)
        buf = fh.read(scan_rows * 128 * dt.itemsize)
    teacher = np.frombuffer(buf, dtype=dt).reshape(scan_rows, 128)[rows]
    norm = np.logaddexp.reduce(teacher[:1024].astype(np.float32), axis=1)
    if not np.allclose(norm, 0.0, atol=2e-3):
        raise ValueError("dev teacher rows are not normalized log-probabilities")
    np.savez_compressed(out, tokens=tokens[rows], actions=actions[rows],
                        winprob=winprob[rows].astype(np.float32), teacher=teacher)
    return {"rows": int(len(rows)), "scan_rows": scan_rows, "shard": DEV_SHARD}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--hf-repo", default=HF_REPO)
    p.add_argument("--val-size", type=int, default=4_000)
    p.add_argument("--test-size", type=int, default=4_000)
    p.add_argument("--broad-dev-rows", type=int, default=24_000)
    p.add_argument("--tags-file", type=Path, default=ROOT / "configs/ccgavn-2b-shard-tags.json")
    p.add_argument("--max-shards", type=int, default=0, help="smoke only")
    p.add_argument("--upload", action="store_true")
    args = p.parse_args()

    from scripts.kaggle_checkpoint import api as make_api
    api = make_api(ROOT)
    work = args.workdir
    out = work / "cont-v1"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    source = hf_get(api, args.hf_repo, f"{SOURCE_PREFIX}/source.csv.zst", work)
    manifest = json.loads(hf_get(api, args.hf_repo, f"{SOURCE_PREFIX}/manifest.json",
                                 work).read_text(encoding="utf-8"))
    if file_sha256(source) != SOURCE_SHA256 or manifest["source_sha256"] != SOURCE_SHA256:
        raise ValueError("puzzle source digest mismatch")
    excluded_ids = set(manifest["excluded_ids"])
    print(f"[data] source ok; building splits ({time.time()-started:.0f}s)", flush=True)

    splits = build_splits(source, excluded_ids, args.val_size, args.test_size)
    write_split_csv(out / "val.csv", splits["val"])
    write_split_csv(out / "test.csv", splits["test"])
    print(f"[data] val={len(splits['val'])} test={len(splits['test'])}", flush=True)

    # Benchmark positions to keep out of training: MATE rows, the matched
    # official puzzle lines, and every split puzzle line.
    official_rows = splits["official_rows"]
    excluded = np.unique(np.concatenate([
        mate_hashes(ROOT / "data/positions"),
        extra_exclusion_hashes(official_rows),
        extra_exclusion_hashes(splits["val"]),
        extra_exclusion_hashes(splits["test"]),
    ]))
    print(f"[data] {len(excluded):,} excluded position hashes", flush=True)

    shards = complete_shards(api, args.hf_repo)
    if args.max_shards:
        shards = shards[: args.max_shards]
    print(f"[data] {len(shards)} complete labeled shards", flush=True)
    builder = PackBuilder(splits["split"], splits["ratings"], excluded)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {tag: pool.submit(load_shard, api, args.hf_repo, tag, work / "shards")
                   for tag in shards[:12]}
        pending = list(shards[12:])
        for n, tag in enumerate(shards, 1):
            meta, (tokens, actions, winprob), npz_path, meta_path = futures.pop(tag).result()
            expected = min(ROOTS_PER_SHARD, manifest["included_roots"] - tag * ROOTS_PER_SHARD)
            builder.add_shard(tag, tokens, actions, winprob, expected)
            if int(meta["rows"]) != len(actions):
                raise ValueError(f"shard {tag:05d}: meta rows != array rows")
            npz_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            if pending:
                nxt = pending.pop(0)
                futures[nxt] = pool.submit(load_shard, api, args.hf_repo, nxt, work / "shards")
            if n % 100 == 0:
                print(f"[data] {n}/{len(shards)} shards ({time.time()-started:.0f}s) "
                      f"{builder.stats}", flush=True)
    arrays = builder.finish()
    print(f"[data] pack: {builder.stats}", flush=True)

    bounds = np.linspace(0, len(arrays["lens"]), PACK_PARTS + 1).astype(int)
    offsets = np.concatenate([[0], np.cumsum(arrays["lens"].astype(np.int64))])
    parts = []
    for i in range(PACK_PARTS):
        a, b = bounds[i], bounds[i + 1]
        piece = {k: arrays[k][a:b] for k in ("tokens", "lens", "rating", "index", "is_dev")}
        piece["actions"] = arrays["actions"][offsets[a]:offsets[b]]
        piece["winprob"] = arrays["winprob"][offsets[a]:offsets[b]]
        path = out / "pack" / f"part-{i:02d}.npz"
        save_pack(path, piece)
        parts.append({"file": f"pack/part-{i:02d}.npz", "sha256": file_sha256(path),
                      "roots": int(b - a)})
    del arrays

    dev_info = None
    if args.broad_dev_rows:
        dev_info = build_broad_dev(api, args.hf_repo, args.tags_file, excluded,
                                   out / "broad-dev.npz", args.broad_dev_rows)
        print(f"[data] broad dev slice {dev_info}", flush=True)

    summary = {
        "format": "cont-data-v1", "source_sha256": SOURCE_SHA256,
        "split_version": "cont-split-v1", "val_rows": len(splits["val"]),
        "test_rows": len(splits["test"]),
        "val_csv_sha256": file_sha256(out / "val.csv"),
        "test_csv_sha256": file_sha256(out / "test.csv"),
        "official_rating_hist": {str(k): v for k, v in sorted(splits["official_hist"].items())},
        "pack_stats": builder.stats, "pack_parts": parts, "shards": shards,
        "excluded_hashes": int(len(excluded)), "broad_dev": dev_info,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary["manifest_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[data] done in {time.time()-started:.0f}s", flush=True)

    if args.upload:
        from scripts.hf_upload import retry_hf_call
        retry_hf_call(lambda: api.upload_folder(
            folder_path=str(out), path_in_repo="cont-v1", repo_id=args.hf_repo,
            repo_type="dataset", commit_message="Add cont1 splits, tactical pack and dev slice"),
            "upload cont-v1")
        print("[data] uploaded cont-v1", flush=True)


if __name__ == "__main__":
    main()
