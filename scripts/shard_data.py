"""Shared multi-shard data manager for the sharded ChessBench training set.

The 8 ChessBench shards live on HF as
``<prefix>/shard-<tag>/train_set.npz`` (tokens/actions/winprob) and
``<prefix>/shard-<tag>/teacher_logp.npy`` (9M teacher, fp16). Both trainers
must sample from ALL shards, not just shard 0.

Strategy (12h-resumable, Kaggle-safe):
  - download every shard file once into a cache dir (skips files already
    present; sequential with progress prints),
  - read row counts from the npz/npy headers WITHOUT decompressing,
  - build a deterministic step -> shard schedule that allocates steps
    proportionally to shard rows (recomputed identically on resume),
  - materialize one shard's arrays at a time (~1.4GB RAM), freeing the
    previous segment; the teacher .npy files are always true memmaps.
"""
from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

import numpy as np


def _npy_header_shape(fh) -> tuple:
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(fh)
    elif version == (2, 0):
        shape, _, _ = np.lib.format.read_array_header_2_0(fh)
    else:
        raise ValueError(f"unsupported npy version {version}")
    return shape


def npz_member_shape(path: Path, member: str) -> tuple:
    """Row shape of one member of a .npz without decompressing the payload."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for candidate in (f"{member}.npy", member):
            if candidate in names:
                with z.open(candidate) as fh:
                    return _npy_header_shape(fh)
        raise KeyError(f"member {member} not found in {path} (has {sorted(names)})")


def npy_shape(path: Path) -> tuple:
    with open(path, "rb") as fh:
        return _npy_header_shape(fh)


class ShardManager:
    def __init__(self, repo: str, prefix: str, cache_dir: Path,
                 token: str | None = None, log=print, expect_tags: int = 8):
        from huggingface_hub import HfApi
        self.repo, self.prefix, self.cache_dir = repo, prefix, Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.log = log
        files = HfApi(token=token).list_repo_files(repo_id=repo, repo_type="dataset")
        tags = set()
        for f in files:
            if f.startswith(f"{prefix}/shard-") and f.endswith("train_set.npz"):
                tags.add(f.split("/")[1])
        self.tags = sorted(tags)
        if len(self.tags) < expect_tags:
            raise RuntimeError(
                f"{repo}:{prefix} has {len(self.tags)} shards, expected {expect_tags}")
        self.rows: dict[str, int] = {}
        self.total = 0

    # ---- download ----
    def _dest(self, tag: str, fname: str) -> Path:
        return self.cache_dir / f"shard-{tag}-{fname}"

    def ensure_downloaded(self, tags=None, max_records: int = 0) -> None:
        """Download the requested shards (all by default; shard 0 only when
        max_records is set for a smoke run)."""
        from huggingface_hub import hf_hub_download
        wanted = self.tags if tags is None else list(tags)
        if max_records:
            wanted = wanted[:1]
        for tag in wanted:
            for fname in ("train_set.npz", "teacher_logp.npy"):
                dest = self._dest(tag, fname)
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                t0 = time.time()
                self.log(f"[shards] downloading {prefix_str(self.prefix, tag, fname)}")
                hf_hub_download(repo_id=self.repo, repo_type="dataset",
                                filename=f"{self.prefix}/shard-{tag}/{fname}",
                                local_dir=str(self.cache_dir), token=self.token)
                cached = self.cache_dir / self.prefix / f"shard-{tag}" / fname
                if cached.exists():
                    cached.replace(dest)
                    shutil.rmtree(self.cache_dir / self.prefix, ignore_errors=True)
                mb = dest.stat().st_size / 1e6
                self.log(f"[shards] {tag}/{fname}: {mb:.0f}MB in {time.time()-t0:.0f}s")

    # ---- row counts (no decompress) ----
    def count_rows(self, tags=None, max_records: int = 0) -> None:
        wanted = self.tags if tags is None else list(tags)
        if max_records:
            wanted = wanted[:1]
        for tag in wanted:
            if tag in self.rows:
                continue
            shape = npz_member_shape(self._dest(tag, "train_set.npz"), "tokens")
            self.rows[tag] = int(shape[0])
        self.total = sum(self.rows.values())

    # ---- schedule ----
    def schedule(self, steps: int, rng: np.random.Generator,
                 max_records: int = 0) -> np.ndarray:
        """Deterministic step -> tag array, steps allocated ~ rows per shard."""
        wanted = self.tags if not max_records else self.tags[:1]
        rows = np.array([self.rows[t] for t in wanted], dtype=np.float64)
        frac = rows / rows.sum()
        counts = np.maximum(1, np.round(frac * steps).astype(int))
        counts[-1] += steps - counts.sum()
        sched = np.repeat(np.array(wanted), counts)
        rng.shuffle(sched)  # segment order varies by seed; identical on resume
        return sched

    # ---- materialize one shard ----
    def load(self, tag: str, max_records: int = 0):
        tokens_f = self._dest(tag, "train_set.npz")
        d = np.load(tokens_f, mmap_mode="r")
        tokens = d["tokens"]
        actions = d["actions"]
        winprob = np.asarray(d["winprob"], dtype=np.float32) if "winprob" in d else None
        teacher_f = self._dest(tag, "teacher_logp.npy")
        t = np.load(teacher_f, mmap_mode="r")
        teacher = t["teacher_logp"] if isinstance(t, np.lib.npyio.NpzFile) else t
        if max_records:
            tokens, actions = tokens[:max_records], actions[:max_records]
            if winprob is not None:
                winprob = winprob[:max_records]
            teacher = teacher[:max_records]
        return tokens, actions, winprob, teacher


def prefix_str(prefix: str, tag: str, fname: str) -> str:
    return f"{prefix}/shard-{tag}/{fname}"
