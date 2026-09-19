"""Shared multi-shard data manager for the sharded ChessBench training set.

The 8 ChessBench shards live on HF as
``<prefix>/shard-<tag>/train_set.npz`` (tokens/actions/winprob) and
``<prefix>/shard-<tag>/teacher_logp.npy`` (9M teacher, fp16). Both trainers
must sample from ALL shards, not just shard 0.

Strategy (12h-resumable, Kaggle-safe, 2B-scale safe):
  - row counts come from exact overrides (configs/chessbench_rows.json) or a
    raw-bag size estimate via HTTP HEAD (~80 bytes/row, calibrated), so no
    full-corpus download is needed before training,
  - download ONE shard on demand when the schedule reaches it, delete the
    previous shard's files (schedule blocks are contiguous per shard),
  - materialize one shard's arrays at a time (~1.4GB RAM), freeing the
    previous segment; the teacher .npy files are always true memmaps.
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

GCS_BASE = "https://storage.googleapis.com/searchless_chess/data/train"
BYTES_PER_ROW = 80.0  # calibrated on shard 00000 (1,341,410,000 B / 16,764,154 rows)
ROWS_OVERRIDE = Path(__file__).resolve().parent.parent / "configs" / "chessbench_rows.json"


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
                name = f.split("/")[1]
                tags.add(name[len("shard-"):])
        self.tags = sorted(tags)
        if len(self.tags) < expect_tags:
            raise RuntimeError(
                f"{repo}:{prefix} has {len(self.tags)} shards, expected {expect_tags}")
        self.rows: dict[str, int] = {}
        self.total = 0

    # ---- download (streaming: one shard at a time, evict the previous) ----
    def _dest(self, tag: str, fname: str) -> Path:
        return self.cache_dir / f"shard-{tag}-{fname}"

    def _download(self, tag: str) -> None:
        from huggingface_hub import hf_hub_download
        for fname in ("train_set.npz", "teacher_logp.npy"):
            dest = self._dest(tag, fname)
            if dest.exists() and self._verify(dest, tag, fname):
                continue
            if dest.exists():
                self.log(f"[shards] {tag}/{fname} failed integrity check; re-downloading")
                dest.unlink()
            t0 = time.time()
            self.log(f"[shards] downloading {prefix_str(self.prefix, tag, fname)}")
            hf_hub_download(repo_id=self.repo, repo_type="dataset",
                            filename=f"{self.prefix}/shard-{tag}/{fname}",
                            local_dir=str(self.cache_dir), token=self.token)
            cached = self.cache_dir / self.prefix / f"shard-{tag}" / fname
            if cached.exists():
                cached.replace(dest)
                shutil.rmtree(self.cache_dir / self.prefix, ignore_errors=True)
            if not self._verify(dest, tag, fname):
                raise RuntimeError(f"[shards] {tag}/{fname} failed integrity check after download")
            mb = dest.stat().st_size / 1e6
            self.log(f"[shards] {tag}/{fname}: {mb:.0f}MB in {time.time()-t0:.0f}s")

    def _evict(self, tag: str) -> None:
        for fname in ("train_set.npz", "teacher_logp.npy"):
            self._dest(tag, fname).unlink(missing_ok=True)

    def ensure_downloaded(self, tags=None, max_records: int = 0) -> None:
        """Smoke runs prefetch the first shard; full runs download on demand."""
        if max_records:
            wanted = (self.tags if tags is None else list(tags))[:1]
            for tag in wanted:
                self._download(tag)

    def _verify(self, dest: Path, tag: str, fname: str) -> bool:
        try:
            if fname.endswith(".npz"):
                npz_member_shape(dest, "tokens")
            else:
                npy_shape(dest)
            return True
        except Exception:
            return False

    # ---- row counts (no download required) ----
    @staticmethod
    def _override_rows() -> dict:
        try:
            return {str(k): int(v)
                    for k, v in json.loads(ROWS_OVERRIDE.read_text()).items()}
        except Exception:
            return {}

    def _gcs_rows(self, tag: str) -> int:
        url = f"{GCS_BASE}/action_value-{tag}-of-02148_data.bag"
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            size = int(r.headers.get("Content-Length", 0))
        if size <= 0:
            raise ValueError(f"no content length for shard {tag}")
        return int(size / BYTES_PER_ROW)

    def count_rows(self, tags=None, max_records: int = 0) -> None:
        wanted = self.tags if tags is None else list(tags)
        if max_records:
            wanted = wanted[:1]
        overrides = self._override_rows()
        for tag in wanted:
            if tag in self.rows:
                continue
            if tag in overrides:
                self.rows[tag] = overrides[tag]
                continue
            try:
                self.rows[tag] = self._gcs_rows(tag)
            except Exception:
                # offline / GCS unavailable: fall back to a local header read
                try:
                    shape = npz_member_shape(self._dest(tag, "train_set.npz"), "tokens")
                    self.rows[tag] = int(shape[0])
                except Exception as exc:
                    raise RuntimeError(f"cannot determine row count for shard {tag}: {exc}")
        self.total = sum(self.rows.values())

    # ---- schedule ----
    def schedule(self, steps: int, rng: np.random.Generator,
                 max_records: int = 0) -> np.ndarray:
        """Deterministic step -> tag array. Steps are allocated ~ rows per
        shard as CONTIGUOUS blocks (one download per shard), only the block
        order is shuffled by the seeded rng, so the schedule is identical on
        resume."""
        wanted = list(self.tags if not max_records else self.tags[:1])
        rows = np.array([self.rows[t] for t in wanted], dtype=np.float64)
        if steps < len(wanted):
            return np.array([wanted[i % len(wanted)] for i in range(steps)])
        frac = rows / rows.sum()
        counts = np.maximum(1, np.round(frac * steps).astype(int))
        counts[int(np.argmax(counts))] += steps - counts.sum()
        order = rng.permutation(len(wanted))
        sched = np.concatenate([np.repeat(wanted[i], counts[i]) for i in order])
        return sched

    # ---- materialize one shard ----
    def load(self, tag: str, max_records: int = 0):
        if getattr(self, "_current_tag", None) and self._current_tag != tag:
            self._evict(self._current_tag)
        self._download(tag)
        self._current_tag = tag
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
