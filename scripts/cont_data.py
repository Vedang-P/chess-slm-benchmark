"""Data for the conservative CC-GAVN continuation (cont1).

Three things live here:

* held-out *validation* and sealed *test* puzzle sets drawn from the Lichess
  puzzle database. They are rating-matched to the official 10,000 puzzles,
  disjoint from them, and every position on their solution lines is excluded
  from training;
* the *tactical pack*: each labeled puzzle root once, as per-root tokens plus
  the concatenated legal actions and Stockfish win probabilities. Split
  puzzles, benchmark positions and a 1% position-hash dev fold are removed
  when the pack is built;
* stateless samplers. Every batch is a pure function of (seed, step), so a
  resumed session reproduces the exact data stream without stored RNG state.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

SPLIT_VERSION = "cont-split-v1"
ROOTS_PER_SHARD = 5_000
RATING_BIN = 200
CSV_FIELDS = ("PuzzleId", "FEN", "Moves", "Rating")
VAL_BUCKET, TEST_BUCKET = 7, 13        # puzzle_hash % 100 pools; disjoint
DEV_MODULUS = 100                      # position-hash fold held out of the pack
PACK_FIELDS = ("tokens", "lens", "actions", "winprob", "rating", "index", "is_dev")


def puzzle_hash(puzzle_id: str) -> int:
    digest = hashlib.sha256(f"{SPLIT_VERSION}:{puzzle_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rating_bin(rating: int) -> int:
    return max(0, int(rating)) // RATING_BIN


def iter_source(path: Path):
    """Stream rows of the zstd-compressed Lichess puzzle CSV."""
    import zstandard as zstd

    with Path(path).open("rb") as raw:
        stream = zstd.ZstdDecompressor().stream_reader(raw)
        with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
            yield from csv.DictReader(text)


def largest_remainder(weights: dict[int, int], total: int) -> dict[int, int]:
    """Apportion `total` across keys proportionally to `weights` (exact sum)."""
    mass = sum(weights.values())
    if total < 0 or mass <= 0:
        raise ValueError("cannot apportion with empty weights or a negative total")
    exact = {k: total * v / mass for k, v in weights.items()}
    base = {k: int(np.floor(x)) for k, x in exact.items()}
    order = sorted(weights, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[: total - sum(base.values())]:
        base[k] += 1
    return base


def build_splits(source: Path, excluded_ids: set[str], val_size: int = 4_000,
                 test_size: int = 4_000) -> dict:
    """Pick rating-matched validation/test puzzles disjoint from the official set.

    One pass over the source in included order (official ids skipped), so the
    returned per-root arrays line up with the labeled shards' root numbering.
    """
    official_hist: dict[int, int] = {}
    official_rows: list[dict] = []
    pools: dict[str, dict[int, list]] = {"val": {}, "test": {}}
    ratings: list[int] = []
    for row in iter_source(source):
        pid, rating = row["PuzzleId"], int(row["Rating"])
        if pid in excluded_ids:
            b = rating_bin(rating)
            official_hist[b] = official_hist.get(b, 0) + 1
            official_rows.append({k: row[k] for k in CSV_FIELDS})
            continue
        index = len(ratings)
        ratings.append(rating)
        h = puzzle_hash(pid)
        name = {VAL_BUCKET: "val", TEST_BUCKET: "test"}.get(h % 100)
        if name:
            pools[name].setdefault(rating_bin(rating), []).append(
                (h, index, {k: row[k] for k in CSV_FIELDS}))
    if not official_hist:
        raise ValueError("no official puzzle ids matched the source")
    split = np.zeros(len(ratings), dtype=np.uint8)
    chosen: dict[str, list[dict]] = {}
    for code, (name, size) in enumerate((("val", val_size), ("test", test_size)), 1):
        targets = largest_remainder(official_hist, size)
        picked = []
        for b, want in targets.items():
            for h, index, row in sorted(pools[name].get(b, []))[:want]:
                picked.append((h, index, row))
        picked.sort()
        for _, index, _ in picked:
            split[index] = code
        chosen[name] = [row for _, _, row in picked]
    return {"val": chosen["val"], "test": chosen["test"], "split": split,
            "ratings": np.asarray(ratings, dtype=np.uint16),
            "official_hist": official_hist, "official_rows": official_rows}


def write_split_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_split_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def line_tokens(fen: str, moves: list[str]) -> np.ndarray:
    """Tokens of the start position and of the position after every move."""
    import chess
    from scripts.eval_gavn import tokenize_fen

    board = chess.Board(fen)
    out = [tokenize_fen(board.fen())]
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci} in line from {fen}")
        board.push(move)
        out.append(tokenize_fen(board.fen()))
    return np.asarray(out)


def extra_exclusion_hashes(rows: list[dict]) -> np.ndarray:
    """Reflection-orbit hashes of every position on the given puzzle lines."""
    from scripts.data_hygiene import position_hashes

    tokens = [line_tokens(r["FEN"], r["Moves"].split()) for r in rows]
    if not tokens:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(position_hashes(np.concatenate(tokens)))


# ---------------------------------------------------------------- tactical pack
def group_starts(tokens: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """First row of every root: position tokens change, or an action repeats.

    Labeled shards store every legal action of a root consecutively with the
    same token row. Two adjacent roots with an identical position would merge;
    a repeated action inside one group exposes that and splits it.
    """
    n = len(tokens)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = np.any(tokens[1:] != tokens[:-1], axis=1)
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)
    extra = []
    for s, e in zip(starts, ends):
        seen: set[int] = set()
        for i in range(int(s), int(e)):
            a = int(actions[i])
            if a in seen:
                extra.append(i)
                seen = set()
            seen.add(a)
    if extra:
        starts = np.unique(np.concatenate([starts, np.asarray(extra, dtype=np.int64)]))
    return starts


class PackBuilder:
    """Accumulate labeled shards into one per-root pack, applying exclusions."""

    def __init__(self, split: np.ndarray, ratings: np.ndarray,
                 excluded_hashes: np.ndarray, dev_modulus: int = DEV_MODULUS):
        self.split, self.ratings = split, ratings
        self.excluded = np.asarray(excluded_hashes, dtype=np.uint64)
        self.dev_modulus = dev_modulus
        self.chunks: dict[str, list[np.ndarray]] = {k: [] for k in PACK_FIELDS}
        self.stats = {"roots_seen": 0, "dropped_split": 0, "dropped_benchmark": 0,
                      "kept_train": 0, "kept_dev": 0, "shards": 0}

    def add_shard(self, shard: int, tokens: np.ndarray, actions: np.ndarray,
                  winprob: np.ndarray, expected_roots: int) -> None:
        from scripts.data_hygiene import position_hashes

        if not (len(tokens) == len(actions) == len(winprob)):
            raise ValueError(f"shard {shard}: row arrays disagree in length")
        starts = group_starts(tokens, actions)
        if len(starts) != expected_roots:
            raise ValueError(f"shard {shard}: found {len(starts)} roots, "
                             f"expected {expected_roots}")
        ends = np.append(starts[1:], len(tokens))
        lens = ends - starts
        index = shard * ROOTS_PER_SHARD + np.arange(len(starts))
        if index.max() >= len(self.split):
            raise ValueError(f"shard {shard} exceeds the split table")
        hashes = position_hashes(tokens[starts])
        in_split = self.split[index] != 0
        in_bench = np.isin(hashes, self.excluded)
        keep = ~in_split & ~in_bench
        is_dev = (hashes % np.uint64(self.dev_modulus)) == 0
        kept = np.flatnonzero(keep)
        self.stats["roots_seen"] += len(starts)
        self.stats["dropped_split"] += int(in_split.sum())
        self.stats["dropped_benchmark"] += int((in_bench & ~in_split).sum())
        self.stats["kept_dev"] += int(is_dev[kept].sum())
        self.stats["kept_train"] += int((~is_dev[kept]).sum())
        self.stats["shards"] += 1
        if not len(kept):
            return
        kl = lens[kept]
        row_idx = (np.repeat(starts[kept] - np.cumsum(kl) + kl, kl)
                   + np.arange(int(kl.sum())))
        self.chunks["tokens"].append(np.asarray(tokens[starts[kept]], dtype=np.uint8))
        self.chunks["lens"].append(kl.astype(np.int32))
        self.chunks["actions"].append(np.asarray(actions[row_idx], dtype=np.uint16))
        self.chunks["winprob"].append(np.asarray(winprob[row_idx], dtype=np.float16))
        self.chunks["rating"].append(self.ratings[index[kept]])
        self.chunks["index"].append(index[kept].astype(np.uint32))
        self.chunks["is_dev"].append(is_dev[kept])

    def finish(self) -> dict[str, np.ndarray]:
        if not self.chunks["lens"]:
            raise ValueError("pack builder received no roots")
        return {k: np.concatenate(v) for k, v in self.chunks.items()}


def save_pack(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


class TacticalPack:
    """All kept puzzle roots in RAM; rows of a root are contiguous."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        missing = set(PACK_FIELDS) - set(arrays)
        if missing:
            raise ValueError(f"pack lacks fields: {sorted(missing)}")
        self.tokens = arrays["tokens"]
        self.lens = arrays["lens"].astype(np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lens)]).astype(np.int64)
        self.actions = arrays["actions"]
        self.winprob = arrays["winprob"]
        self.rating = arrays["rating"]
        self.index = arrays["index"]
        self.is_dev = arrays["is_dev"].astype(bool)
        if (len(self.tokens) != len(self.lens) or self.offsets[-1] != len(self.actions)
                or len(self.actions) != len(self.winprob)):
            raise ValueError("tactical pack arrays are inconsistent")
        self.train_roots = np.flatnonzero(~self.is_dev)
        self.dev_roots = np.flatnonzero(self.is_dev)

    @classmethod
    def load(cls, paths) -> "TacticalPack":
        parts = []
        for p in paths:
            with np.load(p, allow_pickle=False) as data:
                parts.append({k: data[k] for k in PACK_FIELDS})
        return cls({k: np.concatenate([p[k] for p in parts]) for k in PACK_FIELDS})

    def __len__(self) -> int:
        return len(self.lens)

    def gather(self, roots: np.ndarray):
        """Rows of `roots` (in order): tokens, actions, winprob, group ids."""
        roots = np.asarray(roots, dtype=np.int64)
        lens = self.lens[roots]
        starts = self.offsets[roots]
        total = int(lens.sum())
        rows = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total)
        tokens = np.repeat(self.tokens[roots], lens, axis=0)
        gid = np.repeat(np.arange(len(roots)), lens)
        return (tokens, self.actions[rows].astype(np.int64),
                self.winprob[rows].astype(np.float32), gid, lens)


# ---------------------------------------------------------------- stateless RNG
def step_rng(seed: int, step: int, stream: int) -> np.random.Generator:
    """Independent, reproducible generator for (seed, step, stream)."""
    return np.random.default_rng([int(seed), int(step), int(stream)])


def broad_block_tag(step: int, tags: list[str], block_steps: int, seed: int) -> str:
    """Shard visited at `step`: one shard per block, fresh permutation each epoch."""
    if block_steps <= 0 or not tags:
        raise ValueError("block_steps and tags must be positive")
    epoch, pos = divmod(step // block_steps, len(tags))
    perm = np.random.default_rng([int(seed), 7919, epoch]).permutation(len(tags))
    return tags[int(perm[pos])]


def draw_rows(rng: np.random.Generator, size: int, n: int,
              train_mask: np.ndarray | None) -> np.ndarray:
    """Uniform rows from a shard, rejecting rows outside the training mask."""
    idx = rng.integers(0, size, size=n)
    if train_mask is not None:
        for _ in range(64):
            bad = ~train_mask[idx]
            if not bad.any():
                break
            idx[bad] = rng.integers(0, size, size=int(bad.sum()))
        else:
            raise RuntimeError("train mask rejects nearly every row")
    return idx


def manifest_digest(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
