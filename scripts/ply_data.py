"""Later-ply puzzle roots (puzzle-ply-v1): enumeration, work plan and training pack.

A Lichess puzzle line is ``Moves[0]`` (the opponent's setup move) followed by
alternating solver and opponent moves, so the solver decides at odd indices. The
*ply-K root* is the board after pushing ``Moves[0..K-1]``: it exists when the line
has a stored solution move ``Moves[K]``. Ply 1 is the first-decision root that the
cont-v1 tactical pack already holds; ply 3 and 5 are the second and third decisions
on the solution line, which the model meets in the puzzle evaluation but never saw
densely supervised.

Shards are blocks of ROOTS_PER_SHARD *eligible* puzzles (not official, not
validation/test, a stored move at that ply) in source order, so the numbering is
fixed by cheap id checks alone. Positions whose reflection-orbit hash is in the
benchmark exclusion set are dropped inside the shard by the labeler.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts.cont_data import DEV_MODULUS, PACK_FIELDS, ROOTS_PER_SHARD, TacticalPack, iter_source

FORMAT = "puzzle-ply-v1"
PACK_FORMAT = "cont-ply-pack-v2"
PREFIX = "puzzle-ply-v1"
PACK_PREFIX = "cont-v2/ply-pack"
PLIES = (3, 5)
ENGINE = "Stockfish 16"
SECONDS_PER_ACTION = 0.05
PLY_PACK_FIELDS = PACK_FIELDS + ("ply",)


def check_ply(ply: int) -> int:
    if ply < 1 or ply % 2 == 0:
        raise ValueError(f"ply {ply} is not a solver decision (odd indices of Moves)")
    return int(ply)


def shard_dir(prefix: str, ply: int, shard: int) -> str:
    return f"{prefix.strip('/')}/ply{check_ply(ply)}/shard-{shard:05d}"


def read_ids(*csv_paths: Path) -> set[str]:
    import csv

    ids: set[str] = set()
    for path in csv_paths:
        with Path(path).open("r", encoding="utf-8", newline="") as fh:
            ids |= {row["PuzzleId"] for row in csv.DictReader(fh)}
    return ids


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ enumeration
def eligible_rows(source: Path, official_ids: set[str], skip_ids: set[str], ply: int):
    """Yield ``(included_index, row)`` of every puzzle with a ply-`ply` root, in source order.

    ``included_index`` numbers the rows that survive the official-id exclusion; that is
    the numbering of the cont-v1 split table and tactical pack, so ids stay comparable.
    Validation/test puzzles (`skip_ids`) keep their number but are never yielded.
    """
    check_ply(ply)
    included = -1
    for row in iter_source(source):
        puzzle_id = row["PuzzleId"]
        if puzzle_id in official_ids:
            continue
        included += 1
        if puzzle_id in skip_ids or len(row["Moves"].split()) <= ply:
            continue
        yield included, row


def shard_candidates(source: Path, official_ids: set[str], skip_ids: set[str], ply: int,
                     shard: int, roots_per_shard: int = ROOTS_PER_SHARD) -> list[tuple[int, int, dict]]:
    """``(ordinal, included_index, row)`` of one shard's candidates; empty past the corpus end."""
    if shard < 0 or roots_per_shard <= 0:
        raise ValueError("invalid shard index or size")
    start, end = shard * roots_per_shard, (shard + 1) * roots_per_shard
    out = []
    for ordinal, (included, row) in enumerate(eligible_rows(source, official_ids, skip_ids, ply)):
        if ordinal >= end:
            break
        if ordinal >= start:
            out.append((ordinal, included, row))
    return out


def count_eligible(source: Path, official_ids: set[str], skip_ids: set[str],
                   plies=PLIES) -> dict[int, int]:
    """Number of eligible puzzles per ply in one pass over the source."""
    counts = {check_ply(p): 0 for p in plies}
    for row in iter_source(source):
        puzzle_id = row["PuzzleId"]
        if puzzle_id in official_ids or puzzle_id in skip_ids:
            continue
        n = len(row["Moves"].split())
        for ply in counts:
            counts[ply] += int(n > ply)
    return counts


def shard_counts(eligible: dict[int, int], roots_per_shard: int = ROOTS_PER_SHARD) -> dict[int, int]:
    return {int(p): -(-int(n) // roots_per_shard) for p, n in eligible.items()}


def job_order(shards: dict[int, int]) -> list[tuple[int, int]]:
    """All ``(ply, shard)`` jobs, plies interleaved in proportion to their size.

    Every prefix of the order covers about the same fraction of each ply's range, so
    a partially finished fleet still holds both second and third decisions.
    """
    keyed = [(Fraction(2 * i + 1, 2 * n), ply, i)
             for ply, n in sorted(shards.items()) for i in range(n)]
    return [(ply, i) for _, ply, i in sorted(keyed)]


def worker_jobs(order: list[tuple[int, int]], worker: int, workers: int) -> list[tuple[int, int]]:
    if not 0 <= worker < workers:
        raise ValueError("worker index out of range")
    return order[worker::workers]


def plan_jobs(plan: dict) -> list[tuple[int, int]]:
    return job_order({int(p): int(v["shards"]) for p, v in plan["plies"].items()})


def shard_identity(*, source_sha256: str, ply: int, shard: int, max_roots: int,
                   val_csv_sha256: str, test_csv_sha256: str, exclusion_sha256: str,
                   roots_per_shard: int = ROOTS_PER_SHARD) -> dict:
    """Everything a partial or finished shard must agree on to be mixed with others."""
    return {"format": FORMAT, "source_sha256": source_sha256, "ply": check_ply(ply),
            "shard": int(shard), "roots_per_shard": int(roots_per_shard),
            "max_roots": int(max_roots), "engine": ENGINE,
            "seconds_per_action": SECONDS_PER_ACTION, "val_csv_sha256": val_csv_sha256,
            "test_csv_sha256": test_csv_sha256, "exclusion_sha256": exclusion_sha256}


def complete_shards(files: set[str], prefix: str, plies=PLIES) -> dict[int, set[int]]:
    """Shards whose finished artifacts are on the Hub (labels and meta, never a partial)."""
    out: dict[int, set[int]] = {int(p): set() for p in plies}
    for ply in out:
        head = f"{prefix.strip('/')}/ply{ply}/shard-"
        for name in files:
            if name.startswith(head) and name.endswith("/train_set.npz"):
                tag = name[len(head):-len("/train_set.npz")]
                if tag.isdigit() and f"{head}{tag}/meta.json" in files:
                    out[ply].add(int(tag))
    return out


# ------------------------------------------------------------------ pack
class PlyPackBuilder:
    """Accumulate finished ply shards into one per-root pack, re-checking hygiene.

    The labeler already drops benchmark positions, so any excluded root found here
    means a bad shard; it is an error, not something to filter out quietly.
    """

    def __init__(self, excluded_hashes: np.ndarray, excluded_ids: set[str],
                 dev_modulus: int = DEV_MODULUS):
        self.excluded = np.asarray(excluded_hashes, dtype=np.uint64)
        self.excluded_ids = excluded_ids
        self.dev_modulus = dev_modulus
        self.chunks: dict[str, list[np.ndarray]] = {k: [] for k in PLY_PACK_FIELDS}
        self.stats = {"shards": 0, "roots": 0, "kept_train": 0, "kept_dev": 0, "rows": 0,
                      "dropped_benchmark_by_labeler": 0, "by_ply": {}}

    def add_shard(self, ply: int, shard: int, data: dict[str, np.ndarray], meta: dict) -> None:
        from scripts.data_hygiene import position_hashes

        tokens, actions, winprob = data["tokens"], data["actions"], data["winprob"]
        lens = np.asarray(data["lens"], dtype=np.int64)
        if not (len(tokens) == len(actions) == len(winprob) == int(lens.sum())):
            raise ValueError(f"ply{ply} shard {shard}: row arrays disagree with root lengths")
        roots = len(lens)
        for key in ("index", "rating", "puzzle_id", "ply"):
            if len(data[key]) != roots:
                raise ValueError(f"ply{ply} shard {shard}: {key} has the wrong length")
        if int(meta["roots"]) != roots or int(meta["rows"]) != len(actions):
            raise ValueError(f"ply{ply} shard {shard}: meta counts disagree with the arrays")
        if np.any(np.asarray(data["ply"]) != ply):
            raise ValueError(f"ply{ply} shard {shard}: root ply labels disagree with the shard")
        starts = np.cumsum(lens) - lens
        root_tokens = np.asarray(tokens[starts], dtype=np.uint8)
        if not np.array_equal(np.repeat(root_tokens, lens, axis=0), tokens):
            raise ValueError(f"ply{ply} shard {shard}: a root's rows do not share one position")
        bad_ids = [i for i in data["puzzle_id"].tolist() if i in self.excluded_ids]
        if bad_ids:
            raise ValueError(f"ply{ply} shard {shard}: {len(bad_ids)} official/validation/test "
                             f"puzzles, e.g. {bad_ids[0]}")
        hashes = position_hashes(root_tokens)
        if np.isin(hashes, self.excluded).any():
            raise ValueError(f"ply{ply} shard {shard}: benchmark positions survived labeling")
        is_dev = (hashes % np.uint64(self.dev_modulus)) == 0
        self.chunks["tokens"].append(root_tokens)
        self.chunks["lens"].append(lens.astype(np.int32))
        self.chunks["actions"].append(np.asarray(actions, dtype=np.uint16))
        self.chunks["winprob"].append(np.asarray(winprob, dtype=np.float16))
        self.chunks["rating"].append(np.asarray(data["rating"], dtype=np.uint16))
        self.chunks["index"].append(np.asarray(data["index"], dtype=np.uint32))
        self.chunks["is_dev"].append(is_dev)
        self.chunks["ply"].append(np.full(roots, ply, dtype=np.uint8))
        self.stats["shards"] += 1
        self.stats["roots"] += roots
        self.stats["rows"] += int(len(actions))
        self.stats["kept_dev"] += int(is_dev.sum())
        self.stats["kept_train"] += int((~is_dev).sum())
        self.stats["dropped_benchmark_by_labeler"] += int(meta.get("dropped_benchmark", 0))
        by_ply = self.stats["by_ply"].setdefault(str(ply), {"shards": 0, "roots": 0})
        by_ply["shards"] += 1
        by_ply["roots"] += roots

    def finish(self) -> dict[str, np.ndarray]:
        if not self.chunks["lens"]:
            raise ValueError("pack builder received no roots")
        return {k: np.concatenate(v) for k, v in self.chunks.items()}


class PlyPack(TacticalPack):
    """The tactical pack layout plus the ply of every root."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        if "ply" not in arrays:
            raise ValueError("ply pack lacks the ply field")
        super().__init__(arrays)
        self.ply = np.asarray(arrays["ply"], dtype=np.uint8)
        if len(self.ply) != len(self.lens):
            raise ValueError("ply pack arrays are inconsistent")

    @classmethod
    def load(cls, paths) -> "PlyPack":
        parts = []
        for p in paths:
            with np.load(p, allow_pickle=False) as data:
                parts.append({k: data[k] for k in PLY_PACK_FIELDS})
        return cls({k: np.concatenate([p[k] for p in parts]) for k in PLY_PACK_FIELDS})


def pack_manifest_digest(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return sha256_bytes(json.dumps(body, sort_keys=True).encode())
