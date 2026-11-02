"""Tests for the cont1 data layer, evaluator, loss and trainer (CPU, tiny model)."""
import contextlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chess
import numpy as np

from scripts import cont_data, cont_eval
from scripts.eval_gavn import tokenize_fen

SL_REPO = Path(os.environ.get(
    "SL_REPO", "/private/tmp/claude-501/-Users-vedang--claude-spares-spare-9R7egg/"
               "bf54a92f-3381-494b-9ad0-cb14b8de1788/scratchpad/searchless_chess"))


def load_utils():
    """Load the pinned pure-Python action table by file path (no package import)."""
    path = SL_REPO / "src" / "utils.py"
    if not path.exists():
        raise unittest.SkipTest(f"pinned searchless utils not found at {path}")
    spec = importlib.util.spec_from_file_location("sl_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def action_tables_from(utils):
    src, dst, promo = [], [], []
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    for action in range(utils.NUM_ACTIONS):
        uci = utils.ACTION_TO_MOVE[action]
        s, d = chess.parse_square(uci[:2]), chess.parse_square(uci[2:4])
        src.append((7 - s // 8) * 8 + s % 8)
        dst.append((7 - d // 8) * 8 + d % 8)
        promo.append(promo_map.get(uci[4:].lower(), 0))
    return np.asarray(src), np.asarray(dst), np.asarray(promo), utils


def random_line(rng, plies):
    board = chess.Board()
    fen, moves = board.fen(), []
    for _ in range(plies):
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        move = legal[int(rng.integers(len(legal)))]
        moves.append(move.uci())
        board.push(move)
    return fen, moves


def synthetic_shard(rng, roots):
    """Rows of `roots` random positions, every legal move labeled (ids unique per root)."""
    tokens, actions, winprob = [], [], []
    for r in range(roots):
        fen, moves = random_line(rng, 4 + r % 5)
        board = chess.Board(fen)
        for m in moves:
            board.push_uci(m)
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        tok = tokenize_fen(board.fen())
        for i in range(len(legal)):
            tokens.append(tok)
            actions.append(i)
            winprob.append(float(rng.random()))
    return (np.asarray(tokens, dtype=np.uint8), np.asarray(actions, dtype=np.int64),
            np.asarray(winprob, dtype=np.float32))


def random_teacher(rng, n):
    logits = rng.normal(size=(n, 128)).astype(np.float32)
    return logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)


class FakeScorer:
    """Deterministic pseudo-values per (position, move); mimics BatchedScorer."""

    def __init__(self, utils, prefer=None):
        self.utils, self.prefer = utils, prefer or {}

    def __call__(self, boards):
        out = []
        for board in boards:
            moves = cont_eval.ordered_legal_moves(board, self.utils.MOVE_TO_ACTION)
            key = " ".join(board.fen().split()[:4])
            vals = np.array([(hash((key, m.uci())) % 997) / 997.0 for m in moves], dtype=np.float32)
            for i, m in enumerate(moves):
                if self.prefer.get(key) == m.uci():
                    vals[i] = 2.0
                board.push(m)
                if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                    vals[i] = 0.5
                board.pop()
            out.append((moves, vals))
        return out


def reference_puzzle(scorer, fen, moves_uci):
    """Line-by-line copy of the sequential loop in scripts/eval_gavn.py."""
    board = chess.Board(fen)
    ok = True
    for j, uci in enumerate(moves_uci):
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        if j % 2 == 1:
            legal, values = scorer([board])[0]
            predicted = legal[int(np.argmax(values))].uci()
            if predicted != uci:
                board.push(chess.Move.from_uci(predicted))
                ok = board.is_checkmate()
                break
        board.push(move)
    return ok


class DataTests(unittest.TestCase):
    def test_largest_remainder_is_exact_and_proportional(self):
        self.assertEqual(cont_data.largest_remainder({1: 5, 2: 3, 3: 2}, 10), {1: 5, 2: 3, 3: 2})
        self.assertEqual(sum(cont_data.largest_remainder({1: 1, 2: 1, 3: 1}, 4).values()), 4)

    def test_split_hash_is_stable_and_pools_are_disjoint(self):
        self.assertEqual(cont_data.puzzle_hash("00abc"), cont_data.puzzle_hash("00abc"))
        ids = [f"id{i}" for i in range(2000)]
        val = {i for i in ids if cont_data.puzzle_hash(i) % 100 == cont_data.VAL_BUCKET}
        test = {i for i in ids if cont_data.puzzle_hash(i) % 100 == cont_data.TEST_BUCKET}
        self.assertTrue(val and test and not (val & test))

    def test_group_starts_splits_identical_adjacent_roots(self):
        a = tokenize_fen(chess.Board().fen())
        board = chess.Board()
        board.push_uci("e2e4")
        b = tokenize_fen(board.fen())
        tokens = np.vstack([np.tile(a, (3, 1)), np.tile(a, (3, 1)), np.tile(b, (2, 1))])
        actions = np.array([1, 2, 3, 1, 2, 3, 7, 8])
        self.assertEqual(cont_data.group_starts(tokens, actions).tolist(), [0, 3, 6])

    def test_pack_builder_filters_exclusions_and_gather_round_trips(self):
        from scripts.data_hygiene import position_hashes
        rng = np.random.default_rng(3)
        tokens, actions, winprob = synthetic_shard(rng, roots=12)
        starts = cont_data.group_starts(tokens, actions)
        self.assertEqual(len(starts), 12)
        split = np.zeros(12, dtype=np.uint8)
        split[1] = 1                                   # a validation puzzle
        bench = position_hashes(tokens[starts[[2]]])   # a benchmark position
        builder = cont_data.PackBuilder(split, np.arange(12, dtype=np.uint16) + 800, bench)
        builder.add_shard(0, tokens, actions, winprob, expected_roots=12)
        pack = cont_data.TacticalPack(builder.finish())
        self.assertEqual(len(pack), 10)
        self.assertEqual(builder.stats["dropped_split"], 1)
        self.assertEqual(builder.stats["dropped_benchmark"], 1)
        self.assertNotIn(1, pack.index.tolist())
        self.assertNotIn(2, pack.index.tolist())
        ends = np.append(starts[1:], len(tokens))
        for local, root_index in enumerate(pack.index.tolist()):
            t, a, w, gid, lens = pack.gather(np.array([local]))
            sl = slice(starts[root_index], ends[root_index])
            np.testing.assert_array_equal(t, tokens[sl])
            np.testing.assert_array_equal(a, actions[sl])
            np.testing.assert_allclose(w, winprob[sl], atol=1e-3)   # float16 storage
            self.assertEqual(int(lens[0]), ends[root_index] - starts[root_index])
        t, a, w, gid, lens = pack.gather(np.array([0, 3, 5]))
        self.assertEqual(len(a), int(lens.sum()))
        self.assertEqual(np.bincount(gid).tolist(), lens.tolist())

    def test_stateless_sampling_is_reproducible_and_covers_every_shard(self):
        tags = [f"{i:05d}" for i in range(7)]
        seq = [cont_data.broad_block_tag(s, tags, 10, seed=4) for s in range(0, 70, 10)]
        self.assertEqual(sorted(seq), tags)
        self.assertEqual(seq, [cont_data.broad_block_tag(s, tags, 10, seed=4) for s in range(0, 70, 10)])
        self.assertNotEqual(seq, [cont_data.broad_block_tag(s, tags, 10, seed=5) for s in range(0, 70, 10)])
        a = cont_data.draw_rows(cont_data.step_rng(1, 9, 1), 1000, 64, None)
        b = cont_data.draw_rows(cont_data.step_rng(1, 9, 1), 1000, 64, None)
        np.testing.assert_array_equal(a, b)
        mask = np.zeros(1000, dtype=bool)
        mask[::3] = True
        self.assertTrue(mask[cont_data.draw_rows(cont_data.step_rng(1, 9, 1), 1000, 256, mask)].all())


class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils = load_utils()

    def test_batched_evaluator_matches_sequential_reference(self):
        rng = np.random.default_rng(11)
        puzzles = []
        for i in range(60):
            fen, moves = random_line(rng, 4 + (i % 3) * 2)
            puzzles.append({"id": str(i), "fen": fen, "moves": moves})
        scorer = FakeScorer(self.utils)
        batched = cont_eval.evaluate_puzzles(scorer, puzzles, chunk=7)
        expected = [reference_puzzle(scorer, p["fen"], p["moves"]) for p in puzzles]
        self.assertEqual([r["correct"] for r in batched], expected)
        self.assertEqual([r["id"] for r in batched], [p["id"] for p in puzzles])

    def test_alternative_mate_is_accepted_and_other_moves_are_not(self):
        fen = "6k1/5ppp/8/8/8/8/5PP1/RR4K1 b - - 0 1"
        puzzle = {"id": "m", "fen": fen, "moves": ["g8h8", "b1b8"]}
        after = chess.Board(fen)
        after.push_uci("g8h8")
        key = " ".join(after.fen().split()[:4])
        alt = cont_eval.evaluate_puzzles(FakeScorer(self.utils, {key: "a1a8"}), [puzzle])
        self.assertTrue(alt[0]["correct"])          # a different, mating move
        non = cont_eval.evaluate_puzzles(FakeScorer(self.utils, {key: "a1a2"}), [puzzle])
        self.assertFalse(non[0]["correct"])         # a non-mating deviation fails
        self.assertEqual(non[0]["fail_ply"], 1)

    def test_summary_counts_and_rating_bins(self):
        puzzles = [{"rating": 500}, {"rating": 900}, {"rating": 950}]
        results = [{"correct": True, "fail_ply": None}, {"correct": False, "fail_ply": 1},
                   {"correct": False, "fail_ply": 3}]
        s = cont_eval.summarize(results, puzzles)
        self.assertEqual((s["solved"], s["total"]), (1, 3))
        self.assertEqual((s["failed_at_first_decision"], s["failed_later"]), (1, 1))
        self.assertEqual(s["by_rating"]["400-799"], {"solved": 1, "total": 1})


class LossTests(unittest.TestCase):
    def test_listwise_loss_matches_per_group_loop_and_masks_padding(self):
        import torch
        from scripts.train_cont import listwise_loss
        torch.manual_seed(0)
        lens = torch.tensor([3, 5, 1, 4])
        gid = torch.repeat_interleave(torch.arange(4), lens)
        q = torch.rand(int(lens.sum()), requires_grad=True)
        wp = torch.rand(int(lens.sum()))
        tau = 0.05
        loss, kl, hit, regret = listwise_loss(torch, q, wp, gid, lens, tau)
        expected, o = [], 0
        for n in lens.tolist():
            qs, ws = q[o:o + n], wp[o:o + n]
            expected.append(-(torch.softmax(ws / tau, 0) * torch.log_softmax(qs / tau, 0)).sum())
            o += n
        torch.testing.assert_close(loss, torch.stack(expected).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreaterEqual(float(kl), -1e-6)
        _, _, h1, r1 = listwise_loss(torch, q[8:9], wp[8:9], torch.zeros(1, dtype=torch.long),
                                     torch.tensor([1]), tau)
        self.assertEqual((float(h1), float(r1)), (1.0, 0.0))   # a single-move root

    def test_hit_counts_ties_for_the_best_value_and_regret_measures_the_loss(self):
        import torch
        from scripts.train_cont import listwise_loss
        lens = torch.tensor([3])
        gid = torch.zeros(3, dtype=torch.long)
        wp = torch.tensor([1.0, 1.0, 0.2])                 # two mating moves tie for best
        _, _, hit, regret = listwise_loss(torch, torch.tensor([0.1, 0.9, 0.5]), wp, gid, lens, 0.05)
        self.assertEqual(float(hit), 1.0)                  # picking either tied move is a hit
        _, _, hit, regret = listwise_loss(torch, torch.tensor([0.1, 0.2, 0.9]), wp, gid, lens, 0.05)
        self.assertEqual(float(hit), 0.0)
        self.assertAlmostEqual(float(regret), 0.8, places=4)

    def test_lr_schedule_warms_up_holds_and_decays(self):
        from scripts.train_cont import lr_at
        a = mock.Mock(lr=2e-5, warmup=100, decay_start=-1, decay_steps=0)
        self.assertAlmostEqual(lr_at(0, a), 2e-5 / 100)
        self.assertAlmostEqual(lr_at(500, a), 2e-5)
        b = mock.Mock(lr=2e-5, warmup=100, decay_start=1000, decay_steps=200)
        self.assertAlmostEqual(lr_at(1100, b), 1e-5)
        self.assertAlmostEqual(lr_at(2000, b), 0.0)


class TrainerIntegrationTests(unittest.TestCase):
    """Run the real training loop on CPU with a 2-layer model and synthetic data."""

    @classmethod
    def setUpClass(cls):
        cls.utils = load_utils()

    def build_fixture(self, root: Path):
        import torch
        from scripts.train_ccgavn import CCGAVN, candidate_relation_types
        rng = np.random.default_rng(5)
        cont = root / "data" / "cont-v1"
        (cont / "pack").mkdir(parents=True)
        src, dst, promo, _ = action_tables_from(self.utils)
        tokens, actions, winprob = synthetic_shard(rng, roots=30)
        starts = cont_data.group_starts(tokens, actions)
        is_dev = np.zeros(30, dtype=bool)
        is_dev[:3] = True
        cont_data.save_pack(cont / "pack" / "part-00.npz", {
            "tokens": tokens[starts], "lens": np.diff(np.append(starts, len(tokens))).astype(np.int32),
            "actions": actions.astype(np.uint16), "winprob": winprob.astype(np.float16),
            "rating": np.full(30, 1500, dtype=np.uint16), "index": np.arange(30, dtype=np.uint32),
            "is_dev": is_dev})
        bt, ba, bw = synthetic_shard(rng, roots=8)
        reps = int(np.ceil(64 / len(bt)))
        np.savez(cont / "broad-dev.npz", tokens=np.tile(bt, (reps, 1))[:64], actions=np.tile(ba, reps)[:64],
                 winprob=np.tile(bw, reps)[:64], teacher=random_teacher(rng, 64))
        rows = []
        for i in range(6):
            fen, moves = random_line(rng, 4)
            rows.append({"PuzzleId": f"v{i}", "FEN": fen, "Moves": " ".join(moves), "Rating": "1500"})
        cont_data.write_split_csv(cont / "val.csv", rows)
        cont_data.write_split_csv(cont / "test.csv", rows[:2])
        (cont / "manifest.json").write_text(json.dumps({
            "pack_parts": [{"file": "pack/part-00.npz"}], "manifest_sha256": "m" * 64,
            "val_csv_sha256": "v" * 64}))
        st, sa, sw = synthetic_shard(rng, roots=24)
        shard = (st, sa, sw, random_teacher(rng, len(st)).astype(np.float16))
        tags_file = root / "tags.json"
        tags_file.write_text(json.dumps({"tags": ["00000", "00001"]}))
        model = CCGAVN(torch, 32, 2, 4, src, dst, promo, candidate_relation_types(), model_version="v2")
        init = root / "init" / "base" / "checkpoint-1"
        init.mkdir(parents=True)
        torch.save({"model": model.state_dict()}, init / "state.pt")
        (init / "config.json").write_text(json.dumps({
            "architecture": "cc-gavn-v1", "model_version": "v2",
            "parameter_count": sum(p.numel() for p in model.parameters())}))
        return cont.parent, tags_file, shard

    def run_main(self, root, name, extra=(), resume_dir=None):
        from scripts import train_cont
        if not hasattr(self, "_fx"):
            self._fx = self.build_fixture(root)
        cont_dir, tags_file, shard = self._fx
        utils, statuses = self.utils, []

        class FakeManager:
            def __init__(self, repo, prefix, cache, token=None, expect_tags=0, tags=None, row_counts=None):
                self.tags, self.total = list(tags), 0
            def count_rows(self, *a, **k):
                self.total = 2 * len(shard[0])
            def load(self, tag, max_records=0):
                return shard
            def _download(self, tag):
                pass

        def fake_snapshot(repo, repo_type=None, token=None, local_dir=None, allow_patterns=None, **kw):
            shutil.copytree(root / "init" / "base", Path(local_dir) / "base", dirs_exist_ok=True)

        out = root / f"out-{name}"
        argv = ["--hf-run", name, "--init-hf-run", "base", "--init-checkpoint", "checkpoint-1",
                "--outdir", str(out), "--sl-repo", "x", "--cont-dir", str(cont_dir),
                "--broad-tags-file", str(tags_file), "--exclusion-puzzles", "unused",
                "--dim", "32", "--layers", "2", "--heads", "4", "--broad-rows", "16",
                "--tactical-groups", "3", "--block-steps", "5", "--warmup", "2", "--lr", "1e-3",
                "--val-every", "6", "--ckpt-every", "6", "--milestone-every", "0",
                "--hf-upload-every", "100000", "--dev-batch", "32", "--no-official-eval",
                "--guard-after", "99999", *extra]
        with contextlib.ExitStack() as stack:
            for p in (
                mock.patch.object(train_cont, "action_tables", lambda sl: action_tables_from(utils)),
                mock.patch.object(train_cont, "make_hf_api", lambda r=None: mock.Mock(token="t")),
                mock.patch.object(train_cont, "download_latest", lambda *a, **k: resume_dir),
                mock.patch.object(train_cont, "upload_checkpoint", lambda *a, **k: None),
                mock.patch.object(train_cont, "write_status", lambda c, r, run, s: statuses.append(s)),
                mock.patch.object(train_cont, "ShardManager", FakeManager),
                mock.patch.object(train_cont, "load_exclusions",
                                  lambda m, p: (np.zeros(0, dtype=np.uint64), "excl")),
                mock.patch("huggingface_hub.snapshot_download", fake_snapshot),
            ):
                stack.enter_context(p)
            train_cont.main(argv)
        return out, statuses

    def test_training_runs_checkpoints_and_resumes_exactly(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full, statuses = self.run_main(root, "full", ["--max-steps", "12"])
            for step in (6, 12):
                self.assertTrue((full / f"checkpoint-{step}" / "state.pt").exists())
            self.assertTrue(any(s.startswith("DONE") for s in statuses))
            cfg = json.loads((full / "checkpoint-12" / "config.json").read_text())
            self.assertEqual(cfg["recipe_version"], "cont1")
            self.assertIn("val_pct", cfg["baseline"])
            metrics = json.loads((full / "checkpoint-12" / "metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["train_loss"]))
            self.assertIn("val_pct", metrics)

            half, _ = self.run_main(root, "half", ["--max-steps", "6"])
            resumed, _ = self.run_main(root, "resumed", ["--max-steps", "12", "--resume-from-hf"],
                                       resume_dir=half / "checkpoint-6")
            a = torch.load(full / "checkpoint-12" / "state.pt", weights_only=False)
            b = torch.load(resumed / "checkpoint-12" / "state.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
                torch.testing.assert_close(a["ema"][key], b["ema"][key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(a["model"][k], a["ema"][k]) for k in a["model"]
                                if a["model"][k].dtype.is_floating_point))

    def test_resume_refuses_a_changed_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            half, _ = self.run_main(root, "half", ["--max-steps", "6"])
            with self.assertRaises(ValueError) as ctx:
                self.run_main(root, "resumed", ["--max-steps", "12", "--resume-from-hf", "--lr", "5e-4"],
                              resume_dir=half / "checkpoint-6")
            self.assertIn("recipe", str(ctx.exception))

    def test_session_deadline_stops_cleanly_with_a_resumable_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, statuses = self.run_main(root, "short", ["--session-seconds", "0.001",
                                                          "--kernel-started-epoch", "1"])
            self.assertTrue(any(s.startswith("IN PROGRESS checkpoint-") for s in statuses))
            self.assertTrue(list(out.glob("checkpoint-*/state.pt")))


if __name__ == "__main__":
    unittest.main()
