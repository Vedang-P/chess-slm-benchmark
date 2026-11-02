"""Tests for the cont1 data layer, evaluator, loss and trainer (CPU, tiny model)."""
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import shutil
import signal
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

import chess
import chess.engine
import numpy as np

from scripts import (compare_ply_runs, cont_config, cont_data, cont_eval, label_ply_shard,
                     ply_data, watch_ply_labels)
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

    def test_exact_policy_loss_matches_a_naive_loop_and_anchors_to_the_reference(self):
        import torch
        from scripts.train_cont import exact_policy_loss
        torch.manual_seed(1)
        lens = torch.tensor([3, 4, 2])
        gid = torch.repeat_interleave(torch.arange(3), lens)
        q = torch.rand(int(lens.sum()), requires_grad=True)
        q_ref = torch.rand(int(lens.sum()))
        wp = torch.rand(int(lens.sum()))
        tau, beta = 0.05, 0.3
        loss, value, kl = exact_policy_loss(torch, q, wp, gid, lens, tau, q_ref, beta)
        js, kls, o = [], [], 0
        for n in lens.tolist():
            pi = torch.softmax(q[o:o + n] / tau, 0)
            ref = torch.softmax(q_ref[o:o + n] / tau, 0)
            js.append((pi * wp[o:o + n]).sum())
            kls.append((pi * (pi.log() - ref.log())).sum())
            o += n
        torch.testing.assert_close(loss, -(torch.stack(js) - beta * torch.stack(kls)).mean())
        torch.testing.assert_close(value, torch.stack(js).mean())
        torch.testing.assert_close(kl, torch.stack(kls).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        # with the reference equal to the policy the KL is zero and the loss is -J
        same, _, k0 = exact_policy_loss(torch, q.detach(), wp, gid, lens, tau, q.detach(), beta)
        self.assertAlmostEqual(float(k0), 0.0, places=6)
        # moving value toward the best move lowers the loss
        qq = torch.tensor([0.4, 0.5, 0.6], requires_grad=True)
        w3 = torch.tensor([0.2, 0.3, 0.9])
        exact_policy_loss(torch, qq, w3, torch.zeros(3, dtype=torch.long), torch.tensor([3]), tau)[0].backward()
        self.assertLess(float(qq.grad[2]), 0.0)     # d(loss)/dq of the best move is negative

    def test_exact_policy_loss_accepts_half_precision_reference_values(self):
        """On GPU the reference expectation comes out of an fp16 autocast region."""
        import torch
        from scripts.train_cont import exact_policy_loss
        lens = torch.tensor([3, 2])
        gid = torch.repeat_interleave(torch.arange(2), lens)
        q = torch.rand(5, requires_grad=True)
        loss, _, kl = exact_policy_loss(torch, q, torch.rand(5), gid, lens, 0.05,
                                        torch.rand(5).half(), 0.05)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(q.grad).all())

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
        ply_tokens, ply_actions, ply_winprob = synthetic_shard(rng, roots=24)
        ply_starts = cont_data.group_starts(ply_tokens, ply_actions)
        ply_dev = np.zeros(24, dtype=bool)
        ply_dev[:4] = True
        ply_dir = cont.parent / "cont-v2" / "ply-pack"
        ply_dir.mkdir(parents=True)
        cont_data.save_pack(ply_dir / "part-00.npz", {
            "tokens": ply_tokens[ply_starts],
            "lens": np.diff(np.append(ply_starts, len(ply_tokens))).astype(np.int32),
            "actions": ply_actions.astype(np.uint16), "winprob": ply_winprob.astype(np.float16),
            "rating": np.full(24, 1500, dtype=np.uint16), "index": np.arange(24, dtype=np.uint32),
            "is_dev": ply_dev, "ply": np.where(np.arange(24) % 3 == 0, 5, 3).astype(np.uint8)})
        (ply_dir / "manifest.json").write_text(json.dumps({
            "pack_parts": [{"file": "part-00.npz"}], "manifest_sha256": "p" * 64}))
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
                "--ply-dir", str(cont_dir),
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

    def test_stage2_rl_term_runs_and_records_its_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, statuses = self.run_main(Path(tmp), "rl", ["--max-steps", "6", "--rl-weight", "1.0",
                                                            "--rl-beta", "0.05"])
            self.assertTrue((out / "checkpoint-6" / "state.pt").exists())
            cfg = json.loads((out / "checkpoint-6" / "config.json").read_text())
            self.assertEqual(cfg["rl_weight"], 1.0)
            self.assertTrue(any(s.startswith("DONE") for s in statuses))

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


    def test_ply_groups_extend_the_batch_resume_exactly_and_record_their_identity(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra = ["--ply-groups", "2"]
            full, _ = self.run_main(root, "plyfull", ["--max-steps", "12", *extra])
            cfg = json.loads((full / "checkpoint-12" / "config.json").read_text())
            self.assertEqual(cfg["ply_groups"], 2)
            self.assertEqual(cfg["ply_pack_digest"], "p" * 64)
            metrics = json.loads((full / "checkpoint-12" / "metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["ply_dev_hit"]))
            half, _ = self.run_main(root, "plyhalf", ["--max-steps", "6", *extra])
            resumed, _ = self.run_main(root, "plyresumed", ["--max-steps", "12", "--resume-from-hf", *extra],
                                       resume_dir=half / "checkpoint-6")
            a = torch.load(full / "checkpoint-12" / "state.pt", weights_only=False)
            b = torch.load(resumed / "checkpoint-12" / "state.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            # the extra groups really change training, and a plain run keeps no ply identity
            plain, _ = self.run_main(root, "plain", ["--max-steps", "12"])
            c = torch.load(plain / "checkpoint-12" / "state.pt", weights_only=False)
            self.assertTrue(any(not torch.equal(a["model"][k], c["model"][k]) for k in a["model"]
                                if a["model"][k].dtype.is_floating_point))
            plain_cfg = json.loads((plain / "checkpoint-12" / "config.json").read_text())
            self.assertEqual(plain_cfg["ply_groups"], 0)
            self.assertNotIn("ply_pack_digest", plain_cfg)

    def test_resume_refuses_a_run_that_adds_the_ply_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            half, _ = self.run_main(root, "half", ["--max-steps", "6"])
            with self.assertRaises(ValueError) as ctx:
                self.run_main(root, "resumed", ["--max-steps", "12", "--resume-from-hf",
                                                "--ply-groups", "2"], resume_dir=half / "checkpoint-6")
            self.assertIn("ply_pack_digest", str(ctx.exception))

    def test_ply_groups_work_without_first_decision_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, statuses = self.run_main(Path(tmp), "plyonly", ["--max-steps", "6", "--tactical-groups", "0",
                                                                "--ply-groups", "3"])
            self.assertTrue((out / "checkpoint-6" / "state.pt").exists())
            self.assertTrue(any(s.startswith("DONE") for s in statuses))

    def test_val_detail_saves_per_puzzle_results_and_the_failure_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self.run_main(Path(tmp), "detail", ["--max-steps", "6", "--val-detail"])
            metrics = json.loads((out / "checkpoint-6" / "metrics.json").read_text())
            results = json.loads((out / "checkpoint-6" / "val-results.json").read_text())
            self.assertEqual(len(results["ids"]), 6)
            self.assertEqual(len(results["correct"]), 6)
            by_ply = metrics["val_failed_by_ply"]
            self.assertEqual(sum(by_ply.values()),
                             metrics["val_failed_first_decision"] + metrics["val_failed_later"])
            self.assertEqual(sum(1 for c, f in zip(results["correct"], results["fail_ply"])
                                 if not c and f is not None), sum(by_ply.values()))
            plain, _ = self.run_main(Path(tmp), "nodetail", ["--max-steps", "6"])
            self.assertFalse((plain / "checkpoint-6" / "val-results.json").exists())
            self.assertNotIn("val_failed_by_ply", json.loads(
                (plain / "checkpoint-6" / "metrics.json").read_text()))



class MonitorTests(unittest.TestCase):
    def test_cont1_page_renders_a_realistic_snapshot_without_errors(self):
        import shutil
        import subprocess
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        root = Path(__file__).resolve().parents[1]
        snapshot = {
            "run": "cont1", "updated_at": "2026-11-02T10:00:00+00:00", "phase": "noop",
            "main_run": "ccgavn-cont1-main", "base_step": 3896658,
            "reference": {"base_puzzles": 7620, "gate_pct": 86.13},
            "supervisor": {"reason": "learner active on ['vedanggggg']"},
            "provider_warnings": [], "links": {"checkpoints": "https://example.org"},
            "accounts": [{"account": "acct-1", "gpu_hours": 28.5, "learner": "RUNNING"},
                         {"account": "acct-2", "gpu_hours": None, "learner": "unknown"}],
            "runs": {
                "ccgavn-cont1-a2": {"step": 10000, "status": "IN PROGRESS", "baseline": {
                    "val_pct": 75.9, "val_total": 4000},
                    "recipe": {"lr": 2e-5, "tactical_groups": 20, "lambda_list": 1.0},
                    "evaluations": [{"step": 30000, "mate": {"correct": 3700},
                                     "puzzles": {"solved": 7700}}]},
                "ccgavn-cont1-main": {"step": 0, "status": "", "baseline": None, "recipe": {},
                                      "evaluations": []}},
            "history": {"metrics": {"ccgavn-cont1-a2": [
                {"step": 5000, "val_pct": 75.5, "tac_hit": 0.61, "dev_dist": 2.91},
                {"step": 10000, "val_pct": 76.4, "tac_hit": 0.64, "dev_dist": 2.90}]}},
            "events": [{"title": "t", "detail": "d", "tone": "good", "at": "2026-11-02T09:00:00Z"}]}
        script = (
            "import { HTML } from './src/page.js';"
            "const m=[...HTML.matchAll(/<script(?:\\s[^>]*)?>([\\s\\S]*?)<\\/script>/gi)][0][1];"
            "const els={};const el=id=>els[id]??={style:{},hidden:false,textContent:'',innerHTML:'',className:''};"
            "globalThis.document={getElementById:el};"
            f"globalThis.fetch=async()=>({{ok:true,json:async()=>({json.dumps(snapshot)})}});"
            "globalThis.setInterval=()=>0;"
            "new Function(m)();await new Promise(r=>setTimeout(r,50));"
            "if(els.badge.textContent!=='noop')throw new Error('badge: '+els.badge.textContent+' / '+els.fresh.textContent);"
            "if(!els.runs.innerHTML.includes('a2'))throw new Error('run table not rendered');"
            "if(!els['chart-val'].innerHTML.includes('polyline'))throw new Error('validation chart empty');"
            "if(els['best-official'].textContent!=='77.00%')throw new Error('best official '+els['best-official'].textContent);"
            "console.log('rendered');")
        result = subprocess.run([node, "--input-type=module", "-e", script], cwd=root / "monitor",
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")


# ---------------------------------------------------------------- later-ply roots
class StubTokenizer:
    @staticmethod
    def tokenize(fen):
        return tokenize_fen(fen)


def uci_action_table() -> dict[str, int]:
    names = [chess.square_name(i) for i in range(64)]
    ucis = ([a + b for a in names for b in names if a != b]
            + [a + b + p for a in names for b in names if a != b for p in "qrbn"])
    return {m: i for i, m in enumerate(ucis)}


class StubUtils:
    """Action table and value transform of the pinned searchless utils, without the checkout."""
    MOVE_TO_ACTION = uci_action_table()

    @staticmethod
    def centipawns_to_win_probability(centipawns):
        return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)


class FakeEngine:
    """Deterministic pseudo-scores per (position, root move); the Stockfish interface used here."""

    def __init__(self, hook=None):
        self.id = {"name": "Stockfish 16"}
        self.calls, self.hook = 0, hook

    def analyse(self, board, limit, root_moves=None, info=None):
        self.calls += 1
        if self.hook is not None:
            self.hook()
        time.sleep(0.0003)
        key = f"{board.fen()}|{root_moves[0].uci()}"
        value = zlib.crc32(key.encode()) % 2400 - 1200
        score = chess.engine.Mate(2) if value % 211 == 0 else chess.engine.Cp(int(value))
        return {"score": chess.engine.PovScore(score, chess.WHITE)}

    def close(self):
        pass


def random_puzzle(rng, puzzle_id, n_moves):
    """A legal line of `n_moves` from a random middlegame position (FEN precedes Moves[0])."""
    while True:
        board = chess.Board()
        for _ in range(int(rng.integers(4, 14))):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[int(rng.integers(len(legal)))])
        if board.is_game_over():
            continue
        fen, moves = board.fen(), []
        for _ in range(n_moves):
            legal = sorted(board.legal_moves, key=lambda m: m.uci())
            if not legal:
                break
            move = legal[int(rng.integers(len(legal)))]
            moves.append(move.uci())
            board.push(move)
        if len(moves) == n_moves:
            return {"PuzzleId": puzzle_id, "FEN": fen, "Moves": " ".join(moves),
                    "Rating": str(1000 + int(rng.integers(0, 1500)))}


class PlyFixture:
    """A tiny frozen corpus: 30 puzzles (2, 4, 6, 8 moves in turn), two official, one validation,
    one test. With four roots per shard the first ply-3 shard is P001 P002 P005 P006 and the
    second ply-5 shard is P015 P018 P019 P022."""

    OFFICIAL, VAL, TEST = ("P003", "P010"), ("P007",), ("P012",)

    def __init__(self, root: Path, bench_ply: int = 3, bench_puzzle: str = "P005"):
        import zstandard as zstd
        rng = np.random.default_rng(21)
        self.rows = [random_puzzle(rng, f"P{i:03d}", (2, 4, 6, 8)[i % 4]) for i in range(30)]
        header = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays",
                  "Themes", "GameUrl", "OpeningTags"]
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=header)
        writer.writeheader()
        for row in self.rows:
            writer.writerow({**{k: "" for k in header}, **row})
        self.source = root / "source.csv.zst"
        self.source.write_bytes(zstd.ZstdCompressor().compress(text.getvalue().encode()))
        self.official = set(self.OFFICIAL)
        self.manifest = root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "format": "lichess-puzzle-corpus-v1", "source_sha256": label_ply_shard.file_sha256(self.source),
            "excluded_ids": sorted(self.official), "included_roots": 28}))
        cont_data.write_split_csv(root / "val.csv", [r for r in self.rows if r["PuzzleId"] in self.VAL])
        cont_data.write_split_csv(root / "test.csv", [r for r in self.rows if r["PuzzleId"] in self.TEST])
        self.val_csv, self.test_csv = root / "val.csv", root / "test.csv"
        # one benchmark position: the ply-`bench_ply` root of `bench_puzzle`
        from scripts.data_hygiene import position_hashes
        row = next(r for r in self.rows if r["PuzzleId"] == bench_puzzle)
        root_tokens = tokenize_fen(label_ply_shard.root_board(row, bench_ply).fen())
        self.bench_puzzle, self.bench_ply = bench_puzzle, bench_ply
        self.hash_path = root / "exclusion-hashes.npy"
        np.save(self.hash_path, np.unique(position_hashes(root_tokens[None, :])))
        self.root = root

    def build(self, ply, shard, name="run", engines=1, roots_per_shard=4, **kwargs):
        engines_made = kwargs.pop("engines_made", [])
        hook = kwargs.pop("hook", None)

        def factory(_path):
            engine = FakeEngine(hook)
            engines_made.append(engine)
            return engine

        folder = self.root / name
        with mock.patch.object(label_ply_shard, "OFFICIAL_COUNT", len(self.official)), \
                mock.patch.object(label_ply_shard, "INCLUDED_ROOTS", 28):
            return label_ply_shard.build(
                self.source, self.manifest, ply, shard, Path("stockfish"), folder / "train_set.npz",
                folder / "partial.npz", Path("sl"), val_csv=self.val_csv, test_csv=self.test_csv,
                exclusion_path=self.hash_path, engines=engines, engine_factory=factory,
                modules=(StubTokenizer, StubUtils), roots_per_shard=roots_per_shard, **kwargs)

    def output(self, name="run"):
        with np.load(self.root / name / "train_set.npz", allow_pickle=False) as data:
            return {k: data[k] for k in data.files}


class PlyLabelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._handlers = {n: signal.getsignal(n) for n in (signal.SIGINT, signal.SIGTERM)}
        label_ply_shard.STOP = False
        self.fx = PlyFixture(self.root)

    def tearDown(self):
        for number, handler in self._handlers.items():
            signal.signal(number, handler)
        label_ply_shard.STOP = False
        self._tmp.cleanup()

    def test_root_board_follows_the_line_and_guards_its_inputs(self):
        row = next(r for r in self.fx.rows if r["PuzzleId"] == "P011")       # 8 moves
        moves = row["Moves"].split()
        board = chess.Board(row["FEN"])
        for uci in moves[:3]:
            board.push_uci(uci)
        self.assertEqual(label_ply_shard.root_board(row, 3).fen(), board.fen())
        for uci in moves[3:5]:
            board.push_uci(uci)
        self.assertEqual(label_ply_shard.root_board(row, 5).fen(), board.fen())
        self.assertEqual(label_ply_shard.root_board(row, 7).turn, board.turn)   # Moves[7] exists
        with self.assertRaises(ValueError):
            label_ply_shard.root_board(row, 8)                                 # no stored move
        with self.assertRaises(ValueError):
            label_ply_shard.root_board(row, 2)                                 # not a solver decision
        broken = dict(row, Moves=" ".join(moves[:3] + ["a1a1"] + moves[4:]))
        with self.assertRaises(ValueError):
            label_ply_shard.root_board(broken, 5)
        bad_solution = dict(row, Moves=" ".join(moves[:3] + ["h1h8"]))
        with self.assertRaises(ValueError):
            label_ply_shard.root_board(bad_solution, 3)

    def test_ply_one_matches_the_frozen_first_decision_labeler(self):
        from scripts import label_puzzle_shard
        row = self.fx.rows[5]
        engine = FakeEngine()
        ours = label_ply_shard.label_root(row, 1, engine, StubTokenizer, StubUtils)
        frozen = label_puzzle_shard.label_root(row, engine, StubTokenizer, StubUtils)
        for a, b in zip(ours, frozen):
            np.testing.assert_array_equal(a, b)

    def test_enumeration_numbers_rows_like_the_split_table_and_applies_every_exclusion(self):
        skip = ply_data.read_ids(self.fx.val_csv, self.fx.test_csv)
        for ply in (3, 5):
            got = list(ply_data.eligible_rows(self.fx.source, self.fx.official, skip, ply))
            ids = [row["PuzzleId"] for _, row in got]
            want = [r["PuzzleId"] for r in self.fx.rows
                    if len(r["Moves"].split()) > ply
                    and r["PuzzleId"] not in self.fx.official | skip]
            self.assertEqual(ids, want)
            # the number counts validation/test rows, as the cont-v1 split table does
            included = [r["PuzzleId"] for r in self.fx.rows if r["PuzzleId"] not in self.fx.official]
            self.assertEqual([i for i, _ in got], [included.index(x) for x in ids])
        self.assertEqual(ply_data.count_eligible(self.fx.source, self.fx.official, skip, (3, 5)),
                         {3: len(list(ply_data.eligible_rows(self.fx.source, self.fx.official, skip, 3))),
                          5: len(list(ply_data.eligible_rows(self.fx.source, self.fx.official, skip, 5)))})
        blocks = [ply_data.shard_candidates(self.fx.source, self.fx.official, skip, 3, k, 4)
                  for k in range(4)]
        self.assertEqual([len(b) for b in blocks][:2], [4, 4])
        flat = [ordinal for b in blocks for ordinal, _, _ in b]
        self.assertEqual(flat, list(range(len(flat))))
        self.assertEqual(ply_data.shard_candidates(self.fx.source, self.fx.official, skip, 3, 40, 4), [])

    def test_job_order_interleaves_plies_and_partitions_exactly(self):
        order = ply_data.job_order({3: 10, 5: 4})
        self.assertEqual(sorted(order), sorted([(3, i) for i in range(10)] + [(5, i) for i in range(4)]))
        # every prefix holds both plies in about their proportion
        head = order[:7]
        self.assertTrue(sum(1 for p, _ in head if p == 5) in (2, 3))
        self.assertEqual([s for p, s in order if p == 3], list(range(10)))
        parts = [ply_data.worker_jobs(order, w, 5) for w in range(5)]
        self.assertEqual(sorted(j for part in parts for j in part), sorted(order))
        self.assertEqual(ply_data.worker_jobs(order, 0, 5)[0], order[0])
        with self.assertRaises(ValueError):
            ply_data.worker_jobs(order, 5, 5)

    def test_identity_names_the_ply_and_every_input(self):
        base = dict(source_sha256="s", ply=3, shard=2, max_roots=0, val_csv_sha256="v",
                    test_csv_sha256="t", exclusion_sha256="e")
        a = ply_data.shard_identity(**base)
        self.assertEqual(a["ply"], 3)
        for key, value in (("ply", 5), ("shard", 3), ("exclusion_sha256", "x"), ("max_roots", 7)):
            self.assertNotEqual(a, ply_data.shard_identity(**{**base, key: value}))
        with self.assertRaises(ValueError):
            ply_data.shard_identity(**{**base, "ply": 4})

    def test_labeler_writes_whole_roots_and_drops_only_benchmark_positions(self):
        skip = ply_data.read_ids(self.fx.val_csv, self.fx.test_csv)
        cands = ply_data.shard_candidates(self.fx.source, self.fx.official, skip, 3, 0, 4)
        self.assertIn(self.fx.bench_puzzle, [row["PuzzleId"] for _, _, row in cands])   # in this shard
        result = self.fx.build(3, 0)
        self.assertTrue(result["complete"])
        self.assertEqual((result["candidates"], result["dropped_benchmark"], result["roots"]), (4, 1, 3))
        data = self.fx.output()
        self.assertNotIn(self.fx.bench_puzzle, data["puzzle_id"].tolist())
        self.assertEqual(data["puzzle_id"].tolist(),
                         [row["PuzzleId"] for _, _, row in cands if row["PuzzleId"] != self.fx.bench_puzzle])
        self.assertEqual(int(data["lens"].sum()), len(data["actions"]))
        self.assertTrue((data["ply"] == 3).all())
        starts = np.cumsum(data["lens"]) - data["lens"]
        for k, (_, included, row) in enumerate(c for c in cands if c[2]["PuzzleId"] != self.fx.bench_puzzle):
            board = label_ply_shard.root_board(row, 3)
            legal = list(board.legal_moves)
            sl = slice(starts[k], starts[k] + data["lens"][k])
            self.assertEqual(data["lens"][k], len(legal))
            np.testing.assert_array_equal(data["tokens"][sl], np.tile(tokenize_fen(board.fen()), (len(legal), 1)))
            np.testing.assert_array_equal(data["actions"][sl], [StubUtils.MOVE_TO_ACTION[m.uci()] for m in legal])
            expected = []
            for m in legal:
                value = zlib.crc32(f"{board.fen()}|{m.uci()}".encode()) % 2400 - 1200
                score = 10_000 if value % 211 == 0 else value          # white POV; flip for black
                expected.append(StubUtils.centipawns_to_win_probability(
                    score if board.turn == chess.WHITE else -score))
            np.testing.assert_allclose(data["winprob"][sl], expected, atol=1e-6)
            self.assertEqual(int(data["index"][k]), included)
            self.assertEqual(int(data["rating"][k]), int(row["Rating"]))
        self.assertEqual(result["identity"]["ply"], 3)
        self.assertEqual(result["train_set_sha256"], label_ply_shard.file_sha256(self.root / "run" / "train_set.npz"))
        for official_or_split in (*self.fx.OFFICIAL, *self.fx.VAL, *self.fx.TEST):
            self.assertNotIn(official_or_split, data["puzzle_id"].tolist())

    def test_parallel_engines_match_one_engine_and_all_of_them_work(self):
        self.fx.build(5, 0, name="one")
        made = []
        self.fx.build(5, 0, name="four", engines=4, engines_made=made)
        self.assertEqual(len(made), 4)
        self.assertGreater(sum(e.calls > 0 for e in made), 1)
        one, four = self.fx.output("one"), self.fx.output("four")
        self.assertEqual(sorted(one), sorted(four))
        for key in one:
            np.testing.assert_array_equal(one[key], four[key])

    def test_a_stopped_shard_resumes_to_exactly_the_uninterrupted_result(self):
        self.fx.build(3, 1, name="whole", roots_per_shard=8)
        seen = []

        def stop_midway():
            seen.append(1)
            if len(seen) == 20:
                label_ply_shard.STOP = True

        part = self.fx.build(3, 1, name="split", engines=2, hook=stop_midway, roots_per_shard=8)
        self.assertFalse(part["complete"])
        self.assertTrue((self.root / "split" / "partial.npz").is_file())
        self.assertFalse((self.root / "split" / "train_set.npz").exists())
        label_ply_shard.STOP = False
        done = self.fx.build(3, 1, name="split", engines=3, roots_per_shard=8)
        self.assertTrue(done["complete"])
        whole, split = self.fx.output("whole"), self.fx.output("split")
        for key in whole:
            np.testing.assert_array_equal(whole[key], split[key])

    def test_a_partial_from_another_ply_or_shard_is_rejected(self):
        label_ply_shard.STOP = True
        self.fx.build(3, 0, name="mix")                    # stops at once: nothing saved
        label_ply_shard.STOP = False
        seen = []

        def stop_early():
            seen.append(1)
            if len(seen) == 5:
                label_ply_shard.STOP = True

        self.assertFalse(self.fx.build(3, 0, name="mix", hook=stop_early)["complete"])
        label_ply_shard.STOP = False
        with self.assertRaises(ValueError) as ctx:
            self.fx.build(5, 0, name="mix")
        self.assertIn("identity", str(ctx.exception))
        with self.assertRaises(ValueError):
            self.fx.build(3, 1, name="mix")

    def test_a_shard_past_the_corpus_end_and_a_wrong_engine_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.fx.build(3, 40)
        wrong = FakeEngine()
        wrong.id = {"name": "Stockfish 17"}
        with self.assertRaises(ValueError):
            label_ply_shard._valid_engine(wrong)


class PlyPackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._handlers = {n: signal.getsignal(n) for n in (signal.SIGINT, signal.SIGTERM)}
        label_ply_shard.STOP = False
        self.fx = PlyFixture(self.root, bench_ply=5, bench_puzzle="P018")   # in the second ply-5 shard

    def tearDown(self):
        for number, handler in self._handlers.items():
            signal.signal(number, handler)
        self._tmp.cleanup()

    def shard(self, ply, shard):
        result = self.fx.build(ply, shard, name=f"p{ply}s{shard}")
        return self.fx.output(f"p{ply}s{shard}"), result

    def builder(self):
        return ply_data.PlyPackBuilder(np.load(self.fx.hash_path),
                                       set(self.fx.OFFICIAL) | set(self.fx.VAL) | set(self.fx.TEST))

    def test_pack_round_trips_roots_adds_ply_and_marks_the_dev_fold(self):
        from scripts.data_hygiene import position_hashes
        b = self.builder()
        shards = [(3, 0), (3, 1), (5, 0)]
        outputs = {}
        for ply, shard in shards:
            data, result = self.shard(ply, shard)
            outputs[(ply, shard)] = data
            b.add_shard(ply, shard, data, result)
        arrays = b.finish()
        pack = ply_data.PlyPack(arrays)
        total = sum(len(d["lens"]) for d in outputs.values())
        self.assertEqual(len(pack), total)
        self.assertEqual(b.stats["roots"], total)
        self.assertEqual(sorted(set(pack.ply.tolist())), [3, 5])
        self.assertEqual(int((pack.ply == 5).sum()), len(outputs[(5, 0)]["lens"]))
        # gather returns the stored rows of each root, in order
        k = 0
        for key in shards:
            data = outputs[key]
            starts = np.cumsum(data["lens"]) - data["lens"]
            for r in range(len(data["lens"])):
                t, a, w, gid, lens = pack.gather(np.array([k]))
                sl = slice(starts[r], starts[r] + data["lens"][r])
                np.testing.assert_array_equal(t, data["tokens"][sl])
                np.testing.assert_array_equal(a, data["actions"][sl])
                np.testing.assert_allclose(w, data["winprob"][sl], atol=1e-3)
                self.assertEqual(int(pack.index[k]), int(data["index"][r]))
                k += 1
        hashes = position_hashes(pack.tokens)
        np.testing.assert_array_equal(pack.is_dev, hashes % np.uint64(100) == 0)
        self.assertEqual(len(pack.train_roots) + len(pack.dev_roots), len(pack))

    def test_pack_saves_and_loads_through_the_part_files(self):
        b = self.builder()
        for ply, shard in ((3, 0), (5, 0)):
            data, result = self.shard(ply, shard)
            b.add_shard(ply, shard, data, result)
        arrays = b.finish()
        offsets = np.concatenate([[0], np.cumsum(arrays["lens"].astype(np.int64))])
        cut = len(arrays["lens"]) // 2
        paths = []
        for i, (a, c) in enumerate(((0, cut), (cut, len(arrays["lens"])))):
            piece = {k: arrays[k][a:c] for k in ("tokens", "lens", "rating", "index", "is_dev", "ply")}
            piece["actions"], piece["winprob"] = (arrays["actions"][offsets[a]:offsets[c]],
                                                  arrays["winprob"][offsets[a]:offsets[c]])
            cont_data.save_pack(self.root / f"part-{i}.npz", piece)
            paths.append(self.root / f"part-{i}.npz")
        loaded, direct = ply_data.PlyPack.load(paths), ply_data.PlyPack(arrays)
        for name in ("tokens", "lens", "actions", "winprob", "ply", "index", "is_dev"):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(direct, name))
        with self.assertRaises(ValueError):
            ply_data.PlyPack({k: v for k, v in arrays.items() if k != "ply"})

    def test_pack_refuses_shards_that_break_the_hygiene_rules(self):
        data, result = self.shard(3, 1)
        ids = data["puzzle_id"].tolist()
        for bad in (self.fx.OFFICIAL[0], self.fx.VAL[0], self.fx.TEST[0]):
            tampered = dict(data, puzzle_id=np.asarray([bad] + ids[1:], dtype="U8"))
            with self.assertRaises(ValueError) as ctx:
                self.builder().add_shard(3, 1, tampered, result)
            self.assertIn("official/validation/test", str(ctx.exception))
        # a benchmark position that survived labeling: re-label the shard that holds one with
        # the exclusion set emptied, then pack it against the real set
        empty = self.root / "none.npy"
        np.save(empty, np.zeros(0, dtype=np.uint64))
        keep, self.fx.hash_path = self.fx.hash_path, empty
        leaked, leaked_result = self.shard(5, 1)
        self.fx.hash_path = keep
        self.assertIn(self.fx.bench_puzzle, leaked["puzzle_id"].tolist())
        with self.assertRaises(ValueError) as ctx:
            self.builder().add_shard(5, 1, leaked, leaked_result)
        self.assertIn("benchmark", str(ctx.exception))
        with self.assertRaises(ValueError):
            self.builder().add_shard(5, 1, dict(data, ply=np.full(len(data["lens"]), 3, dtype=np.uint8)),
                                     result)                        # wrong ply for the shard
        with self.assertRaises(ValueError):
            self.builder().add_shard(3, 1, dict(data, actions=data["actions"][:-1]), result)

    def test_pack_script_reads_finished_shards_and_writes_a_verifiable_pack(self):
        import sys as _sys
        from scripts import build_ply_pack
        sha = label_ply_shard.file_sha256
        plan = {"source_sha256": sha(self.fx.source), "val_csv_sha256": sha(self.fx.val_csv),
                "test_csv_sha256": sha(self.fx.test_csv), "exclusion_sha256": sha(self.fx.hash_path),
                "roots_per_shard": 4}
        hub = {}
        for ply, shard in ((3, 0), (3, 1), (5, 0)):
            result = self.fx.build(ply, shard, name=f"p{ply}s{shard}")
            folder, remote = self.root / f"p{ply}s{shard}", ply_data.shard_dir(ply_data.PREFIX, ply, shard)
            (folder / "meta.json").write_text(json.dumps(result))
            hub[f"{remote}/train_set.npz"], hub[f"{remote}/meta.json"] = folder / "train_set.npz", folder / "meta.json"
        (self.root / "plan.json").write_text(json.dumps(plan))
        hub.update({f"{ply_data.PREFIX}/plan.json": self.root / "plan.json",
                    f"{ply_data.PREFIX}/exclusion-hashes.npy": self.fx.hash_path,
                    "puzzle-corpus-v1/manifest.json": self.fx.manifest,
                    "cont-v1/val.csv": self.fx.val_csv, "cont-v1/test.csv": self.fx.test_csv})

        def fake_get(api, repo, path, workdir, attempts=5):
            target = Path(workdir) / path          # a copy: the script deletes shards it has used
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hub[path], target)
            return target

        client = mock.Mock(token="t")
        client.list_repo_files.return_value = list(hub) + [
            f"{ply_data.PREFIX}/ply5/shard-00001/partial.npz"]           # a partial must not be packed
        work = self.root / "work"
        argv = ["build_ply_pack.py", "--workdir", str(work)]
        with mock.patch.object(build_ply_pack, "hf_get", fake_get), \
                mock.patch("scripts.kaggle_checkpoint.api", lambda root=None: client), \
                mock.patch.object(_sys, "argv", argv):
            build_ply_pack.main()
        out = work / ply_data.PACK_PREFIX
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["format"], ply_data.PACK_FORMAT)
        self.assertEqual(manifest["shards"], {"3": [0, 1], "5": [0]})
        self.assertEqual(manifest["manifest_sha256"], ply_data.pack_manifest_digest(manifest))
        for part in manifest["pack_parts"]:
            self.assertEqual(sha(out / part["file"]), part["sha256"])
        pack = ply_data.PlyPack.load([out / part["file"] for part in manifest["pack_parts"]])
        self.assertEqual(len(pack), manifest["pack_stats"]["roots"])
        self.assertEqual(sorted(set(pack.ply.tolist())), [3, 5])
        self.assertEqual(sum(part["roots"] for part in manifest["pack_parts"]), len(pack))
        # a shard that is not the one the plan describes is refused, not repacked
        meta = json.loads((self.root / "p3s0" / "meta.json").read_text())
        meta["identity"]["max_roots"] = 40
        (self.root / "p3s0" / "meta.json").write_text(json.dumps(meta))
        with mock.patch.object(build_ply_pack, "hf_get", fake_get), \
                mock.patch("scripts.kaggle_checkpoint.api", lambda root=None: client), \
                mock.patch.object(_sys, "argv", argv):
            with self.assertRaises(ValueError) as ctx:
                build_ply_pack.main()
        self.assertIn("identity gate", str(ctx.exception))


class PlyFleetTests(unittest.TestCase):
    HEAD, OTHER = "a" * 40, "b" * 40
    NOW = 1_000_000.0

    def decide(self, **kw):
        args = {"remaining": 5, "finished": 2, "kaggle_status": 'x has status "KernelWorkerStatus.COMPLETE"',
                "worker_status": "OK labels", "commit": self.HEAD, "marker": None, "now": self.NOW}
        args.update(kw)
        return watch_ply_labels.decide_worker(**args)

    def test_a_worker_is_relaunched_only_while_it_has_unfinished_shards(self):
        self.assertEqual(self.decide()["action"], "launch")
        self.assertEqual(self.decide(remaining=0)["action"], "done")
        self.assertEqual(self.decide(kaggle_status='x has status "KernelWorkerStatus.RUNNING"')["action"], "active")
        self.assertEqual(self.decide(kaggle_status='x has status "KernelWorkerStatus.QUEUED"')["action"], "active")
        self.assertEqual(self.decide(kaggle_status=None)["action"], "wait")
        recent = {"commit": self.HEAD, "finished": 2, "attempt": 1, "at_epoch": self.NOW - 60}
        self.assertEqual(self.decide(marker=recent)["action"], "wait")

    def test_a_failure_blocks_only_its_own_commit(self):
        failed = f"# 2026-11-02T00:00:00Z\nFAILED labels source_commit={self.HEAD} ValueError: boom"
        self.assertEqual(self.decide(worker_status=failed)["action"], "blocked")
        self.assertEqual(self.decide(commit=self.OTHER, worker_status=failed)["action"], "launch")
        self.assertEqual(self.decide(worker_status="FAILED labels ValueError: unstamped")["action"], "blocked")

    def test_launches_that_finish_nothing_stop_after_three_attempts(self):
        marker = {"commit": self.HEAD, "finished": 2, "attempt": 1, "at_epoch": self.NOW - 7200}
        d = self.decide(marker=marker)
        self.assertEqual((d["action"], d["attempt"]), ("launch", 2))
        marker["attempt"] = 3
        self.assertEqual(self.decide(marker=marker)["action"], "blocked")
        self.assertEqual(self.decide(marker=marker, finished=3)["action"], "launch")     # progress resets
        self.assertEqual(self.decide(marker=marker, commit=self.OTHER)["action"], "launch")

    def test_unfinished_jobs_follow_the_hub_listing(self):
        order = ply_data.job_order({3: 6, 5: 3})
        files = {f"{ply_data.PREFIX}/ply3/shard-00000/train_set.npz", f"{ply_data.PREFIX}/ply3/shard-00000/meta.json",
                 f"{ply_data.PREFIX}/ply3/shard-00001/train_set.npz",            # labels without meta: not done
                 f"{ply_data.PREFIX}/ply5/shard-00002/partial.npz",               # a partial is not done
                 f"{ply_data.PREFIX}/ply5/shard-00001/meta.json",
                 f"{ply_data.PREFIX}/ply5/shard-00001/train_set.npz"}
        complete = ply_data.complete_shards(files, ply_data.PREFIX)
        self.assertEqual(complete, {3: {0}, 5: {1}})
        pending = watch_ply_labels.unfinished_jobs(order, complete, workers=4)
        self.assertEqual(sum(len(v) for v in pending.values()), len(order) - 2)
        self.assertNotIn((3, 0), [j for v in pending.values() for j in v])

    def test_tick_launches_down_workers_with_work_and_respects_pause_and_completion(self):
        plan = {"plies": {"3": {"shards": 6}, "5": {"shards": 3}}}
        order = ply_data.job_order({3: 6, 5: 3})
        files = {watch_ply_labels.PLAN_PATH}
        store = {watch_ply_labels.PLAN_PATH: json.dumps(plan)}
        uploads, pushes = [], []

        def fake_read(client, listing, path):
            return store.get(path, "")

        def fake_launch(workers, commit):
            pushes.append(list(workers))
            return list(workers), {}

        def fake_upload(client, repo, data, path, **kw):
            uploads.append((path, json.loads(data)))

        statuses = {w: 'x has status "KernelWorkerStatus.COMPLETE"' for w in range(9)}
        statuses[0] = 'x has status "KernelWorkerStatus.RUNNING"'
        statuses[1] = None
        with mock.patch.object(watch_ply_labels, "read_text", fake_read), \
                mock.patch.object(watch_ply_labels, "observe", lambda workers: {w: statuses[w] for w in workers}), \
                mock.patch.object(watch_ply_labels, "launch", fake_launch), \
                mock.patch.object(watch_ply_labels, "upload_file", fake_upload):
            self.assertEqual(watch_ply_labels.tick(None, set(), self.HEAD)["phase"], "disabled")
            state = watch_ply_labels.tick(None, files, self.HEAD)
            self.assertEqual(state["phase"], "launched")
            self.assertEqual(pushes, [[2, 3, 4, 5, 6, 7, 8]])           # not 0 (running), 1 (unknown), 9+ (no work)
            self.assertEqual(state["workers"]["active"], 1)
            self.assertEqual(state["workers"]["wait"], 1)
            self.assertEqual(sorted(uploads[0][1]), [f"{w:02d}" for w in range(2, 9)])
            self.assertEqual(uploads[0][1]["02"]["commit"], self.HEAD)
            self.assertEqual(state["shards"], {"total": 9, "complete": 0,
                                               "by_ply": {"3": {"complete": 0, "total": 6},
                                                          "5": {"complete": 0, "total": 3}}})
            checked = watch_ply_labels.tick(None, files, self.HEAD, check_only=True)
            self.assertEqual(len(pushes), 1)                              # check-only launches nothing
            self.assertEqual(checked["phase"], "running")
            paused = watch_ply_labels.tick(None, files | {watch_ply_labels.PAUSE_PATH}, self.HEAD)
            self.assertEqual(paused["phase"], "paused")
            self.assertEqual(len(pushes), 1)
            done = files | {f"{ply_data.PREFIX}/ply{p}/shard-{s:05d}/{name}" for p, s in order
                            for name in ("train_set.npz", "meta.json")}
            self.assertEqual(watch_ply_labels.tick(None, done, self.HEAD)["phase"], "complete")
            store[watch_ply_labels.worker_status_path(2)] = f"FAILED labels source_commit={self.HEAD} boom"
            blocked = watch_ply_labels.tick(None, files, self.HEAD, check_only=True)
            self.assertEqual(blocked["phase"], "blocked")
            self.assertIn("02", blocked["blocked"])

    def test_every_worker_has_one_slug_and_a_valid_account(self):
        slugs = {watch_ply_labels.worker_slug(w) for w in range(watch_ply_labels.WORKERS)}
        self.assertEqual(len(slugs), 25)
        self.assertEqual({watch_ply_labels.worker_account(w) for w in range(25)},
                         set(watch_ply_labels.ACCOUNTS))
        self.assertEqual([watch_ply_labels.worker_account(w) for w in (0, 4, 5, 24)],
                         [watch_ply_labels.ACCOUNTS[i] for i in (0, 0, 1, 4)])
        cfg = watch_ply_labels.worker_config(7, self.HEAD)
        self.assertEqual((cfg["worker_index"], cfg["worker_count"], cfg["engines"]), (7, 25, 4))
        self.assertTrue(watch_ply_labels.TEMPLATE.read_text().count(f'"""{watch_ply_labels.MARKER}"""') == 1)


class PlyCompareTests(unittest.TestCase):
    def run_pair(self, solved_only_t, solved_only_c, both, first_t, first_c, later_t, later_c):
        """Two runs on the same puzzles with the given discordant pairs and failure counts."""
        n = solved_only_t + solved_only_c + both + 60
        ids = [f"p{i}" for i in range(n)]
        c, t = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        t[:solved_only_t] = True
        c[solved_only_t:solved_only_t + solved_only_c] = True
        c[solved_only_t + solved_only_c:solved_only_t + solved_only_c + both] = True
        t[solved_only_t + solved_only_c:solved_only_t + solved_only_c + both] = True

        def plies(correct, first, later):
            out, idx = [None] * n, [i for i in range(n) if not correct[i]]
            for k, i in enumerate(idx[:first + later]):
                out[i] = 1 if k < first else 3
            return out

        return ({"ids": ids, "correct": c.astype(int).tolist(), "fail_ply": plies(c, first_c, later_c)},
                {"ids": ids, "correct": t.astype(int).tolist(), "fail_ply": plies(t, first_t, later_t)})

    def test_exact_mcnemar_matches_known_values(self):
        self.assertEqual(compare_ply_runs.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(compare_ply_runs.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(compare_ply_runs.mcnemar_exact(10, 0), 2 / 1024)
        self.assertAlmostEqual(compare_ply_runs.mcnemar_exact(0, 10), 2 / 1024)
        self.assertAlmostEqual(compare_ply_runs.mcnemar_exact(3, 1), 2 * 5 / 16)     # n=4: (1+4)/16 twice

    def test_paired_counts_gain_discordant_pairs_and_failure_split(self):
        control, treatment = self.run_pair(40, 10, 100, first_t=20, first_c=20, later_t=15, later_c=30)
        s = compare_ply_runs.paired(control, treatment)
        self.assertEqual((s["only_treatment"], s["only_control"], s["gain"]), (40, 10, 30))
        self.assertEqual(s["failures"]["treatment"], {"first": 20, "later": 15, "by_ply": {"1": 20, "3": 15}})
        self.assertLess(s["mcnemar_p"], 0.001)
        with self.assertRaises(ValueError):
            compare_ply_runs.paired(control, dict(treatment, ids=treatment["ids"][::-1]))

    def verdict_for(self, early, middle, final, dev):
        per = {10_000: compare_ply_runs.paired(*self.run_pair(**early)),
               12_500: compare_ply_runs.paired(*self.run_pair(**middle)),
               15_000: compare_ply_runs.paired(*self.run_pair(**final))}
        return compare_ply_runs.verdict(per, dev)

    def test_verdict_adopts_only_when_every_criterion_holds(self):
        good = dict(solved_only_t=45, solved_only_c=15, both=200, first_t=40, first_c=40, later_t=30, later_c=45)
        early = dict(good, solved_only_t=35, solved_only_c=15)
        ok = self.verdict_for(early, early, good, {"control": 3.77, "treatment": 3.78})
        self.assertEqual(ok["outcome"], "ADOPT")
        self.assertEqual(ok["gains"], {10_000: 20, 12_500: 20, 15_000: 30})
        # a good last evaluation alone is not consistent enough
        flat = dict(good, solved_only_t=15, solved_only_c=15)
        lucky = self.verdict_for(flat, flat, good, {"control": 3.77, "treatment": 3.78})
        self.assertFalse(lucky["checks"]["consistent"])
        self.assertEqual(lucky["outcome"], "INCONCLUSIVE")
        # the gain must come from later plies
        wrong = dict(good, later_t=44, first_t=30)
        self.assertFalse(self.verdict_for(early, early, wrong,
                                          {"control": 3.77, "treatment": 3.78})["checks"]["mechanism"])
        # forgetting the broad corpus fails the last check
        hurt = self.verdict_for(early, early, good, {"control": 3.77, "treatment": 3.90})
        self.assertFalse(hurt["checks"]["broad_loss"])
        self.assertEqual(hurt["outcome"], "INCONCLUSIVE")

    def test_verdict_rejects_a_flat_or_worse_treatment(self):
        flat = dict(solved_only_t=12, solved_only_c=10, both=200, first_t=40, first_c=40, later_t=44, later_c=45)
        out = self.verdict_for(flat, flat, flat, {"control": 3.77, "treatment": 3.77})
        self.assertEqual(out["outcome"], "REJECT")

    def test_ply_arms_differ_only_in_ply_groups_and_start_from_the_main_run(self):
        p0, p1 = cont_config.ARMS["p0"]["args"], cont_config.ARMS["p1"]["args"]
        self.assertEqual(p0[:-1], p1[:-1])
        self.assertEqual((p0[-2:], p1[-2:]), (["--ply-groups", "0"], ["--ply-groups", "10"]))
        self.assertEqual(p0[p0.index("--init-hf-run") + 1], "ccgavn-cont1-a4")
        self.assertEqual(p0[p0.index("--init-weights") + 1], "ema")
        self.assertNotIn("--init-checkpoint", p0)         # chosen at launch, identically for both


if __name__ == "__main__":
    unittest.main()
