import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from scripts.data_hygiene import eligible_mask, position_hashes, set_learning_rate
from scripts.eval_gavn import tokenize_fen
from scripts.shard_data import ShardManager
from scripts.train_ccgavn import development_mask, reflect_horizontal


def tiny_ccgavn(version):
    from scripts.train_ccgavn import CCGAVN, candidate_relation_types
    src = np.array([0, 1, 2, 3])
    dst = np.array([1, 2, 3, 0])
    promo = np.zeros(4, dtype=np.int64)
    return CCGAVN(torch, 8, 1, 2, src, dst, promo, candidate_relation_types(),
                  model_version=version)


class CorrectnessTests(unittest.TestCase):
    def test_all_optimizer_groups_follow_lr(self):
        a, b = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.AdamW([{'params': [a]}, {'params': [b]}], lr=5e-4)
        set_learning_rate(opt, 0.0)
        a.grad = torch.ones(1); b.grad = torch.ones(1)
        opt.step()
        self.assertEqual([g['lr'] for g in opt.param_groups], [0., 0.])
        self.assertEqual(a.item(), 1.); self.assertEqual(b.item(), 1.)

    def test_castling_examples_are_never_reflected(self):
        b = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
        tokens = tokenize_fen(b.fen())[None]
        result, actions = reflect_horizontal(tokens, np.array([0]), np.array([1, 0]), np.array([True]))
        np.testing.assert_array_equal(tokens, result)
        np.testing.assert_array_equal(actions, [0])

    def test_reflection_roundtrip_and_split_invariance(self):
        t = tokenize_fen('4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 20')[None]
        r, a = reflect_horizontal(t, np.array([0]), np.array([1, 0]), np.array([True]))
        back, act = reflect_horizontal(r, a, np.array([1, 0]), np.array([True]))
        np.testing.assert_array_equal(t, back)
        np.testing.assert_array_equal(act, [0])
        np.testing.assert_array_equal(position_hashes(t), position_hashes(r))
        np.testing.assert_array_equal(development_mask(t, 100, 0), development_mask(r, 100, 0))

    def test_exclusions_ignore_clocks_and_candidate_action(self):
        a = tokenize_fen('4k3/8/8/8/8/8/4P3/4K3 w - - 0 1')
        b = tokenize_fen('4k3/8/8/8/8/8/4P3/4K3 w - - 31 67')
        np.testing.assert_array_equal(position_hashes(a[None]), position_hashes(b[None]))
        for shard in (a[None], b[None]):
            self.assertFalse(eligible_mask(shard, position_hashes(a[None]), 100, 0)[0])

    def test_dev_fold_excluded_in_every_shard(self):
        t = tokenize_fen(chess.STARTING_FEN)[None]
        fold = int(position_hashes(t)[0] % 100)
        for _ in range(3):
            self.assertFalse(eligible_mask(t, np.array([], dtype=np.uint64), 100, fold)[0])

    def test_2b_launcher_is_disarmed_by_default(self):
        from scripts import watch_2b
        original = watch_2b.ARMED_MARKER
        try:
            watch_2b.ARMED_MARKER = Path("/nonexistent/ccgavn-2b-ARMED")
            os.environ.pop("CCGAVN2B_ARMED", None)
            self.assertFalse(watch_2b.armed())
            os.environ["CCGAVN2B_ARMED"] = "1"
            self.assertTrue(watch_2b.armed())
        finally:
            os.environ.pop("CCGAVN2B_ARMED", None)
            watch_2b.ARMED_MARKER = original

    def test_model_v2_metadata_is_field_aware(self):
        tokens = np.zeros((1, 77), dtype=np.int64)
        tokens[0, 1:65] = 5
        tokens[0, :1] = 3
        tokens[0, 65:69] = 3
        tokens[0, 69:71] = 3
        tokens[0, 71:] = 3
        tokens[0, 71] = 7
        permuted = tokens.copy()
        permuted[0, 0] = tokens[0, 71]
        permuted[0, 71] = tokens[0, 0]
        action = np.array([0])
        torch.manual_seed(0)
        v1, v2 = tiny_ccgavn("v1").eval(), tiny_ccgavn("v2").eval()
        with torch.no_grad():
            # v1 is exactly permutation-invariant up to float summation order
            # (measured max |delta| ~2.4e-7); v2 changes by ~2e-2 or more.
            v1_difference = (v1(torch.as_tensor(tokens), torch.as_tensor(action))
                             - v1(torch.as_tensor(permuted), torch.as_tensor(action))).abs().max()
            self.assertLess(v1_difference.item(), 1e-5)
            v2.global_field.data.copy_(torch.ones(13, 8))
            v2.global_field.data[0] *= 2.0
            v2.global_field.data[11] *= 0.4
            v2.global_embed.weight.data[3].fill_(1.0)
            v2.global_embed.weight.data[7].fill_(3.0)
            difference = (v2(torch.as_tensor(tokens), torch.as_tensor(action))
                          - v2(torch.as_tensor(permuted), torch.as_tensor(action))).abs().max()
            self.assertGreater(difference.item(), 1e-4)

    def test_model_v2_drops_dead_query_side_bias(self):
        v1, v2 = tiny_ccgavn("v1"), tiny_ccgavn("v2")
        self.assertEqual(v1.blocks[0]["dynamic"].out_features, 2 * 2 * 65)
        self.assertEqual(v2.blocks[0]["dynamic"].out_features, 2 * 65)

    def test_eval_jsonl_roundtrip_and_clustering(self):
        from scripts.eval_gavn import _append_jsonl, _load_jsonl, position_key
        from scripts.analyze_mate import compare, summarize
        records = [
            {"file": "a", "row": 0, "position": "P", "correct": True},
            {"file": "b", "row": 0, "position": "P", "correct": False},
            {"file": "a", "row": 1, "position": "Q", "correct": True},
        ]
        self.assertEqual(position_key("8/8/8/8/8/8/8/K6k w - - 3 7"),
                         "8/8/8/8/8/8/8/K6k w - -")
        out = Path(self.id().replace("/", "_") + ".jsonl")
        try:
            for record in records:
                _append_jsonl(out, record)
            self.assertEqual(_load_jsonl(out), records)
        finally:
            out.unlink(missing_ok=True)
        summary = summarize(records, np.random.default_rng(0), 200)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["unique_positions"], 2)
        lo, hi = summary["ci95_clustered"]
        self.assertLessEqual(lo, hi)
        other = [dict(r, correct=not r["correct"]) for r in records]
        result = compare(records, other, np.random.default_rng(0), 200)
        self.assertEqual(result["common_rows"], 3)
        self.assertAlmostEqual(result["delta_b_minus_a"], -1 / 3)

    def test_eval_resume_is_checkpoint_scoped(self):
        from scripts.eval_gavn import (checkpoint_identity, dataset_identity,
                                       validate_examples_identity)
        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = {"dim": 8, "layers": 1, "heads": 2, "source_commit": "abc"}
            checkpoints = {}
            for name in ("checkpoint-1800000", "checkpoint-1900000"):
                cp = tmp / name
                cp.mkdir()
                (cp / "state.pt").write_bytes(b"weights-" + name.encode())
                checkpoints[name] = cp
            mate = tmp / "mate.json"
            mate.write_text("[]")
            dataset = dataset_identity([mate], None)
            id_a = {"checkpoint": checkpoint_identity(
                        checkpoints["checkpoint-1800000"], cfg, "cc-gavn-v1", "v1", "dist"),
                    "dataset": dataset}
            id_b = {"checkpoint": checkpoint_identity(
                        checkpoints["checkpoint-1900000"], cfg, "cc-gavn-v1", "v1", "dist"),
                    "dataset": dataset}
            self.assertNotEqual(id_a, id_b)
            dir_a = tmp / "examples-1800000"
            validate_examples_identity(dir_a, id_a)
            (dir_a / "mate.jsonl").write_text(json.dumps(
                {"file": "mate.json", "row": 0, "position": "P", "correct": True}) + "\n")
            validate_examples_identity(dir_a, id_a)  # same checkpoint resumes
            # Consecutive checkpoint evaluation with a fresh dir is clean...
            dir_b = tmp / "examples-1900000"
            validate_examples_identity(dir_b, id_b)
            self.assertFalse((dir_b / "mate.jsonl").exists())
            # ...and reusing the previous checkpoint's dir must fail hard.
            with self.assertRaises(ValueError):
                validate_examples_identity(dir_a, id_b)
            # Dataset or score changes invalidate saved rows too.
            changed = json.loads(json.dumps(id_a))
            changed["checkpoint"]["score"] = "q"
            with self.assertRaises(ValueError):
                validate_examples_identity(dir_a, changed)
            changed_dataset = json.loads(json.dumps(id_a))
            changed_dataset["dataset"]["mate_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                validate_examples_identity(dir_a, changed_dataset)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_eval_kernel_scopes_examples_per_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "kernels/eval-preview/eval_preview.py").read_text(encoding="utf-8")
        self.assertIn('examples = WORK / f"examples-{step}"', source)
        self.assertNotIn('examples = WORK / "examples"', source)
        self.assertIn('"identity.json"', source)

    def test_2b_launcher_renders_corrected_config(self):
        import ast
        from scripts.watch_2b import (FINAL_STEP, INIT_RUN, RUN, frozen_tags,
                                      render_kernel)
        self.assertNotEqual(RUN, INIT_RUN)
        cfg = {"run": RUN, "init_run": INIT_RUN, "start_step": 1_715_000,
               "total_steps": FINAL_STEP, "source_commit": "a" * 40,
               "tags": frozen_tags()}
        src = render_kernel(cfg)
        self.assertNotIn("__CCGAVN2B_CONFIG__", src)
        tree = ast.parse(src)
        rendered = next(
            node.value.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "CONFIG_JSON")
        self.assertEqual(json.loads(rendered), cfg)
        for flag in ("--stage-start-step", "--exclusion-puzzles",
                     "--init-from-hf-run", "--init-checkpoint"):
            self.assertIn(f'"{flag}"', src)

    def test_corrected_stage_covers_entire_frozen_corpus(self):
        root = Path(__file__).resolve().parents[1]
        tags = json.loads((root / 'configs/ccgavn-2b-shard-tags.json').read_text())['tags']
        m = object.__new__(ShardManager); m.tags = sorted(tags); m.rows = m._override_rows()
        n = 3896658 - 1715000
        schedule = m.schedule(n, np.random.default_rng(0))
        self.assertEqual(len(schedule), n)
        self.assertEqual(set(schedule), set(tags))
        # A resumed process uses a stage-relative offset into the SAME schedule.
        regenerated = m.schedule(n, np.random.default_rng(0))
        np.testing.assert_array_equal(schedule[1234:], regenerated[1234:])


if __name__ == '__main__':
    unittest.main()
