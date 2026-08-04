"""Engine/dataset gate tests for the chess study.

All 8x8 scoring runs on python-chess (standard chess). This suite verifies:
  1. The committed task sets are self-consistent under python-chess
     (FEN parses, pieces match, oracles legal, mate moves actually mate).
  2. Standard-rules regression: castling, en passant, double-step legal.
  3. Answer extraction: from-to + SAN parsing never invents moves.

`--quick` skips the per-record dataset sweep (the slow part) and runs only
the extraction/scoring/writer regressions.

Usage:
    python scripts/test_engine.py            # full gate
    python scripts/test_engine.py --quick    # extraction + scoring only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"ok   {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


def test_dataset_standard_chess() -> None:
    """Committed 8x8 datasets must be fully consistent under standard chess."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "positions"
    for name in ("mate1-lichess", "mate2-lichess", "bestmove-8x8"):
        recs = json.loads((data_dir / f"{name}.json").read_text())
        check(f"dataset {name} non-empty", len(recs) >= 40)
        for rec in recs:
            try:
                board = chess.Board(rec["presented_fen"])
            except ValueError:
                check(f"{rec['id']} fen parses", False, rec["presented_fen"])
                continue
            fen_map = {chess.square_name(sq): p.symbol()
                       for sq in chess.SQUARES if (p := board.piece_at(sq)) is not None}
            rec_map = {p["sq"]: (p["kind"] if p["color"] == "w" else p["kind"].lower())
                       for p in rec["pieces"]}
            check(f"{rec['id']} pieces match fen", fen_map == rec_map)
            check(f"{rec['id']} turn matches",
                  ("w" if board.turn == chess.WHITE else "b") == rec["turn"])
            extra = rec["task_extra"]
            if name == "mate1-lichess":
                for uci in extra.get("mate_moves", []):
                    b = chess.Board(rec["presented_fen"])
                    b.push_uci(uci)
                    check(f"{rec['id']} {uci} mates", b.is_checkmate())
            elif name == "mate2-lichess":
                mv = chess.Move.from_uci(extra["first_move"])
                check(f"{rec['id']} first_move legal", mv in board.legal_moves)
            else:  # bestmove
                mv = chess.Move.from_uci(extra["best_move"])
                check(f"{rec['id']} best_move legal", mv in board.legal_moves)

    # MATE move-selection set: FEN parses, candidates are legal, truth label sane
    mate = json.loads((data_dir / "mate-selection-test.json").read_text())
    check("dataset mate-selection-test non-empty", len(mate) >= 100)
    for rec in mate:
        extra = rec["task_extra"]
        board = chess.Board(rec["fen"])
        ca = chess.Move.from_uci(extra["candidate_a"])
        cb = chess.Move.from_uci(extra["candidate_b"])
        check(f"{rec['id']} candidate_a legal", ca in board.legal_moves)
        check(f"{rec['id']} candidate_b legal", cb in board.legal_moves)
        check(f"{rec['id']} truth label sane",
              extra["truth_label"] in ("A", "B"))
        check(f"{rec['id']} candidates differ", extra["candidate_a"] != extra["candidate_b"])

    # standard-rules regression: these moves MUST be legal
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    check("standard: castling e1g1 legal", chess.Move.from_uci("e1g1") in b.legal_moves)
    b = chess.Board("rnbqkbnr/pppppppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    check("standard: en passant d4e3 legal", chess.Move.from_uci("d4e3") in b.legal_moves)
    b = chess.Board()
    check("standard: double-step e2e4 legal", chess.Move.from_uci("e2e4") in b.legal_moves)


def test_answer_extraction() -> None:
    """Extraction must never invent a move. from-to first (last MOVE: line),
    then line-anchored, then SAN via python-chess (legal-only)."""
    from src.benchmarks.games.tasks import parse_move_output

    def board_from_fen(fen):
        return chess.Board(fen)

    cases = [
        # (fen, output, expected_uci, expected_fmt)
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: d2f3", "d2f3", "fromto"),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: Nf3", "d2f3", "san"),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "I think Rc8 is best.", None, None),  # prose without an answer
        ("2kr1b1r/p1p2pp1/2pqN3/7p/6n1/2NPB3/PPP2PPP/R2Q1RK1 b - - 0 1",
         "MOVE: bKb8#", "c8b8", "san"),      # color-prefixed king move
        ("4r3/1k6/pp3P2/1b5p/3R1p2/P1R2P2/1P4PP/6K1 b - - 0 1",
         "MOVE: Rc8", "e8c8", "san"),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "candidate MOVE: a2a3 then MOVE: e1g1", "e1g1", "fromto"),
        # last MOVE: line wins over earlier drafts
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: h2h4 MOVE: d2f3", "d2f3", "fromto"),
        # line-anchored bare from-to
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "the answer is\nd2f3", "d2f3", "fromto"),
        # prose containing a move string must NOT parse (no MOVE:, not a line)
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "e2e4 would be bad here.", None, None),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
         "MOVE: O-O", "e1g1", "san"),
        ("8/1P6/8/8/8/8/8/k1K5 w - - 0 1",
         "MOVE: b8=Q", "b7b8q", "san"),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "", None, None),
        ("r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19",
         "MOVE: z9z9", None, None),   # malformed squares never parse
    ]
    for fen, out, exp_uci, exp_fmt in cases:
        board = board_from_fen(fen) if fen else None
        uci, fmt = parse_move_output(out, board)
        check(f"extract {out!r}",
              uci == exp_uci and fmt == exp_fmt,
              f"got {(uci, fmt)} want {(exp_uci, exp_fmt)}")


def test_api_errors_are_not_answers() -> None:
    """A gateway/transport failure must never become a scored move.

    Regression: the client returned content="ERROR HTTP 400: {...}" and the
    runner scored that string as model text. An error body containing a
    square-like token parsed as SAN and was recorded as a LEGAL move
    ('ERROR HTTP 400: {"code":"e4"...}' -> e3e4), which both fabricates an
    answer and counts infrastructure noise as a model failure.
    """
    from src.benchmarks.games.tasks import parse_move_output, score_mate1

    fen = "r5k1/pp3p1p/2b2qp1/3pr3/8/4P2P/R1PN1PP1/Q3K2R w K - 0 19"
    board = chess.Board(fen)
    rec = {"id": "mate1-x", "presented_fen": fen,
           "task_extra": {"mate_moves": ["a1a8"]}}

    # the exact string that used to parse into a legal move
    poison = 'ERROR HTTP 400: {"code":"e4","detail":"bad request"}'
    check("error body would parse as a move (why content must stay empty)",
          parse_move_output(poison, board) == ("e3e4", "san"))

    scored = score_mate1(rec, "win", "", api_error="HTTP 500: gateway down")
    check("api_error status", scored["status"] == "api_error", str(scored))
    check("api_error keeps the reason", "gateway down" in scored.get("error", ""))

    empty = score_mate1(rec, "win", "")
    check("empty answer without api_error is no_answer",
          empty["status"] == "no_answer", str(empty))

    from src.report import aggregate_samples

    agg = aggregate_samples([
        {"condition": "win", "status": "legal", "compliance": True},
        {"condition": "win", "status": "legal", "compliance": False},
        {"condition": "win", "status": "api_error"},
    ])["conditions"]["win"]
    check("api_error excluded from n", agg["n"] == 2, str(agg))
    check("api_error counted separately", agg["api_error"] == 1)
    check("api_error not in accuracy denominator",
          agg["compliance_strict"] == 0.5, str(agg["compliance_strict"]))


def test_prompts_match_their_task() -> None:
    """mate-in-2 must not be asked for a mate in one.

    Regression: PROMPT_BUILDERS["mate2"] was build_mate1_prompt, so every
    mate-in-2 position was prompted with "Deliver CHECKMATE in exactly one
    move" — a demand the position cannot satisfy — and then scored against
    the first move of a two-move line.
    """
    from src.benchmarks.games.tasks import PROMPT_BUILDERS

    rec = {"n": 8, "turn": "w", "presented_fen": chess.Board().fen(), "pieces": []}
    p1 = PROMPT_BUILDERS["mate1"](rec, "win", variant="fen")
    p2 = PROMPT_BUILDERS["mate2"](rec, "win", variant="fen")
    check("mate1 prompt asks for mate in one", "in exactly one move" in p1)
    check("mate2 prompt does NOT ask for mate in one",
          "in exactly one move" not in p2, p2)
    check("mate2 prompt asks for the first move of a forced mate",
          "CHECKMATE IN TWO" in p2 and "FIRST move" in p2, p2)


def test_mate_selection_parser() -> None:
    """The A/B parser must score the model's FINAL choice."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_mate_eval import parse_choice

    a, b = "e2e4", "d5d4"
    cases = [
        ("MoveB:d5d4", "B"),
        ("MoveA:e2e4", "A"),
        # reasoning inside `content` (every local gemma run): the answer is last
        ("MoveA loses a piece, therefore MoveB:d5d4", "B"),
        ("MoveB hangs the rook. Answer: MoveA:e2e4", "A"),
        ("movea:e2e4", "A"),
        ("no choice in here at all", None),
        ("", None),
    ]
    for text, want in cases:
        got, _ = parse_choice(text, a, b)
        check(f"mate-selection parse {text!r}", got == want, f"got {got} want {want}")


def test_result_writer_does_not_append_across_runs() -> None:
    """Re-running a cell must replace its samples, not append to them.

    Regression: a stale/partial cell re-run left both attempts' rows in the
    JSONL that feeds the paper figures, while the summary — computed from
    memory — looked clean.
    """
    import tempfile

    from src.report import ResultWriter

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        w1 = ResultWriter(d, "cell", {})
        w1.add({"condition": "win", "status": "legal", "compliance": True})
        w1.finish({})
        w2 = ResultWriter(d, "cell", {})
        w2.add({"condition": "win", "status": "legal", "compliance": True})
        summary = w2.finish({})
        rows = [ln for ln in (d / "cell.samples.jsonl").read_text().splitlines() if ln.strip()]
        check("re-run truncates the samples file", len(rows) == 1, f"{len(rows)} rows")
        check("n_samples matches the file", summary["n_samples"] == len(rows))

        w3 = ResultWriter(d, "cell", {}, resume=True, prior_samples=len(rows))
        w3.add({"condition": "win", "status": "legal", "compliance": True})
        s3 = w3.finish({})
        rows3 = [ln for ln in (d / "cell.samples.jsonl").read_text().splitlines() if ln.strip()]
        check("resume appends", len(rows3) == 2, f"{len(rows3)} rows")
        check("resume counts prior samples in n_samples",
              s3["n_samples"] == 2, str(s3["n_samples"]))


def test_backends_share_one_generate_signature() -> None:
    """Every backend must accept the arguments the runners actually pass.

    Regression: HFModel.generate had no on_chunk/thinking_* parameters, so
    every non-smoke gemma run — the entire local half of the study — died
    with TypeError on the first position.
    """
    import inspect

    from src.models import HFModel, OpenCodeGoModel

    required = {"prompt", "max_new_tokens", "stream", "on_chunk",
                "thinking_budget", "thinking_disabled"}
    for cls in (HFModel, OpenCodeGoModel):
        params = set(inspect.signature(cls.generate).parameters)
        missing = required - params
        check(f"{cls.__name__}.generate accepts the runner's kwargs",
              not missing, f"missing {sorted(missing)}")


class _FakeSSEResp:
    """Minimal stand-in for the urlopen() response _sse_chunks iterates.
    `lines` are raw SSE lines (already utf-8-encodable); an empty list
    simulates a stream that opens and closes with zero bytes -- exactly the
    mate-sel-00543 failure signature (stream_events=0, no finish_reason)."""

    def __init__(self, lines: list) -> None:
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for line in self._lines:
            yield (line + "\n").encode()


class _SlowFakeSSEResp(_FakeSSEResp):
    """Like _FakeSSEResp, but iteration blocks for `delay_s` before yielding
    anything -- simulates the mate-sel-02999 pattern (connection open,
    genuinely waiting, no bytes yet) long enough for the heartbeat to fire."""

    def __init__(self, lines: list, delay_s: float) -> None:
        super().__init__(lines)
        self._delay_s = delay_s

    def __iter__(self):
        import time as _time
        _time.sleep(self._delay_s)
        yield from super().__iter__()


def test_silent_stream_is_retried_not_recorded_as_truncated() -> None:
    """A stream that opens (real HTTP response) and closes with ZERO delta
    events and no finish_reason is a stalled transport, not a model answer.

    Regression, measured 2026-08-04: mate-sel-00543 failed this exact way
    three times on Kaggle (stream_events=0, connection open 92-147s, no
    tokens, silence, close -- recorded as no_answer/truncated) then answered
    correctly in 30.7s on an isolated retry with nothing else competing for
    the gateway. Silence-with-no-finish-signal must be retried like a raised
    transport error; a real outcome (any token, or an explicit finish_reason
    including a genuine length cutoff) must be accepted on the first try and
    never retried, so a real truncation is still recorded honestly.
    """
    from src.models import OpenCodeGoModel

    real_chunk = ('data: {"choices":[{"delta":{"reasoning_content":"thinking"},'
                 '"finish_reason":null}]}')
    stop_chunk = ('data: {"choices":[{"delta":{"content":"MOVE: e2e4"},'
                 '"finish_reason":"stop"}]}')

    def run(post_sequence, expect_attempts_ge, expect_content, expect_reasoning_nonempty):
        m = OpenCodeGoModel("deepseek-v4-flash")
        m.key = "test"
        calls = {"n": 0}

        def fake_post_with_retry(payload, stream=False):
            i = calls["n"]
            calls["n"] += 1
            lines = post_sequence[min(i, len(post_sequence) - 1)]
            return _FakeSSEResp(lines), 1

        m._post_with_retry = fake_post_with_retry
        out = m.generate("prompt", max_new_tokens=100, stream=True)
        check(f"silent-stream case calls={calls['n']}: content",
              out["content"] == expect_content, repr(out["content"]))
        check(f"silent-stream case calls={calls['n']}: reasoning present",
              bool(out["reasoning"].strip()) == expect_reasoning_nonempty)
        check(f"silent-stream case calls={calls['n']}: retried the right number of times",
              calls["n"] >= expect_attempts_ge, f"only {calls['n']} calls")
        return out

    # 1) silent, then real content+stop on the retry -> must recover
    run([[], [real_chunk, stop_chunk]], expect_attempts_ge=2,
        expect_content="MOVE: e2e4", expect_reasoning_nonempty=True)

    # 2) real content on the FIRST try -> must NOT retry (a real answer is
    #    never discarded just to try again)
    out = run([[real_chunk, stop_chunk], []], expect_attempts_ge=1,
              expect_content="MOVE: e2e4", expect_reasoning_nonempty=True)

    # 3) a genuine length-cutoff (reasoning flowed, no explicit finish_reason,
    #    matching a real truncation like mate-sel-02999) must be accepted
    #    immediately, not treated as silent
    reasoning_only_chunk = ('data: {"choices":[{"delta":'
                           '{"reasoning_content":"still thinking"},'
                           '"finish_reason":null}]}')
    m = OpenCodeGoModel("deepseek-v4-flash")
    m.key = "test"
    calls = {"n": 0}

    def fake_post_with_retry(payload, stream=False):
        calls["n"] += 1
        return _FakeSSEResp([reasoning_only_chunk]), 1

    m._post_with_retry = fake_post_with_retry
    out = m.generate("prompt", max_new_tokens=100, stream=True)
    check("real partial reasoning (no finish_reason) is accepted, not retried",
          calls["n"] == 1, f"{calls['n']} calls")
    check("real partial reasoning still recorded (not discarded)",
          "still thinking" in out["reasoning"])

    # 4) persistently silent (every attempt empty) -> gives up honestly,
    #    capped retries, never fabricates content
    out = run([[], [], []], expect_attempts_ge=3,
              expect_content="", expect_reasoning_nonempty=False)


def test_heartbeat_pings_dashboard_during_a_silent_wait() -> None:
    """A stream that stays open but silent must still call on_chunk
    periodically, so the live dashboard's timestamp keeps ticking instead of
    looking identical to a frozen process for however long the wait lasts
    (measured 2026-08-04: up to 13 minutes on mate-sel-02999). The heartbeat
    must never write into the scored content/reasoning -- only empty pings.
    """
    from src.models import OpenCodeGoModel

    m = OpenCodeGoModel("deepseek-v4-flash")
    m.key = "test"
    m.HEARTBEAT_INTERVAL_S = 0.05  # fast for the test; real value is 5s

    def fake_post_with_retry(payload, stream=False):
        return _SlowFakeSSEResp([], delay_s=0.3), 1

    m._post_with_retry = fake_post_with_retry

    pings = []
    out = m.generate("prompt", max_new_tokens=100, stream=True,
                     on_chunk=lambda p: pings.append(p))
    heartbeats = [p for p in pings if p.get("phase") == "reasoning"]
    check("heartbeat fired at least once during a silent 0.3s wait",
          len(heartbeats) >= 1, f"{len(pings)} total on_chunk calls")
    check("heartbeat pings never carry content or reasoning text",
          all(p.get("content") == "" and p.get("reasoning") == "" for p in heartbeats))
    check("a persistently silent stream still returns empty, not fabricated",
          out["content"] == "" and out["reasoning"] == "")


def test_offset_partitions_dataset_for_parallel_workers() -> None:
    """5 Kaggle workers must cover positions 100-999 with no gaps and no
    overlaps, and never re-touch the first 100 (already scored, archived,
    not to be re-run -- user decision 2026-08-04). 5, not 6: Kaggle caps
    concurrent CPU kernel sessions at 5 per account (measured directly --
    a 6th push failed with "Maximum batch CPU session count of 5 reached"
    -- user decision 2026-08-04: design around 5, don't queue a 6th).

    Regression risk: a fencepost error in --offset/--n slicing would either
    silently drop positions (a "1000-position" run that's actually short)
    or silently double-score some (wasted spend, and a position appearing
    twice in the merged dataset would corrupt any per-position analysis).
    """
    data_path = (Path(__file__).resolve().parent.parent
                 / "data" / "positions" / "mate-selection-test.json")
    all_records = json.loads(data_path.read_text())
    check("dataset has exactly 1000 positions", len(all_records) == 1000,
          f"{len(all_records)}")

    baseline_ids = {r["id"] for r in all_records[:100]}
    worker_slices = [(100 + i * 180, 180) for i in range(5)]  # 5x180 = 900
    check("worker slices cover exactly 900 positions",
          sum(n for _, n in worker_slices) == 900)

    seen_ids: dict = {}
    for offset, n in worker_slices:
        for r in all_records[offset: offset + n]:
            seen_ids[r["id"]] = seen_ids.get(r["id"], 0) + 1
    check("no worker slice overlaps another",
          all(count == 1 for count in seen_ids.values()),
          f"{sum(1 for c in seen_ids.values() if c != 1)} ids seen != once")
    check("worker slices touch none of the first 100 (already scored)",
          not (seen_ids.keys() & baseline_ids),
          f"overlap: {sorted((seen_ids.keys() & baseline_ids))[:5]}")
    check("baseline + 6 worker slices union to all 1000 positions",
          (seen_ids.keys() | baseline_ids) == {r["id"] for r in all_records},
          f"{1000 - len(seen_ids.keys() | baseline_ids)} ids missing from the union")


def test_final_metrics_accuracy_math() -> None:
    """final_metrics() (used for BOTH the periodic HF checkpoint mid-run
    and the true end-of-run summary -- see run_mate_eval.py) must compute
    correct, honest rates on a small hand-verified set.

    Regression risk this guards: an earlier draft of the periodic
    checkpoint wrote a placeholder metrics dict (only n_attempted, no real
    accuracy) straight to summary.json and uploaded that to the public HF
    archive. Being the exact same function for both call sites is what
    makes that class of bug impossible now, but the function's own math
    still needs an independent, hand-computed check.
    """
    from run_mate_eval import final_metrics

    def row(status, compliance):
        return {"status": status, "compliance": compliance}

    rows = [
        row("correct", True), row("correct", True), row("correct", True),
        row("wrong", False),
        row("no_answer", None),
        row("parse_error", None),
        row("api_error", None), row("api_error", None),
    ]
    m = final_metrics(rows)["accuracy"]
    # 8 rows total, 2 api_error -> 6 scored; of those 4 are parsed (3 correct + 1 wrong)
    check("final_metrics n excludes api_error", m["n"] == 6, str(m["n"]))
    check("final_metrics n_attempted counts every row", m["n_attempted"] == 8, str(m["n_attempted"]))
    check("final_metrics api_error count", m["api_error"] == 2, str(m["api_error"]))
    check("final_metrics correct count", m["correct"] == 3, str(m["correct"]))
    check("final_metrics wrong count", m["wrong"] == 1, str(m["wrong"]))
    check("final_metrics parse_rate = parsed/scored = 4/6",
          m["parse_rate"] == round(4 / 6, 4), str(m["parse_rate"]))
    check("final_metrics accuracy_strict = correct/scored = 3/6",
          m["accuracy_strict"] == 0.5, str(m["accuracy_strict"]))
    check("final_metrics accuracy_of_parsed = correct/parsed = 3/4",
          m["accuracy_of_parsed"] == 0.75, str(m["accuracy_of_parsed"]))


def test_combine_mate_metrics_matches_pooled_recompute() -> None:
    """combine_mate_metrics(per-worker dicts) must exactly equal
    compute_mate_metrics() run once over ALL workers' rows pooled together
    -- for every field that's a sum of raw counts. If those two ever
    diverge, the multi-worker dashboard's headline accuracy would silently
    lie relative to what a from-scratch recompute over the same data says,
    which is exactly the failure mode this whole design exists to prevent
    (see src/mate_metrics.py docstring).

    positions_per_hour is deliberately excluded from the equality check: it
    is a per-process wall-clock throughput, not derivable from pooled rows
    without knowing each worker's own elapsed time, so it is checked
    separately by construction (parallel rates must simply sum).
    """
    from src.mate_metrics import combine_mate_metrics, compute_mate_metrics

    def row(i, status, compliance, label, truth, reasoning_reason=None):
        return {
            "status": status, "compliance": compliance, "label": label,
            "no_answer_reason": reasoning_reason,
            "position_metadata": {"task_extra": {"truth_label": truth}},
            "latency_ms": 1000 + i * 10,
            "token_usage": {"output_tokens": 100 + i, "reasoning_tokens": 50 + i},
        }

    worker_rows = [
        [row(0, "correct", True, "A", "A"), row(1, "wrong", False, "B", "A"),
         row(2, "correct", True, "B", "B"), row(3, "no_answer", None, None, "A", "gave_up")],
        [row(4, "correct", True, "A", "A"), row(5, "correct", True, "B", "B"),
         row(6, "wrong", False, "A", "B"), row(7, "api_error", None, None, None)],
        [row(8, "correct", True, "B", "B"), row(9, "no_answer", None, None, "B", "gave_up"),
         row(10, "parse_error", None, None, "A")],
    ]
    parts = [compute_mate_metrics(rows, done_here=0, elapsed_h=None) for rows in worker_rows]
    combined = combine_mate_metrics(parts)
    pooled = compute_mate_metrics([r for rows in worker_rows for r in rows],
                                  done_here=0, elapsed_h=None)

    exact_fields = ["n", "n_attempted", "answered", "correct", "wrong", "no_answer",
                    "parse_error", "api_error", "accuracy", "accuracy_of_answered",
                    "answer_rate", "picked_a", "picked_b", "truth_a", "truth_b",
                    "correct_truth_a", "correct_truth_b", "accuracy_truth_a",
                    "accuracy_truth_b", "no_answer_reasons",
                    "mean_latency_s", "mean_output_tokens", "mean_reasoning_tokens"]
    for field in exact_fields:
        check(f"combine_mate_metrics[{field}] matches pooled recompute",
              combined[field] == pooled[field],
              f"combined={combined[field]!r} pooled={pooled[field]!r}")

    # positions_per_hour: parallel throughput sums, it does not pool from rows
    parts_with_rate = [dict(p, positions_per_hour=r) for p, r in
                       zip(parts, [10.0, 12.5, None])]
    check("positions_per_hour sums across concurrently-running workers",
          combine_mate_metrics(parts_with_rate)["positions_per_hour"] == 22.5,
          str(combine_mate_metrics(parts_with_rate)["positions_per_hour"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the per-record dataset sweep (CPU-cheap gate)")
    args = ap.parse_args()
    if not args.quick:
        test_dataset_standard_chess()
    test_answer_extraction()
    test_api_errors_are_not_answers()
    test_prompts_match_their_task()
    test_mate_selection_parser()
    test_result_writer_does_not_append_across_runs()
    test_backends_share_one_generate_signature()
    test_silent_stream_is_retried_not_recorded_as_truncated()
    test_heartbeat_pings_dashboard_during_a_silent_wait()
    test_offset_partitions_dataset_for_parallel_workers()
    test_final_metrics_accuracy_math()
    test_combine_mate_metrics_matches_pooled_recompute()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES", flush=True)
        sys.exit(1)
    print("\nALL TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
