"""Windows-patched runner for the OFFICIAL searchless_chess puzzle eval.

Applies the orbax/tensorstore ocdbt spec fixes required on Windows (see
scripts/sl_mate_eval.py) and then executes the unmodified official
`puzzles.py` (same dataset, same protocol, same code path that produced
the paper's Table 1 puzzle accuracies: 9M 88.9% / 136M 94.5% / 270M 95.4%).

Usage (from repo root; SL_REPO overrides the official repo location):
    python3 scripts/run_sl_puzzles.py --num_puzzles 10000 --agent 9M
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


import re


def _fix_base(spec):
    """Rewrite ocdbt kvstore base 'file://C:\\...' -> 'file:///C:/...'."""
    kv = spec.get("kvstore")
    if (isinstance(kv, dict) and isinstance(kv.get("base"), str)
            and re.match(r"file://[A-Za-z]:", kv["base"])):
        kv["base"] = "file:///" + kv["base"][7:].replace("\\", "/")


def main() -> None:
    sl_repo = os.environ.get("SL_REPO", "C:/tmp/searchless_chess")
    src_dir = Path(sl_repo) / "src"
    if not (src_dir / "puzzles.py").exists():
        sys.exit(f"official repo not found at {sl_repo} (set SL_REPO)")
    os.chdir(src_dir)
    sys.path.insert(0, str(Path(sl_repo).parent))

    import orbax.checkpoint.type_handlers as th
    import jax.experimental.array_serialization.serialization as _ser

    _orig_ts_spec = th.get_tensorstore_spec

    def _win_ts_spec(directory, *a, **k):
        spec = _orig_ts_spec(directory, *a, **k)
        _fix_base(spec)
        return spec

    th.get_tensorstore_spec = _win_ts_spec

    _orig_ad = _ser.async_deserialize

    def _win_ad(sharding, tspec, **kw):
        _fix_base(tspec)
        return _orig_ad(sharding, tspec, **kw)

    # DeepMind's internal eval convention uses the EMA params (train.py
    # evaluates with use_ema_params=True); the released puzzles.py defaults
    # to 'params'. SL_EMA=1 switches the checkpoint variant.
    import searchless_chess.src.training_utils as sl_tu

    if os.environ.get("SL_EMA") == "1":
        _orig_lp = sl_tu.load_parameters

        def _ema_lp(params, *a, **kw):
            kw["use_ema_params"] = True
            return _orig_lp(params, *a, **kw)

        sl_tu.load_parameters = _ema_lp

    from absl import app
    from searchless_chess.src.puzzles import main as puzzles_main

    app.run(puzzles_main)


if __name__ == "__main__":
    main()
