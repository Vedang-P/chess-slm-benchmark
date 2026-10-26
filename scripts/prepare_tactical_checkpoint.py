"""Create and publish the function-preserving 3.9M starting checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402

from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402
from scripts.repair_checkpoint import repair_training_checkpoint  # noqa: E402


def prepare(source_run: str, source_step: int, target_run: str,
            repo: str, sl_repo: Path, workdir: Path) -> dict:
    client = make_hf_api(ROOT)
    target = f"{target_run}/checkpoint-{source_step}"
    existing = set(client.list_repo_files(repo, repo_type="dataset"))
    required = {f"{target}/state.pt", f"{target}/config.json",
                f"{target}/repair-report.json"}
    if required <= existing:
        print(f"[transplant] existing complete checkpoint: {target}", flush=True)
        report_path = Path(hf_hub_download(repo, f"{target}/repair-report.json",
                                           repo_type="dataset", token=client.token))
        return json.loads(report_path.read_text(encoding="utf-8"))

    source = workdir / source_run / f"checkpoint-{source_step}"
    source.mkdir(parents=True, exist_ok=True)
    for name in ("state.pt", "config.json", "metrics.json"):
        try:
            downloaded = hf_hub_download(repo, f"{source_run}/checkpoint-{source_step}/{name}",
                                         repo_type="dataset", local_dir=str(workdir),
                                         token=client.token)
            cached = workdir / source_run / f"checkpoint-{source_step}" / name
            if Path(downloaded) != cached and Path(downloaded).is_file():
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(Path(downloaded).read_bytes())
        except Exception:
            if name != "metrics.json":
                raise
    output = workdir / target
    report = repair_training_checkpoint(source, output, sl_repo)
    client.upload_folder(folder_path=str(output), path_in_repo=target,
                         repo_id=repo, repo_type="dataset",
                         commit_message=f"Prepare exact 3.9M continuation checkpoint {source_step}")
    verified = set(client.list_repo_files(repo, repo_type="dataset"))
    if not required <= verified:
        raise RuntimeError(f"transplanted checkpoint upload incomplete: {sorted(required-verified)}")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-step", required=True, type=int)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    parser.add_argument("--sl-repo", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_run, args.source_step, args.target_run,
            args.hf_repo, args.sl_repo, args.workdir)


if __name__ == "__main__":
    main()
