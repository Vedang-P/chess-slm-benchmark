#!/usr/bin/env python3
"""Poll HF checkpoints + kernel statuses every 5 min, log milestones to logs/runs_watch.log"""
import json, sys, time, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import api
sys.path.insert(0, str(ROOT / 'scripts'))
from launch_trainers import env_for_account

RUNS = {
 'vedanggggg/baseline-5m-seed0':'account1-baseline-5m-seed0',
 'vedanggggg/gavn-3m-seed0':'account1-gavn-3m-seed0',
 'vedangpandeyyy/gavn-5m-seed0':'account2-gavn-5m-seed0',
 'vedangpandeyyy/gavn-3m-seed1':'account2-gavn-3m-seed1',
 'softmaxsimp/gavn-5m-geometry':'account3-gavn-5m-geometry',
 'softmaxsimp/gavn-5m-loss':'account3-gavn-5m-loss',
}
LOG = ROOT / 'logs' / 'runs_watch.log'
seen = {}
def log(m):
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')
def ckpt_latest(c, prefix):
    try:
        files = c.list_repo_files('vedangfake/chess-slm-benchmark', repo_type='dataset')
        hits = [f.split('/')[1] for f in files if f.startswith(prefix+'/') and 'checkpoint-' in f]
        return max(hits, key=lambda x: int(x.split('-')[-1])) if hits else 'NONE'
    except Exception as e:
        return f'ERR {e}'
def status(ref, owner):
    try:
        r = subprocess.run([sys.executable,'-m','kaggle','kernels','status',ref],
                           capture_output=True,text=True,env=env_for_account(owner),timeout=90)
        s = (r.stdout or r.stderr)
        return 'RUNNING' if 'RUNNING' in s else ('COMPLETE' if 'COMPLETE' in s else ('ERROR' if 'ERROR' in s else s.strip()[:60]))
    except Exception as e:
        return f'ERR {e}'
log('watcher armed')
while True:
    c = api(ROOT)
    for ref, prefix in RUNS.items():
        owner = ref.split('/')[0]
        ck = ckpt_latest(c, prefix)
        st = status(ref, owner)
        key = f"{ref}|{ck}"
        if key != seen.get(ref):
            log(f"{ref}: status={st} latest={ck}")
            seen[ref] = key
        elif st not in ('RUNNING','') and st != seen.get(ref+'_s'):
            log(f"{ref}: STATUS CHANGE -> {st}")
            seen[ref+'_s'] = st
    time.sleep(300)
