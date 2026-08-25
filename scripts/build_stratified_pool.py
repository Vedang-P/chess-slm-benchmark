"""Build 5k stratified RLVR pool: 30% easy>150 /40% med 60-150 /30% hard<=60
Samples 30k from 600k train, scores with Stockfish d12, then stratifies.
Usage: python3 scripts/build_stratified_pool.py --train data/... --out results/rlvr-pool/train-5k.jsonl --sample 30000 --seed 42
"""
import argparse, json, random, re
from pathlib import Path
import chess, chess.engine

USER_RE = re.compile(r"MoveA:\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)\s+MoveB:\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)")
ASSIST_RE = re.compile(r"\bMove\s*([AB])\b\s*[:.\-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)?", re.I)

def parse_row(row):
    msgs=row.get("messages") or []
    if len(msgs)<2: return None
    m=USER_RE.search(msgs[0].get("content",""))
    if not m: return None
    ca,cb=m.group(1),m.group(2)
    am=list(ASSIST_RE.finditer(msgs[1].get("content","")))
    if not am: return None
    last=am[-1]
    label=last.group(1).upper()
    move=last.group(2)
    if move and move not in (ca,cb): return None
    if not move: move=ca if label=="A" else cb
    return {"fen":row.get("fen"),"candidate_a":ca,"candidate_b":cb,"truth_label":label}

def load_exclude(path):
    if not path: return set()
    txt=Path(path).read_text().strip()
    if txt.startswith("["):
        return {r.get("fen") for r in json.loads(txt) if r.get("fen")}
    return {json.loads(l).get("fen") for l in txt.splitlines() if l.strip()}

def score(fen, move, engine):
    board=chess.Board(fen)
    if chess.Move.from_uci(move) not in board.legal_moves:
        return None
    board.push_uci(move)
    try:
        info=engine.analyse(board, chess.engine.Limit(depth=12))
    except Exception:
        return None
    sc=info.get("score")
    if sc is None: return None
    cp=sc.white().score(mate_score=100000)
    return -cp if board.turn else cp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="data/positions/mate-selection-test-noexplain.json")
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    args=ap.parse_args()
    rows=[]
    dropped=0
    print(f"parsing {args.train} ...", flush=True)
    for line in Path(args.train).read_text().splitlines():
        if not line.strip(): continue
        r=parse_row(json.loads(line))
        if r is None: dropped+=1
        else: rows.append(r)
    print(f"parsed {len(rows)} kept, {dropped} dropped",flush=True)
    excl=load_exclude(args.exclude)
    if excl:
        rows=[r for r in rows if r["fen"] not in excl]
        print(f"after exclude {len(excl)} fens: {len(rows)} rows",flush=True)
    rng=random.Random(args.seed)
    if len(rows)>args.sample:
        rows=rng.sample(rows, args.sample)
        print(f"sampled {len(rows)}",flush=True)
    engine=chess.engine.SimpleEngine.popen_uci(args.stockfish)
    buckets={"easy":[],"medium":[],"hard":[]}
    skipped=0
    for i,r in enumerate(rows):
        ea=score(r["fen"], r["candidate_a"], engine)
        eb=score(r["fen"], r["candidate_b"], engine)
        if ea is None or eb is None:
            skipped+=1
            continue
        gap=abs(ea-eb)
        r["eval_a"]=round(ea); r["eval_b"]=round(eb); r["gap"]=gap
        if gap>150: buckets["easy"].append(r)
        elif gap>60: buckets["medium"].append(r)
        else: buckets["hard"].append(r)
        if (i+1)%2000==0:
            print(f"  scored {i+1}/{len(rows)} easy{len(buckets['easy'])} med{len(buckets['medium'])} hard{len(buckets['hard'])} skip{skipped}",flush=True)
    engine.quit()
    print(f"buckets: easy {len(buckets['easy'])} medium {len(buckets['medium'])} hard {len(buckets['hard'])} skipped {skipped}",flush=True)
    # target 30/40/30
    want_easy=int(args.target*0.3); want_med=int(args.target*0.4); want_hard=args.target-want_easy-want_med
    print(f"want easy {want_easy} med {want_med} hard {want_hard}",flush=True)
    # if not enough hard, take all hard and fill with med/easy proportionally
    def take(bucket, want):
        rng.shuffle(bucket)
        return bucket[:want] if len(bucket)>=want else bucket
    easy=take(buckets["easy"], want_easy)
    med=take(buckets["medium"], want_med)
    hard=take(buckets["hard"], want_hard)
    # if hard short, fill from medium then easy
    deficit=want_hard-len(hard)
    if deficit>0:
        extra_med=buckets["medium"][len(med):len(med)+deficit]
        hard+=extra_med[:deficit]
        deficit=want_hard-len(hard)
        if deficit>0:
            extra_easy=buckets["easy"][len(easy):len(easy)+deficit]
            hard+=extra_easy[:deficit]
    # if still short, fill any remaining from leftovers to reach target
    out=easy+med+hard
    # if under target, fill randomly from leftovers
    if len(out)<args.target:
        leftovers=[r for k in buckets.values() for r in k if r not in out]
        rng.shuffle(leftovers)
        out+=leftovers[:args.target-len(out)]
    rng.shuffle(out)
    out=out[:args.target]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w") as f:
        for r in out:
            # keep minimal fields + evals for analysis
            f.write(json.dumps({k:r[k] for k in ["fen","candidate_a","candidate_b","truth_label","eval_a","eval_b","gap"]})+"\n")
    print(f"wrote {args.out} {len(out)} rows (easy {len([r for r in out if r['gap']>150])} med {len([r for r in out if 60<r['gap']<=150])} hard {len([r for r in out if r['gap']<=60])})",flush=True)
    # also write stats
    Path(args.out+".stats.json").write_text(json.dumps({"easy":len([r for r in out if r['gap']>150]),"medium":len([r for r in out if 60<r['gap']<=150]),"hard":len([r for r in out if r['gap']<=60]),"total":len(out)}, indent=2))

if __name__=="__main__":
    main()
