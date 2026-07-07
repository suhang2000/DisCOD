#!/usr/bin/env python3
"""Step 4: drop the highest-disagreement fraction of pseudo-labels from the manifest."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scores", required=True, help="disagreement scores json")
    ap.add_argument("--out", required=True, help="filtered manifest jsonl")
    ap.add_argument("--drop-ratio", type=float, default=0.05)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.manifest)]
    scores = json.load(open(args.scores))
    scores.sort(key=lambda s: -s["dis"])
    k = int(len(scores) * args.drop_ratio)
    drop = {s["id"] for s in scores[:k]}

    keep = [r for r in recs if r["id"] not in drop]
    with open(args.out, "w") as f:
        f.write("\n".join(json.dumps(r) for r in keep) + "\n")
    print(f"[done] dropped {len(recs) - len(keep)}/{len(recs)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
