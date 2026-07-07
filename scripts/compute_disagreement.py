#!/usr/bin/env python3
"""Step 3: compute the cross-prompt box-disagreement score d_i for every sample.

Queries the VLM with K=5 prompts of distinct reasoning perspectives and scores
each image by one minus the mean pairwise IoU of the returned boxes. Images
with fewer than two valid boxes get d_i = 1.
"""
import argparse
import json
import pathlib

import numpy as np

from discod.geometry import bbox_iou
from discod.prompts import DISAGREEMENT_PROMPTS
from discod.vlm import BoxGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest jsonl from generate_masks.py")
    ap.add_argument("--out", required=True, help="output disagreement scores json")
    ap.add_argument("--out-boxes", default=None, help="optional dump of per-prompt boxes json")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.manifest)]
    gen = BoxGenerator(args.model)

    allb = [dict() for _ in DISAGREEMENT_PROMPTS]
    for pi, (sp, up) in enumerate(DISAGREEMENT_PROMPTS):
        for b in range(0, len(recs), args.batch):
            batch = recs[b:b + args.batch]
            allb[pi].update(gen.boxes([(r["id"], r["img_path"]) for r in batch], sp, up))
        print(f"  prompt {pi}: {len(allb[pi])}/{len(recs)} boxes", flush=True)
        if args.out_boxes:
            pathlib.Path(args.out_boxes).parent.mkdir(parents=True, exist_ok=True)
            open(args.out_boxes, "w").write(json.dumps([{k: v for k, v in d.items()} for d in allb]))

    scores = []
    for r in recs:
        bs = [allb[pi].get(r["id"]) for pi in range(len(DISAGREEMENT_PROMPTS))]
        bs = [b for b in bs if b is not None]
        if len(bs) < 2:
            d = 1.0
        else:
            ious = [bbox_iou(bs[i], bs[j]) for i in range(len(bs)) for j in range(i + 1, len(bs))]
            d = 1.0 - float(np.mean(ious))
        scores.append({"id": r["id"], "dis": d})

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").write(json.dumps(scores))
    print(f"[done] {len(scores)} scores -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
