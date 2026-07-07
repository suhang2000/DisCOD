#!/usr/bin/env python3
"""Step 1: generate VLM bounding boxes for unlabeled training images."""
import argparse
import json
import pathlib

from discod.prompts import DETECT_SYSTEM, DETECT_USER
from discod.vlm import BoxGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs-dir", required=True, help="directory of unlabeled training images")
    ap.add_argument("--out", required=True, help="output boxes jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    imgs_dir = pathlib.Path(args.imgs_dir)
    files = sorted(p for p in imgs_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    gen = BoxGenerator(args.model)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(args.out, "w") as out:
        for b in range(0, len(files), args.batch):
            batch = files[b:b + args.batch]
            boxes = gen.boxes([(p.stem, p) for p in batch], DETECT_SYSTEM, DETECT_USER)
            for p in batch:
                if p.stem in boxes:
                    out.write(json.dumps({"id": p.stem, "img_path": str(p),
                                          "bbox": boxes[p.stem]}) + "\n")
                    n_ok += 1
            out.flush()
            print(f"  [{min(b + args.batch, len(files))}/{len(files)}] boxes={n_ok}", flush=True)
    print(f"[done] {n_ok}/{len(files)} boxes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
