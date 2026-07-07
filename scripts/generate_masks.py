#!/usr/bin/env python3
"""Step 2: convert VLM boxes into SAM2 pseudo-masks and write the training manifest."""
import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True, help="boxes jsonl from generate_boxes.py")
    ap.add_argument("--masks-dir", required=True, help="output directory for pseudo-mask PNGs")
    ap.add_argument("--out", required=True, help="output manifest jsonl")
    ap.add_argument("--sam2-pt", required=True, help="SAM2 checkpoint (.pt)")
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    boxes = [json.loads(l) for l in open(args.boxes)]
    masks_dir = pathlib.Path(args.masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    predictor = SAM2ImagePredictor(build_sam2(args.sam2_cfg, args.sam2_pt).cuda())
    with open(args.out, "w") as man:
        for k, r in enumerate(boxes):
            img_np = np.array(Image.open(r["img_path"]).convert("RGB"))
            with torch.inference_mode():
                predictor.set_image(img_np)
                mk, _, _ = predictor.predict(box=np.array(r["bbox"], dtype=np.float32),
                                             multimask_output=False)
            mask_path = masks_dir / f"{r['id']}.png"
            Image.fromarray((mk[0] > 0).astype(np.uint8) * 255).save(mask_path)
            man.write(json.dumps({"id": r["id"], "img_path": r["img_path"],
                                  "mask_path": str(mask_path)}) + "\n")
            if (k + 1) % 400 == 0:
                print(f"  [{k + 1}/{len(boxes)}]", flush=True)
    print(f"[done] {len(boxes)} masks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
