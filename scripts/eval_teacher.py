#!/usr/bin/env python3
"""Evaluate the zero-shot teacher (Qwen3-VL box + SAM2 mask) on COD benchmarks.

Images whose box generation fails are scored with an empty mask, so the teacher
is measured end-to-end under the same protocol as the student.
"""
import argparse
import pathlib

import numpy as np
import torch
from PIL import Image

from discod.data import LAYOUT
from discod.metric.metric_recorder import MetricRecorder
from discod.prompts import DETECT_SYSTEM, DETECT_USER
from discod.vlm import BoxGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="COD benchmark root (TestDataset/, NC4K/)")
    ap.add_argument("--sam2-pt", required=True, help="SAM2 checkpoint (.pt)")
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--datasets", nargs="+", default=["CAMO", "COD10K", "NC4K"])
    args = ap.parse_args()

    gen = BoxGenerator(args.model)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(args.sam2_cfg, args.sam2_pt).cuda())

    for dsn in args.datasets:
        d = pathlib.Path(args.data_root) / LAYOUT[dsn]
        imgs, gts = d / "Imgs", d / "GT"
        files = sorted(p for p in imgs.iterdir()
                       if p.suffix.lower() in {".jpg", ".png"} and (gts / f"{p.stem}.png").exists())
        boxes = {}
        for b in range(0, len(files), args.batch):
            batch = files[b:b + args.batch]
            boxes.update(gen.boxes([(p.stem, p) for p in batch], DETECT_SYSTEM, DETECT_USER))
            print(f"  [{dsn} boxes {min(b + args.batch, len(files))}/{len(files)}]", flush=True)
        rec = MetricRecorder()
        for p in files:
            gt = np.asarray(Image.open(gts / f"{p.stem}.png").convert("L"), np.uint8)
            gh, gw = gt.shape
            if p.stem in boxes:
                img_np = np.array(Image.open(p).convert("RGB"))
                with torch.inference_mode():
                    predictor.set_image(img_np)
                    mk, _, _ = predictor.predict(box=np.array(boxes[p.stem], dtype=np.float32),
                                                 multimask_output=False)
                pm = (mk[0] > 0).astype(np.uint8) * 255
                if pm.shape != (gh, gw):
                    pm = np.asarray(Image.fromarray(pm).resize((gw, gh), Image.NEAREST), np.uint8)
                rec.update(pm, gt)
            else:
                rec.update(np.zeros((gh, gw), np.uint8), gt)
        mm = rec.show(num_bits=4, return_ndarray=False)["numerical"]
        print(f"[{dsn:9s}] boxes {len(boxes)}/{len(files)} ({len(boxes) / len(files):.1%})  "
              f"SM={mm['SM']:.4f} MAE={mm['MAE']:.4f} wFm={mm['wFm']:.4f} avgE={mm['avgE']:.4f}", flush=True)


if __name__ == "__main__":
    main()
