#!/usr/bin/env python3
"""Measure student inference FPS (bs=1, fp32) and parameter count."""
import argparse
import time

import torch

from discod.model import Student


def bench(backbone, size, n=100, warm=20):
    m = Student(backbone=backbone, pretrained=False).cuda().eval()
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.randn(1, 3, size, size).cuda()
    with torch.no_grad():
        for _ in range(warm):
            m(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            m(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
    print(f"{backbone}@{size}: params={n_params:.2f}M  FPS={n / dt:.1f}  latency={1000 * dt / n:.2f}ms",
          flush=True)
    del m
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["pvt_v2_b0:416", "pvt_v2_b2:416", "pvt_v2_b4:416"])
    args = ap.parse_args()
    for c in args.configs:
        bb, sz = c.split(":")
        bench(bb, int(sz))


if __name__ == "__main__":
    main()
