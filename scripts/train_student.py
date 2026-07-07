#!/usr/bin/env python3
"""Step 5: train the PVTv2 student on filtered pseudo-labels and evaluate on COD benchmarks."""
import argparse
import json
import pathlib
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from discod.data import LAYOUT, MEAN, STD, PseudoDataset
from discod.metric.metric_recorder import MetricRecorder
from discod.model import Student, structure_loss


@torch.no_grad()
def evaluate(model, data_root, ds, size, ebs=32):
    """Canonical CPU PySODMetrics evaluation; predictions are resized back to GT resolution."""
    model.eval()
    d = pathlib.Path(data_root) / LAYOUT[ds]
    imgs, gts = d / "Imgs", d / "GT"
    rec = MetricRecorder()
    files = sorted(p for p in imgs.iterdir()
                   if p.suffix.lower() in {".jpg", ".png"} and (gts / f"{p.stem}.png").exists())
    for i in range(0, len(files), ebs):
        bf = files[i:i + ebs]
        xs = []
        for p in bf:
            im = Image.open(p).convert("RGB")
            x = (np.asarray(im.resize((size, size)), np.float32) / 255 - MEAN) / STD
            xs.append(torch.from_numpy(x).permute(2, 0, 1))
        prs = torch.sigmoid(model(torch.stack(xs).float().cuda()))[:, 0].cpu().numpy()
        for j, p in enumerate(bf):
            gt = np.asarray(Image.open(gts / f"{p.stem}.png").convert("L"), np.uint8)
            gh, gw = gt.shape
            prj = np.asarray(Image.fromarray((prs[j] * 255).astype(np.uint8)).resize((gw, gh)), np.uint8)
            rec.update(prj, gt)
    return rec.show(num_bits=4, return_ndarray=False)["numerical"]


@torch.no_grad()
def gpu_eval(model, data_root, ds, size, ebs=64):
    """Fast GPU-batched S-measure/MAE/E-measure at training resolution."""
    from discod.gpu_metrics import torch_emeasure_avg, torch_smeasure
    model.eval()
    d = pathlib.Path(data_root) / LAYOUT[ds]
    imgs, gts = d / "Imgs", d / "GT"
    files = [p for p in imgs.iterdir()
             if p.suffix.lower() in {".jpg", ".png"} and (gts / f"{p.stem}.png").exists()]
    sms, maes, emes = [], [], []
    for i in range(0, len(files), ebs):
        xs, gs = [], []
        for p in files[i:i + ebs]:
            im = Image.open(p).convert("RGB")
            xs.append(torch.from_numpy(
                ((np.asarray(im.resize((size, size)), np.float32) / 255 - MEAN) / STD)).permute(2, 0, 1))
            g = np.asarray(Image.open(gts / f"{p.stem}.png").convert("L").resize((size, size), Image.NEAREST))
            gs.append(torch.from_numpy((g > 127).astype(np.float32)))
        pred = torch.sigmoid(model(torch.stack(xs).float().cuda()))[:, 0]
        pmin = pred.amin(dim=(1, 2), keepdim=True)
        pmax = pred.amax(dim=(1, 2), keepdim=True)
        pred = torch.where(pmax > pmin, (pred - pmin) / (pmax - pmin + 1e-12), pred)
        gt = torch.stack(gs).cuda()
        sms.append(torch_smeasure(pred, gt))
        emes.append(torch_emeasure_avg(pred, gt))
        maes.append((pred - gt).abs().mean(dim=(1, 2)))
    return {"SM": torch.cat(sms).mean().item(), "MAE": torch.cat(maes).mean().item(),
            "avgE": torch.cat(emes).mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="pseudo-label manifest jsonl")
    ap.add_argument("--data-root", required=True, help="COD benchmark root (TestDataset/, NC4K/)")
    ap.add_argument("--out", required=True, help="checkpoint output directory")
    ap.add_argument("--backbone", default="pvt_v2_b0", help="timm backbone (pvt_v2_b0/b2/b4)")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="mixed-precision training")
    ap.add_argument("--swa", action="store_true", help="stochastic weight averaging over late epochs")
    ap.add_argument("--swa-start-frac", type=float, default=0.6)
    ap.add_argument("--strong-aug", action="store_true",
                    help="paired geometric + photometric augmentation")
    ap.add_argument("--aug-hue", type=float, default=0.3, help="per-channel color jitter amplitude")
    ap.add_argument("--aug-cutout", type=int, default=3, help="max number of cutout patches")
    ap.add_argument("--eval-every", type=int, default=80)
    ap.add_argument("--eval-ds", nargs="+", default=["CAMO", "COD10K", "NC4K"])
    ap.add_argument("--gpu-eval", action="store_true",
                    help="fast GPU metrics during training; paper numbers use the canonical CPU eval")
    ap.add_argument("--eval-ckpt", default=None, help="evaluate a checkpoint and exit")
    ap.add_argument("--tag", default=None, help="checkpoint name tag")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tag = args.tag or args.backbone
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    model = Student(backbone=args.backbone).cuda()
    if args.eval_ckpt:
        model.load_state_dict(torch.load(args.eval_ckpt))
        for dsn in args.eval_ds:
            mm = (gpu_eval(model, args.data_root, dsn, args.size) if args.gpu_eval
                  else evaluate(model, args.data_root, dsn, args.size))
            wfm = f" wFm={mm['wFm']:.4f}" if "wFm" in mm else ""
            print(f"[{dsn:9s}] SM={mm['SM']:.4f} MAE={mm['MAE']:.4f}{wfm} avgE={mm['avgE']:.4f}", flush=True)
        return

    items = [json.loads(l) for l in open(args.manifest)]
    print(f"[train] {len(items)} pseudo samples, backbone={args.backbone} tag={tag}", flush=True)
    dl = DataLoader(
        PseudoDataset(items, args.size, True, strong_aug=args.strong_aug,
                      aug_hue=args.aug_hue, aug_cutout=args.aug_cutout),
        batch_size=args.bs, shuffle=True, num_workers=args.num_workers,
        drop_last=True, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    swa_model = None
    if args.swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
    swa_start = int(args.epochs * args.swa_start_frac)

    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for img, m in dl:
            img, m = img.cuda(), m.cuda()
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = structure_loss(model(img), m)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        if swa_model is not None and ep >= swa_start:
            swa_model.update_parameters(model)
        print(f"  ep{ep + 1}/{args.epochs} loss={tot / len(dl):.4f} ({time.time() - t0:.0f}s)", flush=True)

        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            nets = [("student", model)]
            if ep + 1 == args.epochs and swa_model is not None:
                from torch.optim.swa_utils import update_bn
                update_bn(dl, swa_model, device="cuda")
                nets.append(("swa", swa_model))
            for ntag, net in nets:
                for dsn in args.eval_ds:
                    if args.gpu_eval:
                        mm = gpu_eval(net, args.data_root, dsn, args.size)
                        print(f"    [{ntag}|{dsn:9s}|gpu] SM={mm['SM']:.4f} MAE={mm['MAE']:.4f} "
                              f"avgE={mm['avgE']:.4f}", flush=True)
                    else:
                        mm = evaluate(net, args.data_root, dsn, args.size)
                        print(f"    [{ntag}|{dsn:9s}] SM={mm['SM']:.4f} MAE={mm['MAE']:.4f} "
                              f"wFm={mm['wFm']:.4f} avgE={mm['avgE']:.4f}", flush=True)
            sd = (swa_model.module.state_dict() if (swa_model is not None and ep + 1 == args.epochs)
                  else model.state_dict())
            torch.save(sd, f"{args.out}/student_{tag}_ep{ep + 1}.pth")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
