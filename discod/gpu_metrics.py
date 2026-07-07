"""GPU-batched S-measure and E-measure for fast evaluation during training.

Matches the CPU PySODMetrics implementation on non-degenerate COD masks
(max diff < 1e-4). Paper numbers are always reported with the canonical CPU
implementation in `discod.metric`.
"""
import numpy as np
import torch

EPS = float(np.spacing(1))


def torch_smeasure(pred, gt):
    """pred (B,H,W) float in [0,1], gt (B,H,W) bool/float. Returns (B,) S-measure."""
    B, H, W = pred.shape
    area = float(H * W)
    gt = gt.float()
    pred = pred.float()
    y = gt.mean(dim=(1, 2))
    m0, m1 = (y == 0), (y == 1)
    mm = ~(m0 | m1)
    sm = torch.where(m0, 1 - pred.mean(dim=(1, 2)), torch.zeros_like(y))
    sm = torch.where(m1, pred.mean(dim=(1, 2)), sm)

    fg_n = gt.sum(dim=(1, 2)).clamp(min=1)
    fg_mean = (pred * gt).sum(dim=(1, 2)) / fg_n
    fg_var = (((pred - fg_mean[:, None, None]) ** 2) * gt).sum(dim=(1, 2)) / (fg_n - 1).clamp(min=1)
    s_fg = 2 * fg_mean / (fg_mean ** 2 + 1 + torch.sqrt(fg_var.clamp(min=0)) + EPS)
    bgm = 1 - gt
    p1 = 1 - pred
    bg_n = bgm.sum(dim=(1, 2)).clamp(min=1)
    bg_mean = (p1 * bgm).sum(dim=(1, 2)) / bg_n
    bg_var = (((p1 - bg_mean[:, None, None]) ** 2) * bgm).sum(dim=(1, 2)) / (bg_n - 1).clamp(min=1)
    s_bg = 2 * bg_mean / (bg_mean ** 2 + 1 + torch.sqrt(bg_var.clamp(min=0)) + EPS)
    obj = s_fg * y + s_bg * (1 - y)

    ysd = torch.arange(H, device=pred.device).double()
    xsd = torch.arange(W, device=pred.device).double()
    gtd = gt.double()
    gsum_d = gtd.sum(dim=(1, 2)).clamp(min=1)
    cy = (torch.round((gtd * ysd[None, :, None]).sum(dim=(1, 2)) / gsum_d).long() + 1)
    cx = (torch.round((gtd * xsd[None, None, :]).sum(dim=(1, 2)) / gsum_d).long() + 1)
    Y = torch.arange(H, device=pred.device).float()[None, :, None]
    X = torch.arange(W, device=pred.device).float()[None, None, :]
    cyb, cxb = cy[:, None, None].float(), cx[:, None, None].float()
    lt = (Y < cyb) & (X < cxb)
    rt = (Y < cyb) & (X >= cxb)
    lb = (Y >= cyb) & (X < cxb)
    rb = (Y >= cyb) & (X >= cxb)

    def bssim(mask):
        m = mask.float()
        N = m.sum(dim=(1, 2)).clamp(min=1)
        x = (pred * m).sum(dim=(1, 2)) / N
        yy = (gt * m).sum(dim=(1, 2)) / N
        d = (N - 1).clamp(min=1)
        sx = (((pred - x[:, None, None]) ** 2) * m).sum(dim=(1, 2)) / d
        sy = (((gt - yy[:, None, None]) ** 2) * m).sum(dim=(1, 2)) / d
        sxy = (((pred - x[:, None, None]) * (gt - yy[:, None, None])) * m).sum(dim=(1, 2)) / d
        al = 4 * x * yy * sxy
        be = (x ** 2 + yy ** 2) * (sx + sy)
        return torch.where(al != 0, al / (be + EPS),
                           torch.where(be == 0, torch.ones_like(al), torch.zeros_like(al)))

    cyf, cxf = cy.float(), cx.float()
    w_lt = cxf * cyf / area
    w_rt = cyf * (W - cxf) / area
    w_lb = (H - cyf) * cxf / area
    w_rb = 1 - w_lt - w_rt - w_lb
    reg = bssim(lt) * w_lt + bssim(rt) * w_rt + bssim(lb) * w_lb + bssim(rb) * w_rb

    sm_mm = (0.5 * obj + 0.5 * reg).clamp(min=0)
    return torch.where(mm, sm_mm, sm)


def torch_emeasure_avg(pred, gt):
    """Mean of the 256-threshold E-measure curve. Returns (B,)."""
    B, H, W = pred.shape
    gt = gt.float()
    gt_fg = gt.sum(dim=(1, 2))
    gt_size = float(H * W)
    p255 = (pred * 255).clamp(0, 255).long().view(B, -1)
    gtf = gt.view(B, -1)
    fg_fg = torch.zeros(B, 256, device=pred.device).scatter_add_(1, p255, gtf)
    fg_bg = torch.zeros(B, 256, device=pred.device).scatter_add_(1, p255, 1 - gtf)
    fg_fg_c = torch.flip(fg_fg, dims=[1]).cumsum(dim=1)
    fg_bg_c = torch.flip(fg_bg, dims=[1]).cumsum(dim=1)
    fg_n = fg_fg_c + fg_bg_c
    bg_n = gt_size - fg_n
    bg_fg = gt_fg[:, None] - fg_fg_c
    bg_bg = bg_n - bg_fg
    mean_p = fg_n / gt_size
    mean_g = (gt_fg / gt_size)[:, None]
    dpf, dpb = 1 - mean_p, -mean_p
    dgf, dgb = 1 - mean_g, -mean_g

    def enh(cp, cg):
        al = 2 * cp * cg / (cp ** 2 + cg ** 2 + EPS)
        return (al + 1) ** 2 / 4

    es = enh(dpf, dgf) * fg_fg_c + enh(dpf, dgb) * fg_bg_c + enh(dpb, dgf) * bg_fg + enh(dpb, dgb) * bg_bg
    curve = es / (gt_size - 1 + EPS)
    curve = torch.where((gt_fg == 0)[:, None], bg_n / (gt_size - 1 + EPS), curve)
    curve = torch.where((gt_fg == gt_size)[:, None], fg_n / (gt_size - 1 + EPS), curve)
    return curve.mean(dim=1)
