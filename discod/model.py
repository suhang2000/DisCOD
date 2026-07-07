"""PVTv2 student: timm backbone + top-down FPN decoder, and the training loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, chans, mid=64):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, mid, 1) for c in chans])
        self.smooth = nn.ModuleList([nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(True)) for _ in chans])
        self.head = nn.Conv2d(mid, 1, 1)

    def forward(self, feats):
        p = self.smooth[-1](self.lat[-1](feats[-1]))
        for i in range(len(feats) - 2, -1, -1):
            p = F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p + self.lat[i](feats[i]))
        return self.head(p)


class Student(nn.Module):
    def __init__(self, backbone="pvt_v2_b0", pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        self.decoder = Decoder(self.backbone.feature_info.channels())

    def forward(self, x):
        logit = self.decoder(self.backbone(x))
        return F.interpolate(logit, size=x.shape[-2:], mode="bilinear", align_corners=False)


def structure_loss(pred, mask, bnd_weight=5.0, bnd_pool=31):
    """Boundary-weighted BCE + IoU loss (PraNet / F3Net style)."""
    pad = bnd_pool // 2
    weit = 1 + bnd_weight * torch.abs(F.avg_pool2d(mask, bnd_pool, 1, pad) - mask)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * bce).sum(dim=(1, 2, 3)) / (weit.sum(dim=(1, 2, 3)) + 1e-6)
    p = torch.sigmoid(pred)
    inter = ((p * mask) * weit).sum(dim=(1, 2, 3))
    union = ((p + mask) * weit).sum(dim=(1, 2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
