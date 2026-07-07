"""Pseudo-label training dataset and COD benchmark layout."""
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

LAYOUT = {
    "CAMO": "TestDataset/CAMO",
    "CHAMELEON": "TestDataset/CHAMELEON",
    "COD10K": "TestDataset/COD10K",
    "NC4K": "NC4K",
}


class PseudoDataset(Dataset):
    def __init__(self, items, size, train, strong_aug=False, aug_hue=0.0, aug_cutout=0):
        self.items, self.size, self.train = items, size, train
        self.strong_aug = strong_aug
        self.aug_hue, self.aug_cutout = aug_hue, aug_cutout

    def _strong_aug(self, img, m):
        sz = self.size
        if random.random() < 0.7:
            s = random.uniform(0.8, 1.25)
            ns = max(8, int(round(sz * s)))
            im2 = cv2.resize(img, (ns, ns), interpolation=cv2.INTER_LINEAR)
            m2 = cv2.resize(m, (ns, ns), interpolation=cv2.INTER_NEAREST)
            if ns >= sz:
                y, x = random.randint(0, ns - sz), random.randint(0, ns - sz)
                img, m = im2[y:y + sz, x:x + sz], m2[y:y + sz, x:x + sz]
            else:
                img, m = np.zeros((sz, sz, 3), np.float32), np.zeros((sz, sz), np.float32)
                py, px = (sz - ns) // 2, (sz - ns) // 2
                img[py:py + ns, px:px + ns], m[py:py + ns, px:px + ns] = im2, m2
        if random.random() < 0.5:
            ang = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((sz / 2, sz / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (sz, sz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            m = cv2.warpAffine(m, M, (sz, sz), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        if random.random() < 0.5:
            img = img * random.uniform(0.8, 1.2)
            img = (img - img.mean()) * random.uniform(0.8, 1.2) + img.mean()
            img = np.clip(img, 0.0, 1.0)
        if self.aug_hue > 0 and random.random() < 0.7:
            shift = np.random.uniform(-self.aug_hue, self.aug_hue, 3).astype(np.float32)
            scale = np.random.uniform(1 - self.aug_hue, 1 + self.aug_hue, 3).astype(np.float32)
            img = np.clip(img * scale[None, None, :] + shift[None, None, :], 0.0, 1.0)
        if self.aug_cutout > 0 and random.random() < 0.5:
            for _ in range(random.randint(1, self.aug_cutout)):
                ch = random.randint(sz // 8, sz // 4)
                cy, cx = random.randint(0, sz - ch), random.randint(0, sz - ch)
                img[cy:cy + ch, cx:cx + ch] = float(img.mean())
        return np.ascontiguousarray(img, np.float32), np.ascontiguousarray((m > 0.5).astype(np.float32))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e = self.items[i]
        img = np.asarray(Image.open(e["img_path"]).convert("RGB").resize((self.size, self.size)), np.float32) / 255
        m = np.asarray(Image.open(e["mask_path"]).convert("L").resize((self.size, self.size)), np.float32)
        m = (m > 127).astype(np.float32)
        if self.train and random.random() < 0.5:
            img, m = img[:, ::-1].copy(), m[:, ::-1].copy()
        if self.strong_aug and self.train:
            img, m = self._strong_aug(img, m)
        img = (img - MEAN) / STD
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        m = torch.from_numpy(m)[None].float()
        return img, m
