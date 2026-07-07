# DisCOD: Disagreement-Guided Pseudo-Label Selection for Annotation-Free Camouflaged Object Detection

DisCOD trains a lightweight real-time camouflaged object detector **without any
pixel annotations**. A vision-language model (Qwen3-VL-4B) localizes the
camouflaged object with a bounding box, SAM2 converts the box into a
pseudo-mask, and a compact PVTv2 student is distilled on the pseudo-labels.
The key ingredient is **cross-prompt box disagreement**: querying the VLM with
K=5 prompts of distinct reasoning perspectives and measuring how much the
returned boxes disagree identifies the object-level localization failures that
dominate pseudo-label noise, so the worst samples can be removed before
distillation — all without ground truth. At inference time only the student
runs.

## Pipeline

1. **Box generation** — Qwen3-VL-4B outputs one bounding box per unlabeled
   training image; images with no parseable box are discarded.
2. **Mask generation** — SAM2 turns each box into a pseudo-mask.
3. **Disagreement scoring** — the VLM is queried with K=5 prompts (generic /
   texture / silhouette / anomaly / part-to-whole); each image is scored by
   `d_i = 1 − mean pairwise IoU` of the valid boxes (fewer than two valid
   boxes gives `d_i = 1`).
4. **Selection** — the 5% of samples with the highest `d_i` are dropped.
5. **Distillation** — the PVTv2 student is trained on the filtered
   pseudo-labels with a boundary-weighted structure loss.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

SAM2 is installed separately from <https://github.com/facebookresearch/sam2>
(needed only for pseudo-label generation and teacher evaluation; student
training and inference do not use it). Download the `sam2.1_hiera_large.pt`
checkpoint and pass its path via `--sam2-pt`.

## Data

Use the standard COD benchmark package (SINet layout):

```
data/cod/
├── TrainDataset/Imgs/          # CAMO-train + COD10K-train (4,040 images, no GT used)
├── TestDataset/{CAMO,CHAMELEON,COD10K}/{Imgs,GT}/
└── NC4K/{Imgs,GT}/
```

Only the unlabeled training images are used for pseudo-label generation;
ground-truth masks are used exclusively for test-set evaluation.

## Run

```bash
# 1. VLM boxes on the unlabeled training set
python scripts/generate_boxes.py --imgs-dir data/cod/TrainDataset/Imgs --out outputs/boxes.jsonl

# 2. SAM2 pseudo-masks + training manifest
python scripts/generate_masks.py --boxes outputs/boxes.jsonl --masks-dir outputs/masks \
  --out outputs/manifest.jsonl --sam2-pt /path/to/sam2.1_hiera_large.pt

# 3. Cross-prompt disagreement scores (K=5)
python scripts/compute_disagreement.py --manifest outputs/manifest.jsonl \
  --out outputs/disagreement.json --out-boxes outputs/prompt_boxes.json

# 4. Drop the top-5% highest-disagreement samples
python scripts/filter_manifest.py --manifest outputs/manifest.jsonl \
  --scores outputs/disagreement.json --out outputs/manifest_filtered.jsonl

# 5. Train the student (paper configuration; B0/B2/B4 differ only in backbone + batch size)
python scripts/train_student.py --manifest outputs/manifest_filtered.jsonl \
  --data-root data/cod --out outputs/ckpt \
  --backbone pvt_v2_b0 --bs 48 --amp --swa --strong-aug
# pvt_v2_b2: --bs 24    pvt_v2_b4: --bs 12
```

Evaluate a trained checkpoint (canonical CPU metrics, paper protocol):

```bash
python scripts/train_student.py --eval-ckpt outputs/ckpt/student_pvt_v2_b0_ep80.pth \
  --backbone pvt_v2_b0 --manifest outputs/manifest_filtered.jsonl --data-root data/cod --out outputs/ckpt
```

Zero-shot teacher evaluation and efficiency benchmark:

```bash
python scripts/eval_teacher.py --data-root data/cod --sam2-pt /path/to/sam2.1_hiera_large.pt
python scripts/benchmark.py
```

## Results

Single run, seed 0, 416×416, evaluated with the canonical PySODMetrics
protocol (predictions resized back to ground-truth resolution).

| Student | Params | FPS* | CAMO S<sub>m</sub>/E<sub>m</sub>/F<sup>w</sup><sub>β</sub>/MAE | COD10K | NC4K |
|---|---|---|---|---|---|
| PVTv2-B0 | 3.6M | 308 | .786 / .834 / .696 / .083 | .804 / .871 / .661 / .039 | .839 / .886 / .753 / .052 |
| PVTv2-B2 | 25M | 136 | .829 / .885 / .772 / .065 | .852 / .912 / .754 / .028 | .873 / .918 / .819 / .039 |
| PVTv2-B4 | 62M | 72 | .839 / .894 / .786 / .059 | .860 / .919 / .773 / .026 | .881 / .925 / .834 / .035 |

*bs=1, fp32, RTX 5090.

## Citation

The paper is under review; citation information will be added upon
publication.

## License

MIT (see [LICENSE](LICENSE)). Vendored third-party code is credited in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
