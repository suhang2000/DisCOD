"""Batched bounding-box inference with Qwen3-VL."""
from __future__ import annotations

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from .geometry import is_sane_bbox
from .parsing import extract_bbox


class BoxGenerator:
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", device="cuda:0"):
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, use_fast=True, max_pixels=589824, min_pixels=262144)
        self.processor.tokenizer.padding_side = "left"

    @torch.no_grad()
    def boxes(self, entries, system_prompt, user_prompt):
        """entries: iterable of (image_id, image_path).

        Returns {image_id: [x1, y1, x2, y2]} in image pixels; images whose
        response fails to parse into a sane box are omitted.
        """
        msgs, meta = [], []
        for iid, path in entries:
            im = Image.open(path).convert("RGB")
            meta.append((iid, im.size))
            msgs.append([
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": user_prompt}]},
            ])
        text = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in msgs]
        ii, vi = process_vision_info(msgs)
        inp = self.processor(text=text, images=ii, videos=vi, padding=True,
                             return_tensors="pt").to(self.device)
        gen = self.model.generate(**inp, do_sample=False, use_cache=True, max_new_tokens=64)
        trim = [o[len(i):] for i, o in zip(inp.input_ids, gen)]
        resps = self.processor.batch_decode(trim, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        out = {}
        for (iid, (W, H)), resp in zip(meta, resps):
            coords = extract_bbox(resp, img_wh=(W, H))
            if coords is not None and is_sane_bbox(coords):
                x1, y1, x2, y2 = coords
                out[iid] = [x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H]
        return out
