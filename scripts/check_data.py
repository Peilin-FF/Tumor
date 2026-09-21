#!/usr/bin/env python3
"""CPU sanity check of the data pipeline: builds a few training and inference samples with the
real tokenizer/CLIP processor (no LLM), prints shapes and the decoded prompt/answer/label
masking, and writes augmentation previews to outputs/data_check/."""
import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medplib_bt import paths  # noqa: E402
from medplib_bt.config import load_config  # noqa: E402
from medplib_bt.datasets import brisc  # noqa: E402,F401
from medplib_bt.datasets.brisc import collate_infer, collate_train  # noqa: E402
from medplib_bt.models.medplib import load_tokenizer, set_conversation_template  # noqa: E402
from medplib_bt.registry import DATASETS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train/stage_b.yaml")
args = ap.parse_args()
cfg = load_config(args.config)
from transformers import CLIPImageProcessor
tok = load_tokenizer(cfg["model"]["checkpoint"], cfg["model"]["model_max_length"])
set_conversation_template(cfg["model"].get("conv_template", "llava_v1"))
proc = CLIPImageProcessor.from_pretrained(cfg["model"]["clip_dir"])
data_args = types.SimpleNamespace(image_processor=proc, image_aspect_ratio="pad", is_multimodal=True, mm_use_im_start_end=True)
print("vocab", len(tok), "<SEG> id", tok("<SEG>", add_special_tokens=False).input_ids, "pad", tok.pad_token_id, "eos", tok.eos_token_id)

train_ds = DATASETS.build(cfg["data"], mode="train", subset="train", tokenizer=tok, data_args=data_args, model_cfg=cfg["model"], seed=42)
print("train images", len(train_ds))
out = paths.get("outputs_root") / "data_check"
out.mkdir(parents=True, exist_ok=True)
items = []
for i in [0, 1, 2, 3, 4000, 4500]:
    d = train_ds[i]
    items.append(d)
    ids = d["input_ids"]
    lab = d["labels"]
    sup = ids[lab != -100]
    print(f"\n[{i}] {d['image_id']} class={d['class_key']} task={d['task']} sam={tuple(d['image_sam'].shape)} clip={tuple(d['image_clip'].shape)} "
          f"ids={len(ids)} masks={len(d['masks'])} " + (f"mask_sum={int(d['masks'][0].sum())} resize={d['resize'][0]}" if d["masks"] else ""))
    print("   prompt/answer:", d["question"], "||", d["gt"])
    print("   supervised tokens:", [tok.convert_ids_to_tokens(int(t)) for t in sup])
    # augmentation preview
    img_sam = d["image_sam"]
    img = (img_sam * train_ds.pixel_std + train_ds.pixel_mean).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    panel = img.copy()
    if d["masks"]:
        m = cv2.resize(d["masks"][0].numpy().astype(np.uint8), None, fx=0, fy=0, dsize=(256, 256), interpolation=cv2.INTER_NEAREST) if False else None
    cv2.imwrite(str(out / f"train_{i}_{d['task']}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
batch = collate_train(items[:3])
print("\ncollated train batch:", {k: (tuple(v.shape) if torch.is_tensor(v) else (len(v) if isinstance(v, list) else v)) for k, v in batch.items()
                                  if k in ("images", "images_clip", "input_ids", "labels", "attention_mask", "masks_list", "label_list", "resize_list", "seg_flag", "valid_mask_bool", "offset")})

inf_ds = DATASETS.build(cfg["data"], mode="infer", subset="test", tokenizer=tok, data_args=data_args, model_cfg=cfg["model"], prompt="joint", with_seg=True)
d = inf_ds[0]
print("\ninfer sample", d["image_id"], d["class_key"], "candidates:")
for ids, lab, sm in zip(d["input_ids"], d["labels"], d["score_masks"]):
    print("   scored:", [tok.convert_ids_to_tokens(int(t)) for t in ids[sm]], "| supervised:", [tok.convert_ids_to_tokens(int(t)) for t in ids[lab != -100]])
b = collate_infer([inf_ds[0], inf_ds[1]])
print("collated infer:", {k: (tuple(v.shape) if torch.is_tensor(v) else (len(v) if isinstance(v, list) else v)) for k, v in b.items() if k != "meta"})
print("OK")
