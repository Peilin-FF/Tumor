"""Run a registered baseline on the test split and write predictions in the same layout as
infer.py (predictions_raw.parquet, probs/<image_id>.png), so evaluate.py scores it identically.

    python -m medplib_bt.infer_baseline --config configs/train/baseline_unet.yaml --weights outputs/runs/unet_r34_seed42/final/model.pt --out-dir outputs/preds/unet_r34_seed42
    python -m medplib_bt.infer_baseline --config configs/infer/sam_med2d_box.yaml --out-dir outputs/preds/sam_med2d_oracle_box
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import paths
from .config import load_config
from .datasets import brisc  # noqa: F401
from .datasets.brisc import collate_tensor
from .datasets.task import TaskSpec
from .engine.inference import prob_to_png
from .models import medplib  # noqa: F401
from .registry import DATASETS, MODELS


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a config with model: and data: sections")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--weights", default=None, help="model.pt from train_baseline.py (not needed for prompted SAM)")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config, args.set)
    spec = TaskSpec.from_cfg(cfg["data"])
    device = torch.device("cuda", 0)
    out_dir = Path(args.out_dir)
    (out_dir / "probs").mkdir(parents=True, exist_ok=True)
    mtype = cfg["model"]["type"]
    task = cfg.get("task", "seg" if mtype in ("smp_unet", "sam_med2d_prompt") else "cls")
    size = int(cfg.get("size", 512))
    model = MODELS.build(cfg["model"], num_classes=4)
    if args.weights:
        ck = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        size = int(ck.get("size", size))
    model = model.to(device).eval()
    # seg baselines are scored on all test images (non-tumour ones give the false-mask rate)
    ds = DATASETS.build(cfg["data"], mode="cls", subset=args.subset, size=size, augment=False, keep_original=True,
                        max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=1 if mtype == "sam_med2d_prompt" else int(cfg.get("infer_batch", 16)), shuffle=False,
                        num_workers=int(cfg.get("workers", 8)), collate_fn=collate_tensor)
    rows, t0 = [], time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            n = len(batch["image_id"])
            if mtype == "sam_med2d_prompt":
                probs = [model.predict(batch["orig_image"][0], batch["orig_mask"][0], device)]
                logits = None
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch["image"].to(device)).float()
                if task == "seg":
                    probs = [F.interpolate(torch.sigmoid(out[i:i + 1]), batch["orig_size"][i], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                             for i in range(n)]
                    logits = None
                else:
                    probs, logits = None, out.cpu().numpy()
            for i in range(n):
                row = dict(image_id=batch["image_id"][i], class_key=batch["class_key"][i], class_idx=int(batch["class_idx"][i]),
                           plane=batch["plane"][i], subset=batch["subset"][i], mask_type=batch["mask_type"][i],
                           gt_area_ratio=float(batch["orig_mask"][i].mean()), height=batch["orig_size"][i][0], width=batch["orig_size"][i][1],
                           iou_pred=float("nan"))
                if logits is not None:
                    row["pred_idx"] = int(logits[i].argmax()); row["pred_key"] = spec.class_keys[row["pred_idx"]]
                    for c, k in enumerate(spec.class_keys):
                        row[f"ll_{k}"] = float(logits[i][c])
                else:
                    row["pred_idx"], row["pred_key"] = -1, ""
                    for k in spec.class_keys:
                        row[f"ll_{k}"] = float("nan")
                    cv2.imwrite(str(out_dir / "probs" / f"{batch['image_id'][i]}.png"), prob_to_png(probs[i]))
                rows.append(row)
            if (bi + 1) % 20 == 0:
                print(f"[infer_baseline] {bi+1}/{len(loader)} batches", flush=True)
    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    df.to_parquet(out_dir / "predictions_raw.parquet", index=False)
    info = dict(model=cfg["model"], task=task, size=size, weights=args.weights, subset=args.subset, n_images=len(df),
                minutes=(time.time() - t0) / 60, config=cfg.get("_file"), argv=sys.argv,
                note="prompted SAM uses boxes/points derived from the expert masks (oracle prompts)" if mtype == "sam_med2d_prompt" else None)
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[infer_baseline] {len(df)} images in {info['minutes']:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
