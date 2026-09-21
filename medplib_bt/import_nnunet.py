"""Convert nnU-Net v2 test predictions (--save_probabilities npz files) into the project's prediction
layout (predictions_raw.parquet + probs/<image_id>.png) so evaluate.py can score them.

    python -m medplib_bt.import_nnunet --pred-dir outputs/nnunet/pred_test --out-dir outputs/preds/nnunet_2d
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import load_config
from .datasets.brisc import load_manifest, load_mask
from .datasets.task import TaskSpec
from .engine.inference import prob_to_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infer/default.yaml")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    spec = TaskSpec.from_cfg(cfg["data"])
    root = Path(cfg["data"]["root"])
    out = Path(args.out_dir)
    (out / "probs").mkdir(parents=True, exist_ok=True)
    df = load_manifest(cfg["data"], "test")
    rows = []
    for r in df.itertuples():
        npz = Path(args.pred_dir) / f"{r.image_id}.npz"
        if npz.exists():
            prob = np.load(npz)["probabilities"]
            prob = np.squeeze(prob)            # (C, H, W) after removing nnU-Net's singleton depth axis
            prob = prob[1] if prob.ndim == 3 else prob
        else:                                  # fall back to the hard prediction png
            prob = (cv2.imread(str(Path(args.pred_dir) / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        img_shape = cv2.imread(str(root / r.image_path), cv2.IMREAD_GRAYSCALE).shape
        if prob.shape != img_shape:
            prob = cv2.resize(prob.astype(np.float32), (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out / "probs" / f"{r.image_id}.png"), prob_to_png(prob.astype(np.float32)))
        gt = load_mask(root / r.mask_path if isinstance(r.mask_path, str) and r.mask_path else None, img_shape + (1,))
        row = dict(image_id=r.image_id, class_key=r.class_key, class_idx=int(r.class_idx), plane=r.plane, subset=r.subset,
                   mask_type=r.mask_type, pred_idx=-1, pred_key="", iou_pred=float("nan"), gt_area_ratio=float(gt.mean()),
                   height=img_shape[0], width=img_shape[1])
        for k in spec.class_keys:
            row[f"ll_{k}"] = float("nan")
        rows.append(row)
    pd.DataFrame(rows).to_parquet(out / "predictions_raw.parquet", index=False)
    with open(out / "run_info.json", "w") as f:
        json.dump(dict(model={"type": "nnunet_2d"}, task="seg", subset="test", n_images=len(rows), pred_dir=args.pred_dir, argv=sys.argv), f, indent=2)
    print(f"imported {len(rows)} nnU-Net predictions -> {out}")


if __name__ == "__main__":
    main()
