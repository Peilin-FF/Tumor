"""Export BRISC to nnU-Net v2 layout: Dataset<ID>_BRISC/{imagesTr,labelsTr,imagesTs,dataset.json}.
Training uses the tumour images of the official train split; imagesTs holds all test images.

    python -m medplib_bt.datasets.export_nnunet --raw-dir outputs/nnunet/raw --dataset-id 501
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import load_config
from .brisc import load_manifest, load_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infer/default.yaml")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--dataset-id", type=int, default=501)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    out = Path(args.raw_dir) / f"Dataset{args.dataset_id:03d}_BRISC"
    for d in ("imagesTr", "labelsTr", "imagesTs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    df = load_manifest(cfg["data"], "all")
    n_tr = 0
    for r in df.itertuples():
        img = cv2.imread(str(root / r.image_path), cv2.IMREAD_GRAYSCALE)
        if r.subset == "train":
            if r.mask_type != "expert":
                continue
            cv2.imwrite(str(out / "imagesTr" / f"{r.image_id}_0000.png"), img)
            m = load_mask(root / r.mask_path, img.shape + (1,), int(cfg["data"].get("mask_bin_threshold", 128)))
            cv2.imwrite(str(out / "labelsTr" / f"{r.image_id}.png"), m.astype(np.uint8))
            n_tr += 1
        else:
            cv2.imwrite(str(out / "imagesTs" / f"{r.image_id}_0000.png"), img)
    with open(out / "dataset.json", "w") as f:
        json.dump({"channel_names": {"0": "T1"}, "labels": {"background": 0, "tumor": 1}, "numTraining": n_tr, "file_ending": ".png",
                   "name": "BRISC2025", "description": "BRISC 2025 tumour images (official train split) and all official test images"}, f, indent=2)
    print(f"exported {n_tr} training pairs and {int((df.subset == 'test').sum())} test images -> {out}")


if __name__ == "__main__":
    main()
