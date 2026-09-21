"""Build data/manifest.csv: one row per BRISC image with class, plane, official split,
image/mask paths (non-tumorous images get an empty mask) and mask area.

    python -m medplib_bt.datasets.manifest --config configs/dataset/brisc2025.yaml
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import _interpolate, apply_overrides, load_yaml
from .task import TaskSpec


def parse_name(stem: str):
    # brisc2025_<split>_<index>_<tumor>_<view>_<sequence>
    parts = stem.split("_")
    assert len(parts) == 6 and parts[0] == "brisc2025", stem
    return dict(official_split=parts[1], index=int(parts[2]), code=parts[3], plane=parts[4])


def scan(root: Path, spec: TaskSpec, mask_thresh: int):
    rows = []
    for split in ("train", "test"):
        for folder, key in spec.folders.items():
            files = sorted((root / "classification_task" / split / folder).glob("*.jpg"))
            assert files, f"no images in {root / 'classification_task' / split / folder}"
            for img in files:
                meta = parse_name(img.stem)
                assert meta["official_split"] == split and spec.codes[meta["code"]] == key, img
                mask = root / "segmentation_task" / split / "masks" / (img.stem + ".png")
                has_mask = mask.exists()
                assert has_mask == (key != "non_tumorous"), f"unexpected mask presence for {img}"
                im = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
                h, w = im.shape
                area = 0.0
                if has_mask:
                    m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                    assert m.shape == (h, w), f"mask/image size mismatch: {mask}"
                    area = float((m >= mask_thresh).mean())
                rows.append(dict(
                    image_id=img.stem, class_key=key, class_idx=spec.idx(key), plane=meta["plane"],
                    plane_name=spec.planes.get(meta["plane"], meta["plane"]), subset=split,
                    image_path=str(img.relative_to(root)), mask_path=str(mask.relative_to(root)) if has_mask else "",
                    mask_type="expert" if has_mask else "derived_empty", width=w, height=h, mask_area_ratio=area))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset/brisc2025.yaml")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = _interpolate(apply_overrides(load_yaml(args.config), args.set))
    spec = TaskSpec.from_cfg(cfg)
    df = pd.DataFrame(scan(Path(cfg["root"]), spec, int(cfg.get("mask_bin_threshold", 128))))
    out = Path(cfg["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = {
        "n_images": len(df),
        "by_subset_class": df.groupby(["subset", "class_key"]).size().unstack(0).to_dict(),
        "by_subset_plane": df.groupby(["subset", "plane"]).size().unstack(0).to_dict(),
        "masks": {"expert": int((df.mask_type == "expert").sum()), "derived_empty": int((df.mask_type == "derived_empty").sum())},
        "mask_area_ratio_expert": {k: float(v) for k, v in df[df.mask_type == "expert"].mask_area_ratio.describe().items()},
    }
    with open(out.with_name("manifest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
