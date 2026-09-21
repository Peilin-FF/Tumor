"""Decision-level comparison on the same test images: does a system's class answer agree with its
mask, does it stay silent on non-tumour scans, and how often are both answers right at once?
A single model is read from eval/predictions.parquet. A pipeline pairs a classifier's answer with a
segmenter's mask; "gated" drops the mask whenever the classifier says non-tumorous.

    python -m medplib_bt.consistency --model "MedPLIB v2=outputs/preds/stageB_v2_seed42" \
        --pipeline "EfficientNet-B0 + U-Net=outputs/preds/effnet_b0_seed42,outputs/preds/unet_r34_seed42" \
        --gated "EfficientNet-B0 gate + U-Net=outputs/preds/effnet_b0_seed42,outputs/preds/unet_r34_seed42" --out outputs/results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NEG = "non_tumorous"
COLS = ["image_id", "class_key", "mask_type", "plane"]


def load(pred_dir):
    d = Path(pred_dir)
    p = pd.read_parquet(d / "eval" / "predictions.parquet")
    info = {}
    if (d / "run_info.json").exists():
        with open(d / "run_info.json") as f:
            info = json.load(f)
    has_class = not (info.get("seg_only") or info.get("task") == "seg")
    has_mask = "dice" in p.columns
    return p, has_class, has_mask


def decisions(cls_p, seg_p, gated=False):
    """Per-image decision table. cls_p / seg_p: prediction frames (either may be None)."""
    base = (seg_p if seg_p is not None else cls_p)[COLS].copy()
    base["tumour"] = base.mask_type == "expert"
    if cls_p is not None:
        base = base.merge(cls_p[["image_id", "pred_key"]], on="image_id")
    else:
        base["pred_key"] = None
    if seg_p is not None:
        base = base.merge(seg_p[["image_id", "pred_mask_nonempty", "dice", "pred_area_ratio"]], on="image_id")
    else:
        base["pred_mask_nonempty"], base["dice"], base["pred_area_ratio"] = np.nan, np.nan, np.nan
    if gated:  # classifier says non-tumorous -> empty mask (Dice 1 on a healthy scan, 0 on a tumour)
        g = base.pred_key == NEG
        base.loc[g, "pred_mask_nonempty"] = False
        base.loc[g, "pred_area_ratio"] = 0.0
        base.loc[g, "dice"] = np.where(base.loc[g, "tumour"], 0.0, 1.0)
    has_class, has_mask = cls_p is not None, seg_p is not None
    base["class_correct"] = (base.pred_key == base.class_key) if has_class else np.nan
    if has_mask:
        base["mask_correct"] = np.where(base.tumour, base.dice >= 0.5, ~base.pred_mask_nonempty.astype(bool))
        base["healthy_false_mask"] = ~base.tumour & base.pred_mask_nonempty.astype(bool)
        base["tumour_missed"] = base.tumour & ~base.pred_mask_nonempty.astype(bool)
    else:
        base["mask_correct"] = base["healthy_false_mask"] = base["tumour_missed"] = np.nan
    if has_class and has_mask:
        says_tumour = base.pred_key != NEG
        base["conflict"] = (says_tumour & ~base.pred_mask_nonempty.astype(bool)) | (~says_tumour & base.pred_mask_nonempty.astype(bool))
        base["both_correct"] = base.class_correct.astype(bool) & base.mask_correct.astype(bool)
    else:
        base["conflict"] = base["both_correct"] = np.nan
    return base


def ci(x, rng, n_boot=1000):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.isnan(x).all():
        return [np.nan, np.nan]
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarise(d, rng):
    t, h = d[d.tumour], d[~d.tumour]
    r = dict(n=len(d), class_accuracy=d.class_correct.mean(), conflict_rate=d.conflict.mean(),
             both_correct_rate=d.both_correct.mean(), mask_decision_rate=d.mask_correct.mean(),
             healthy_false_mask_rate=h.healthy_false_mask.mean(), tumour_missed_rate=t.tumour_missed.mean(),
             tumour_dice=t.dice.mean(), all_dice=d.dice.mean())
    r["conflict_ci95"] = ci(d.conflict, rng)
    r["both_correct_ci95"] = ci(d.both_correct, rng)
    r["healthy_false_mask_ci95"] = ci(h.healthy_false_mask, rng)
    r["all_dice_ci95"] = ci(d.dice, rng)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=pred_dir (repeatable)")
    ap.add_argument("--pipeline", action="append", default=[], help="name=classifier_dir,segmenter_dir")
    ap.add_argument("--gated", action="append", default=[], help="name=classifier_dir,segmenter_dir; mask dropped when the class is non-tumorous")
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    rows, per_image = [], []
    for spec in args.model:
        name, d = spec.split("=", 1)
        p, has_class, has_mask = load(d)
        dec = decisions(p if has_class else None, p if has_mask else None)
        rows.append(dict(system=name, kind="single model", **summarise(dec, rng)))
        per_image.append(dec.assign(system=name))
    for gated, specs in ((False, args.pipeline), (True, args.gated)):
        for spec in specs:
            name, dirs = spec.split("=", 1)
            cdir, sdir = dirs.split(",")
            cp, _, _ = load(cdir)
            sp, _, _ = load(sdir)
            dec = decisions(cp, sp, gated=gated)
            rows.append(dict(system=name, kind="gated pipeline" if gated else "pipeline", **summarise(dec, rng)))
            per_image.append(dec.assign(system=name))
    df = pd.DataFrame(rows)
    df.to_csv(out / "consistency.csv", index=False)
    with open(out / "consistency.json", "w") as f:
        json.dump(rows, f, indent=2)
    pd.concat(per_image, ignore_index=True).to_parquet(out / "consistency_per_image.parquet", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(df[["system", "kind", "class_accuracy", "conflict_rate", "healthy_false_mask_rate", "tumour_missed_rate",
                  "tumour_dice", "all_dice", "mask_decision_rate", "both_correct_rate"]].to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
