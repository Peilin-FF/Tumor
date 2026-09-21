"""Collect summary_metrics.json of several inference runs into the report's result tables:
mean +- SD over seeds for the joint model, and the control table (zero-shot, classification-only,
segmentation-only, joint).

    python -m medplib_bt.aggregate --seeds outputs/preds/stageB_seed42 outputs/preds/stageB_seed123 outputs/preds/stageB_seed2026 \
        --zeroshot outputs/preds/zeroshot --cls-only outputs/preds/control_cls_seed42 --seg-only outputs/preds/control_seg_seed42 \
        --out outputs/results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(pred_dir):
    p = Path(pred_dir) / "eval" / "summary_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def flat(s):
    if s is None:
        return {}
    c = s.get("classification", {})
    out = dict(accuracy=c.get("accuracy"), macro_f1=c.get("macro_f1"), balanced_accuracy=c.get("balanced_accuracy"))
    seg = s.get("segmentation_positive")
    if seg:
        out.update(pos_dice=seg["dice"], pos_iou=seg["iou"], pos_sensitivity=seg["sensitivity"])
        for k, v in seg.get("by_class", {}).items():
            out[f"dice_{k}"] = v["dice"]
    alli = s.get("segmentation_all_images")
    if alli:  # all 1000 test images; a non-tumour scan scores 1 for an empty mask, 0 otherwise
        out.update(all_dice=alli["dice"], all_iou=alli["iou"])
    nt = s.get("non_tumor_masks")
    if nt:
        out["mask_fpr"] = nt["mask_fpr"]
    con = s.get("consistency")
    if con:
        out["conflict_rate"] = con["conflict_rate"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True, help="pred dirs of the joint model, one per seed")
    ap.add_argument("--zeroshot", default=None)
    ap.add_argument("--cls-only", default=None)
    ap.add_argument("--seg-only", default=None)
    ap.add_argument("--baseline", action="append", default=[], help="name=pred_dir of a conventional baseline (repeatable)")
    ap.add_argument("--variant", action="append", default=[], help="name=pred_dir of a MedPLIB variant, e.g. stage-B v2 (repeatable)")
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in args.seeds:
        f = flat(load(d))
        if f:
            rows.append(dict(run=Path(d).name, **f))
    seeds = pd.DataFrame(rows)
    metrics = [c for c in seeds.columns if c != "run"]
    summary = pd.DataFrame({"mean": seeds[metrics].mean(), "std": seeds[metrics].std(ddof=1) if len(seeds) > 1 else np.nan,
                            "n_seeds": len(seeds)})
    seeds.to_csv(out / "seeds.csv", index=False)
    summary.to_csv(out / "seeds_mean_sd.csv")

    controls = []
    for name, d in (("zero-shot MedPLIB", args.zeroshot), ("classification-only", args.cls_only),
                    ("segmentation-only", args.seg_only)):
        if d:
            f = flat(load(d))
            if name == "segmentation-only":  # its class scores are meaningless (single "<SEG>" answer)
                for k in ("accuracy", "macro_f1", "balanced_accuracy"):
                    f[k] = np.nan
            controls.append(dict(setting=name, **f))
    for spec_ in args.baseline + args.variant:
        name, d = spec_.split("=", 1)
        s = load(d)
        f = flat(s)
        info = {}
        p_info = Path(d) / "run_info.json"
        if p_info.exists():
            with open(p_info) as fh:
                info = json.load(fh)
        if info.get("task") == "seg":  # segmentation-only models have no class answer
            for k in ("accuracy", "macro_f1", "balanced_accuracy", "conflict_rate"):
                f[k] = np.nan
        controls.append(dict(setting=name, **f))
    controls.append(dict(setting="joint MedPLIB (mean over seeds)", **summary["mean"].to_dict()))
    ctrl = pd.DataFrame(controls)
    ctrl.to_csv(out / "controls.csv", index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("== per seed ==\n", seeds.to_string(index=False))
        print("\n== mean +- SD over seeds ==\n", summary.to_string())
        print("\n== controls ==\n", ctrl[["setting", "accuracy", "macro_f1", "pos_dice", "all_dice", "pos_iou", "mask_fpr", "conflict_rate"]].to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
