"""Evaluate an inference run (CPU): classification (accuracy, macro-F1, balanced accuracy,
per-class precision/recall/F1, confusion matrix), segmentation on tumour images (Dice, IoU,
sensitivity, specificity; by class, plane and lesion size), non-tumour false-positive masks
(mask FPR, foreground area), class/mask consistency, bootstrap 95 % CIs, and figures.

    python -m medplib_bt.evaluate --pred-dir outputs/preds/stageB_seed42
Outputs in <pred-dir>/eval: summary_metrics.json, predictions.parquet, confusion_matrix.png,
dice_by_class.png, qualitative_grid.png, failure_cases.html, masks/<image_id>.png
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import load_config
from .datasets.brisc import load_image_rgb, load_mask
from .datasets.task import TaskSpec
from .engine.inference import mask_metrics, png_to_prob, postprocess_mask, softmax_np


def bootstrap_ci(values, fn, n_boot, seed, **kw):
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = [fn(values[rng.integers(0, n, n)], **kw) for _ in range(n_boot)] if n > 0 else [float("nan")]
    return [float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))]


def classification_block(df, spec, n_boot, seed):
    y, p = df.class_idx.values, df.pred_idx.values
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=range(len(spec.class_keys)), zero_division=0)
    idx = np.arange(len(df))
    return dict(
        n=int(len(df)), accuracy=float((y == p).mean()), macro_f1=float(f1_score(y, p, average="macro")),
        balanced_accuracy=float(balanced_accuracy_score(y, p)),
        accuracy_ci95=bootstrap_ci(idx, lambda s: float((y[s] == p[s]).mean()), n_boot, seed),
        macro_f1_ci95=bootstrap_ci(idx, lambda s: float(f1_score(y[s], p[s], average="macro")), n_boot, seed),
        per_class={k: dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]), support=int(sup[i])) for i, k in enumerate(spec.class_keys)},
        confusion_matrix=confusion_matrix(y, p, labels=range(len(spec.class_keys))).tolist(),
    )


def seg_block(pos, n_boot, seed):
    out = dict(n=int(len(pos)))
    for m in ("dice", "iou", "sensitivity", "specificity"):
        v = pos[m].values.astype(float)
        out[m] = float(np.nanmean(v)) if len(v) else float("nan")
        out[m + "_std"] = float(np.nanstd(v)) if len(v) else float("nan")
        out[m + "_ci95"] = bootstrap_ci(v, lambda s: float(np.nanmean(s)), n_boot, seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="output dir of infer.py")
    ap.add_argument("--config", default="configs/infer/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    spec = TaskSpec.from_cfg(cfg["data"])
    root = Path(cfg["data"]["root"])
    thr = float(cfg.get("seg_threshold", 0.5))
    min_cc = float(cfg.get("min_component_ratio", 0.0))
    ev = cfg.get("evaluation", {})
    n_boot, seed = int(ev.get("bootstrap", 1000)), int(ev.get("seed", 42))
    bins = list(ev.get("lesion_size_bins", [0.0, 0.005, 0.015, 1.0]))

    pred_dir = Path(args.pred_dir)
    out_dir = pred_dir / "eval"
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(pred_dir / "predictions_raw.parquet")
    man = pd.read_csv(cfg["data"]["manifest"]).set_index("image_id")
    ll = df[[f"ll_{k}" for k in spec.class_keys]].values
    probs = softmax_np(ll)
    df["confidence"] = probs.max(1)
    df["correct"] = df.pred_idx == df.class_idx

    # masks
    has_probs = (pred_dir / "probs").exists() and any((pred_dir / "probs").iterdir())
    recs = []
    for r in df.itertuples():
        rec = dict(image_id=r.image_id)
        if has_probs:
            prob = png_to_prob(pred_dir / "probs" / f"{r.image_id}.png")
            mrow = man.loc[r.image_id]
            gt = load_mask(root / mrow.mask_path if isinstance(mrow.mask_path, str) and mrow.mask_path else None,
                           prob.shape + (1,), int(cfg["data"].get("mask_bin_threshold", 128)))
            pred = postprocess_mask(prob, thr, min_cc)
            cv2.imwrite(str(out_dir / "masks" / f"{r.image_id}.png"), pred * 255)
            mm = mask_metrics(pred, gt)
            mm.pop("gt_area_ratio")  # already in predictions_raw
            rec.update(mm)
            rec["pred_mask_nonempty"] = bool(pred.any())
        recs.append(rec)
    df = df.merge(pd.DataFrame(recs), on="image_id")
    if has_probs:
        nt = spec.class_keys.index("non_tumorous")
        df["consistency_flag"] = ((df.pred_idx == nt) & df.pred_mask_nonempty) | ((df.pred_idx != nt) & ~df.pred_mask_nonempty)
        pos = df[df.mask_type == "expert"].copy()
        pos["lesion_size"] = pd.cut(pos.gt_area_ratio, bins=bins, labels=["small", "medium", "large"][:len(bins) - 1], include_lowest=True)
        neg = df[df.mask_type != "expert"]

    summary = dict(pred_dir=str(pred_dir), n_images=int(len(df)), seg_threshold=thr, min_component_ratio=min_cc,
                   classification=classification_block(df, spec, n_boot, seed))
    if has_probs:
        summary["segmentation_positive"] = seg_block(pos, n_boot, seed)
        summary["segmentation_positive"]["by_class"] = {k: seg_block(pos[pos.class_key == k], 200, seed) for k in spec.class_keys if (pos.class_key == k).any()}
        summary["segmentation_positive"]["by_plane"] = {k: seg_block(pos[pos.plane == k], 200, seed) for k in sorted(pos.plane.unique())}
        summary["segmentation_positive"]["by_lesion_size"] = {str(k): seg_block(pos[pos.lesion_size == k], 200, seed) for k in pos.lesion_size.cat.categories}
        # whole test set: a non-tumour image scores Dice/IoU 1 for an empty mask and 0 otherwise
        summary["segmentation_all_images"] = dict(n=int(len(df)), dice=float(df.dice.mean()), iou=float(df.iou.mean()),
                                                  dice_ci95=bootstrap_ci(df.dice.values.astype(float), lambda s: float(np.mean(s)), n_boot, seed))
        summary["non_tumor_masks"] = dict(
            n=int(len(neg)), mask_fpr=float(neg.pred_mask_nonempty.mean()) if len(neg) else float("nan"),
            mask_fpr_ci95=bootstrap_ci(neg.pred_mask_nonempty.values.astype(float), lambda s: float(s.mean()), n_boot, seed),
            mean_foreground_ratio=float(neg.pred_area_ratio.mean()) if len(neg) else float("nan"),
            p95_foreground_ratio=float(neg.pred_area_ratio.quantile(0.95)) if len(neg) else float("nan"))
        summary["consistency"] = dict(conflict_rate=float(df.consistency_flag.mean()),
                                      non_tumor_pred_with_mask=int(((df.pred_idx == nt) & df.pred_mask_nonempty).sum()),
                                      tumor_pred_without_mask=int(((df.pred_idx != nt) & ~df.pred_mask_nonempty).sum()))
    df.to_parquet(out_dir / "predictions.parquet", index=False)
    with open(out_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    c = summary["classification"]
    line = f"[eval] n={c['n']} acc {c['accuracy']:.4f} macro-F1 {c['macro_f1']:.4f} bal-acc {c['balanced_accuracy']:.4f}"
    if has_probs:
        s = summary["segmentation_positive"]
        line += f" | pos Dice {s['dice']:.4f} IoU {s['iou']:.4f} | mask-FPR {summary['non_tumor_masks']['mask_fpr']:.4f} | conflicts {summary['consistency']['conflict_rate']:.4f}"
    print(line)
    if not args.no_figures:
        make_figures(df, summary, spec, root, out_dir, has_probs, man)
    print("->", out_dir)


def make_figures(df, summary, spec, root, out_dir, has_probs, man):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(summary["classification"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(spec.class_keys)), spec.class_keys, rotation=30, ha="right")
    ax.set_yticks(range(len(spec.class_keys)), spec.class_keys)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    if not has_probs:
        return
    pos = df[df.mask_type == "expert"]
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = [k for k in spec.class_keys if (pos.class_key == k).any()]
    ax.boxplot([pos[pos.class_key == k].dice.values for k in keys], labels=keys)
    ax.set_ylabel("Dice (tumour images)")
    fig.tight_layout()
    fig.savefig(out_dir / "dice_by_class.png", dpi=150)
    plt.close(fig)

    # qualitative grid: best / median / worst Dice per class (image, GT, prediction, overlay)
    picks = []
    for k in keys:
        sub = pos[pos.class_key == k].sort_values("dice")
        for name, row in (("worst", sub.iloc[0]), ("median", sub.iloc[len(sub) // 2]), ("best", sub.iloc[-1])):
            picks.append((k, name, row))
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)))
    for r, (k, name, row) in enumerate(picks):
        mrow = man.loc[row.image_id]
        img = load_image_rgb(root / mrow.image_path)
        gt = load_mask(root / mrow.mask_path, img.shape)
        pred = cv2.imread(str(out_dir / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        over = img.copy()
        over[gt > 0] = (0.5 * over[gt > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        over[pred] = (0.5 * over[pred] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        for c, (title, im_) in enumerate((("image", img), ("expert mask", gt * 255), ("prediction", pred.astype(np.uint8) * 255), ("overlay (green GT / red pred)", over))):
            axes[r, c].imshow(im_, cmap=None if im_.ndim == 3 else "gray")
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(f"{k} {name}: {row.image_id}\npred {row.pred_key} conf {row.confidence:.2f} Dice {row.dice:.3f}", fontsize=8, loc="left")
            else:
                axes[r, c].set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative_grid.png", dpi=110)
    plt.close(fig)

    # failure cases
    fails = df[(~df.correct) | (df.consistency_flag) | ((df.mask_type == "expert") & (df.dice < 0.5))].copy()
    fails["failure_type"] = np.where(~fails.correct, "misclassified",
                                     np.where(fails.consistency_flag, "class/mask conflict", "low Dice"))
    cols = ["image_id", "class_key", "pred_key", "confidence", "dice", "iou", "pred_area_ratio", "gt_area_ratio", "failure_type"]
    html = ["<html><body><h2>Failure cases</h2>", f"<p>{len(fails)} of {len(df)} images</p>",
            fails[cols].round(4).to_html(index=False), "</body></html>"]
    with open(out_dir / "failure_cases.html", "w") as f:
        f.write("\n".join(html))


if __name__ == "__main__":
    main()
