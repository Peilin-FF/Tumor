"""Train a registered conventional baseline on BRISC (single GPU, bf16 autocast).

task "seg": U-Net style model, BCE + Dice loss on the tumour images.
task "cls": image classifier, cross-entropy on all images.
Writes outputs/runs/<exp>/{final/model.pt, history.json, config.yaml}.

    python -m medplib_bt.train_baseline --config configs/train/baseline_unet.yaml --exp-name unet_r34_seed42
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import paths
from .config import dump, load_config
from .datasets import brisc  # noqa: F401
from .datasets.brisc import collate_tensor
from .models import medplib  # noqa: F401
from .registry import DATASETS, MODELS


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--outputs-root", default=str(paths.get("outputs_root")))
    return ap.parse_args(argv)


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config, args.set)
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda", 0)
    task = cfg["task"]
    exp_dir = Path(args.outputs_root) / "runs" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w") as f:
        f.write(dump(dict(cfg, argv=sys.argv)))

    size = int(cfg.get("size", 512))
    ds = DATASETS.build(cfg["data"], mode=task, subset="train", seed=seed, augment=cfg.get("augment"), size=size,
                        max_samples=int(cfg.get("max_train_samples", 0)))
    loader = DataLoader(ds, batch_size=int(cfg["batch"]), shuffle=True, num_workers=int(cfg.get("workers", 8)),
                        collate_fn=collate_tensor, drop_last=True, pin_memory=True)
    model = MODELS.build(cfg["model"], num_classes=4).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[baseline] {task} model {cfg['model']['type']} {n_par/1e6:.1f}M params, {len(ds)} train images, {len(loader)} steps/epoch", flush=True)

    o = cfg["optim"]
    epochs = int(cfg["epochs"])
    optim = torch.optim.AdamW(model.parameters(), lr=float(o["lr"]), weight_decay=float(o.get("weight_decay", 1e-4)))
    total = epochs * len(loader)
    warm = int(float(o.get("warmup_ratio", 0.03)) * total)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    w_bce, w_dice = float(cfg.get("loss", {}).get("bce", 1.0)), float(cfg.get("loss", {}).get("dice", 1.0))
    history = []
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        t0, tot, n = time.time(), {}, 0
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            out = out.float()
            if task == "seg":
                y = batch["mask"].to(device)[:, None]
                l_bce = F.binary_cross_entropy_with_logits(out, y)
                l_dice = dice_loss(out, y)
                loss = w_bce * l_bce + w_dice * l_dice
                parts = dict(bce=l_bce.item(), dice=l_dice.item())
            else:
                y = batch["class_idx"].to(device)
                loss = F.cross_entropy(out, y, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
                parts = dict(acc=(out.argmax(1) == y).float().mean().item())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            parts["loss"] = loss.item()
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v
            n += 1
        rec = dict(epoch=epoch, minutes=(time.time() - t0) / 60, lr=sched.get_last_lr()[0], **{k: v / n for k, v in tot.items()})
        history.append(rec)
        print("[epoch %d] " % epoch + " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k not in ("epoch",)), flush=True)
        with open(exp_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    (exp_dir / "final").mkdir(exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), model_cfg=cfg["model"], task=task, size=size), exp_dir / "final" / "model.pt")
    print(f"[baseline] finished -> {exp_dir / 'final' / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
