"""Run inference on a split and write raw predictions: class log-likelihoods per image plus a
full-resolution tumour probability map (probs/<image_id>.png). Thresholding and metrics are
done by evaluate.py. One process per GPU, images sharded across ranks.

    deepspeed --include=localhost:0,1,2,3 --master_port 29611 --module medplib_bt.infer \
        --adapter outputs/runs/stageB_seed42/final --out-dir outputs/preds/stageB_seed42
    deepspeed ... --module medplib_bt.infer --no-adapter --out-dir outputs/preds/zeroshot     (pretrained model)
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import cv2
import deepspeed
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from . import paths
from .config import load_config
from .datasets import brisc  # noqa: F401
from .datasets.brisc import collate_infer, load_manifest
from .datasets.task import TaskSpec
from .engine.inference import prob_to_png, run_candidates
from .models import medplib as M
from .registry import DATASETS, MODELS

paths.add_medplib_to_syspath()
from deepspeed.moe.layer import MoE  # noqa: E402


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    ap.add_argument("--config", default="configs/infer/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--adapter", default=None, help="adapter dir (trainable.pt); omit with --no-adapter for the pretrained model")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--subset", default="test", choices=["train", "test", "all"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config, args.set)
    deepspeed.init_distributed(dist_backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)
    dtype = M.DTYPES[cfg["model"].get("precision", "bf16")]
    spec = TaskSpec.from_cfg(cfg["data"])
    out_dir = Path(args.out_dir)
    if rank == 0:
        (out_dir / "probs").mkdir(parents=True, exist_ok=True)
    dist.barrier()

    tokenizer, model = MODELS.build(cfg["model"], dtype=dtype)
    if not args.no_adapter:
        assert args.adapter, "--adapter or --no-adapter"
        model = M.apply_adapter(model, args.adapter)
    model.to(device).eval()
    for m in model.modules():          # expert-parallel groups of size 1: MoE all-to-all is a local no-op
        if isinstance(m, MoE):
            m.set_deepspeed_parallelism()

    df = load_manifest(cfg["data"], args.subset)
    if args.max_samples:
        df = df.iloc[:args.max_samples]
    vision_tower = M.unwrap(model).get_model().get_vision_tower()
    data_args = types.SimpleNamespace(image_processor=vision_tower.image_processor, image_aspect_ratio="pad",
                                      is_multimodal=True, mm_use_im_start_end=True)
    ds = DATASETS.build(cfg["data"], mode="infer", subset=args.subset, tokenizer=tokenizer, data_args=data_args,
                        model_cfg=cfg["model"], prompt=cfg.get("prompt", "joint"), with_seg=bool(cfg.get("with_seg", True)),
                        seg_only=bool(cfg.get("seg_only", False)), df=df.iloc[rank::world].reset_index(drop=True))
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 4)), shuffle=False, num_workers=int(cfg.get("workers", 4)),
                        collate_fn=collate_infer)

    rows, t0 = [], time.time()
    for bi, batch in enumerate(loader):
        for r in run_candidates(model, tokenizer, batch, device, spec.class_keys, dtype, decode_masks=bool(cfg.get("with_seg", True))):
            meta = r["meta"]
            row = dict(image_id=meta["image_id"], class_key=meta["class_key"], class_idx=meta["class_idx"], plane=meta["plane"],
                       subset=meta["subset"], mask_type=meta["mask_type"], pred_idx=r["pred_idx"], pred_key=r["pred_key"],
                       iou_pred=r["iou_pred"], gt_area_ratio=float(r["gt_mask"].float().mean()),
                       height=meta["orig_size"][0], width=meta["orig_size"][1])
            for c, k in enumerate(spec.class_keys):  # seg_only runs have a single candidate
                row[f"ll_{k}"] = float(r["ll"][c]) if c < len(r["ll"]) else float("nan")
            if r["mask_logits"] is not None:
                cv2.imwrite(str(out_dir / "probs" / f"{meta['image_id']}.png"), prob_to_png(torch.sigmoid(r["mask_logits"]).numpy()))
            rows.append(row)
        if rank == 0 and (bi + 1) % 10 == 0:
            print(f"[infer] {bi+1}/{len(loader)} batches, {(time.time()-t0)/(bi+1):.2f}s/batch", flush=True)
    pd.DataFrame(rows).to_parquet(out_dir / f"_part_rank{rank}.parquet", index=False)
    dist.barrier()
    if rank == 0:
        full = pd.concat([pd.read_parquet(out_dir / f"_part_rank{r}.parquet") for r in range(world)], ignore_index=True)
        full = full.sort_values("image_id").reset_index(drop=True)
        assert full.image_id.is_unique and len(full) == len(df), (len(full), len(df))
        full.to_parquet(out_dir / "predictions_raw.parquet", index=False)
        for r in range(world):
            (out_dir / f"_part_rank{r}.parquet").unlink()
        info = dict(adapter=None if args.no_adapter else str(Path(args.adapter).resolve()), subset=args.subset, n_images=len(full),
                    prompt=ds.prompt, candidates=ds.candidates, with_seg=bool(cfg.get("with_seg", True)),
                    seg_only=bool(cfg.get("seg_only", False)),
                    world_size=world, minutes=(time.time() - t0) / 60, config=cfg.get("_file"), argv=sys.argv)
        with open(out_dir / "run_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print(f"[infer] {len(full)} images in {info['minutes']:.1f} min, accuracy {float((full.pred_idx == full.class_idx).mean()):.4f} -> {out_dir}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
