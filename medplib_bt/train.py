"""Fine-tune MedPLIB-7b-2e on BRISC 2025 with DeepSpeed (ZeRO-2, bf16).

Stage A: router / visual projector / text projector / mask decoder, 1 epoch.
Stage B: stage-A modules + LoRA on attention and expert MLPs, initialised from stage A.
Loss = ce * CE(text) + bce * BCE(mask) + dice * Dice(mask). An adapter (the trained tensors)
is written after every epoch; `final/` is the last one.

    deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 29511 --module medplib_bt.train \
        --config configs/train/stage_a.yaml --exp-name stageA_seed42
    deepspeed ... --module medplib_bt.train --config configs/train/stage_b.yaml --exp-name stageB_seed42 \
        --init-from outputs/runs/stageA_seed42/final
"""
import argparse
import json
from datetime import timedelta
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from . import paths
from .config import dump, load_config
from .datasets import brisc  # noqa: F401
from .datasets.brisc import TaskBatchSampler, collate_train
from .models import medplib as M
from .registry import DATASETS, MODELS

paths.add_medplib_to_syspath()
from utils.utils import dict_to_cuda  # noqa: E402


def parse_args(argv):
    ap = argparse.ArgumentParser(description="MedPLIB BRISC fine-tuning")
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    ap.add_argument("--config", required=True, help="configs/train/*.yaml")
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. --set seed=123 --set model.lora.r=8")
    ap.add_argument("--init-from", default=None, help="adapter dir to initialise from (stage B <- stage A final)")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N optimizer steps")
    ap.add_argument("--outputs-root", default=str(paths.get("outputs_root")))
    return ap.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank0_print(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def make_lr_lambda(total_steps, warmup_steps, min_ratio):
    def f(step):
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config, args.set)
    deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)
    seed = int(cfg.get("seed", 42))
    set_seed(seed + rank)
    dtype = M.DTYPES[cfg["model"].get("precision", "bf16")]
    bcfg, ocfg = cfg["batch"], cfg["optim"]
    exp_dir = Path(args.outputs_root) / "runs" / args.exp_name
    if rank == 0:
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "config.yaml", "w") as f:
            f.write(dump(dict(cfg, world_size=world, effective_batch=bcfg["micro"] * bcfg["grad_accum"] * world,
                              init_from=args.init_from, argv=sys.argv)))
    rank0_print(f"[train] stage {cfg['stage']} exp {args.exp_name} world {world} "
                f"effective batch {bcfg['micro'] * bcfg['grad_accum'] * world}")

    # ---- model ---------------------------------------------------------------------
    tokenizer, model = MODELS.build(cfg["model"], dtype=dtype)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    lora = cfg.get("lora", {})
    model, lora_config = M.configure_trainable(model, cfg["model"], lora_r=lora.get("r"), lora_alpha=lora.get("alpha"),
                                               lora_dropout=lora.get("dropout"))
    if args.init_from:
        n = M.load_trainable_into(model, torch.load(Path(args.init_from) / "trainable.pt", map_location="cpu"))
        rank0_print(f"[train] initialised {n} tensors from {args.init_from}")
    if any(p.requires_grad and "visual_model.image_encoder" in n for n, p in model.named_parameters()):
        M.enable_sam_encoder_grad(model)
        rank0_print("[train] SAM image-encoder parameters are trainable: encoder runs with gradients")
    n_moe = M.mark_moe_params(model)
    tr, tot = M.count_params(model)
    rank0_print(f"[train] trainable params {tr/1e6:.1f}M of {tot/1e9:.2f}B ({100*tr/tot:.3f} %), {n_moe} MoE-flagged tensors")
    if rank == 0:
        with open(exp_dir / "trainable_params.txt", "w") as f:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    f.write(f"{n}\t{tuple(p.shape)}\n")

    # ---- data ----------------------------------------------------------------------
    vision_tower = M.unwrap(model).get_model().get_vision_tower()
    data_args = types.SimpleNamespace(image_processor=vision_tower.image_processor, image_aspect_ratio="pad",
                                      is_multimodal=True, mm_use_im_start_end=True)
    train_ds = DATASETS.build(cfg["data"], mode="train", subset="train", tokenizer=tokenizer, data_args=data_args,
                              model_cfg=cfg["model"], seed=seed, augment=cfg.get("augment"), fixed_task=cfg.get("fixed_task"),
                              max_samples=int(cfg.get("max_train_samples", 0)))
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
    batch_sampler = TaskBatchSampler(sampler, bcfg["micro"], train_ds.task_mix, seed=seed, fixed_task=cfg.get("fixed_task"))
    workers = int(cfg.get("workers", 4))
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=workers, collate_fn=collate_train)
    epochs = int(cfg["epochs"])
    micro_per_epoch = len(train_loader)
    steps_per_epoch = max(1, micro_per_epoch // bcfg["grad_accum"])
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(round(float(ocfg.get("warmup_ratio", 0.03)) * total_steps))
    rank0_print(f"[train] {len(train_ds)} images/rank-shard x {world}; {micro_per_epoch} micro-batches, "
                f"{steps_per_epoch} optimizer steps per epoch, {total_steps} total, warmup {warmup_steps}")

    # ---- optimiser / engine ----------------------------------------------------------
    param_groups = M.build_param_groups(model, float(ocfg["lr_lora"]), float(ocfg["lr_new"]), float(ocfg.get("weight_decay", 0.01)))
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(ocfg.get("betas", (0.9, 0.999))), eps=1e-8)
    lr_lambda = make_lr_lambda(total_steps, warmup_steps, float(ocfg.get("min_lr_ratio", 0.0)))
    ds_config = {
        "train_micro_batch_size_per_gpu": bcfg["micro"],
        "gradient_accumulation_steps": bcfg["grad_accum"],
        "gradient_clipping": float(ocfg.get("grad_clip", 1.0)),
        "bf16": {"enabled": dtype == torch.bfloat16},
        "zero_optimization": {"stage": int(cfg.get("deepspeed", {}).get("zero_stage", 2)), "contiguous_gradients": True,
                              "overlap_comm": True, "reduce_scatter": True, "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8},
        "steps_per_print": 10 ** 9,
        "wall_clock_breakdown": False,
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config,
        lr_scheduler=lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda))

    # ---- loop ----------------------------------------------------------------------
    log_every = int(cfg.get("log_every", 5))
    log_path = exp_dir / "train_log.jsonl"
    history = []
    keys = ("loss", "ce_loss", "mask_bce_loss", "mask_dice_loss", "unscale_mask_bce_loss", "unscale_mask_dice_loss")
    stop = False
    for epoch in range(epochs):
        batch_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        engine.train()
        t0 = time.time()
        meters = {k: 0.0 for k in keys}
        n_meter = 0
        ep_sum = {k: 0.0 for k in keys}
        ep_n = 0
        for it, batch in enumerate(train_loader):
            batch = dict_to_cuda(batch)
            batch["images"] = batch["images"].to(dtype)
            batch["images_clip"] = batch["images_clip"].to(dtype)
            out = engine(**batch)
            engine.backward(out["loss"])
            engine.step()
            for k in keys:
                v = float(out[k])
                meters[k] += v
                ep_sum[k] += v
            n_meter += 1
            ep_n += 1
            if (it + 1) % log_every == 0 and rank == 0:
                lrs = list(engine.get_lr())
                rec = dict(epoch=epoch, micro=it + 1, opt_step=int(engine.global_steps), **{k: v / n_meter for k, v in meters.items()},
                           lr_max=max(lrs), lr_min=min(lrs), sec_per_micro=(time.time() - t0) / (it + 1),
                           mem_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30)
                print(f"[ep {epoch} {it+1}/{micro_per_epoch} step {rec['opt_step']}] loss {rec['loss']:.4f} ce {rec['ce_loss']:.4f} "
                      f"bce {rec['unscale_mask_bce_loss']:.4f} dice {rec['unscale_mask_dice_loss']:.4f} lr {rec['lr_max']:.2e} "
                      f"{rec['sec_per_micro']:.2f}s/it mem {rec['mem_gb']:.1f}G", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                meters = {k: 0.0 for k in keys}
                n_meter = 0
            if args.max_steps and engine.global_steps >= args.max_steps:
                stop = True
                break
        dist.barrier()
        ep_dir = exp_dir / f"epoch_{epoch:02d}"
        if rank == 0:
            M.save_adapter(engine.module, ep_dir, lora_config, extra=dict(epoch=epoch, stage=cfg["stage"], seed=seed,
                                                                        opt_steps=int(engine.global_steps)))
            rec = dict(epoch=epoch, minutes=(time.time() - t0) / 60, opt_steps=int(engine.global_steps),
                       peak_mem_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30, **{k: v / max(1, ep_n) for k, v in ep_sum.items()})
            history.append(rec)
            with open(exp_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            M.copy_adapter_dir(ep_dir, exp_dir / "final")
            print(f"[epoch {epoch} done in {rec['minutes']:.1f} min] mean loss {rec['loss']:.4f} ce {rec['ce_loss']:.4f} "
                  f"bce {rec['unscale_mask_bce_loss']:.4f} dice {rec['unscale_mask_dice_loss']:.4f}", flush=True)
        dist.barrier()
        if stop:
            break
    rank0_print(f"[train] finished -> {exp_dir / 'final'}")
    dist.barrier()


if __name__ == "__main__":
    main()
