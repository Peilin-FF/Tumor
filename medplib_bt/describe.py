"""Free-text generation for a few images: the model answers each prompt with greedy decoding; if the
answer contains <SEG>, the mask is decoded from that token's hidden state (a teacher-forced pass over
the generated sequence, the same path as infer.py). Illustrates the language interface; not a metric.

    deepspeed --include=localhost:0 --master_port 29650 --module medplib_bt.describe \
        --adapter outputs/runs/stageB_v2_seed42/final --ids brisc2025_test_00013_gl_ax_t1 ... --out outputs/report/describe_v2
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import cv2
import deepspeed
import numpy as np
import torch
import torch.distributed as dist

from . import paths
from .config import load_config
from .datasets import brisc  # noqa: F401
from .datasets.brisc import load_manifest
from .datasets.task import TaskSpec
from .engine.inference import IGNORE_INDEX, prob_to_png
from .models import medplib as M
from .registry import DATASETS, MODELS

paths.add_medplib_to_syspath()
from deepspeed.moe.layer import MoE  # noqa: E402

COLON = 29901  # ":" that ends "ASSISTANT:" in the llava_v1 template
PROMPTS = {
    "joint": None,  # filled from the dataset config
    "describe": "Describe this contrast-enhanced T1 brain MRI. Is there a tumor? If so, what type is it and where is it located?",
    "plane": "Which imaging plane is this brain MRI: axial, coronal, or sagittal?",
}


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    ap.add_argument("--config", default="configs/infer/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out", required=True, help="output dir: answers.json + masks/")
    return ap.parse_args(argv)


@torch.no_grad()
def decode_mask(base, input_ids, images_clip, images_sam, resize, orig_size, dtype):
    """Mask from the first <SEG> of a full (prompt + answer) sequence."""
    attn = torch.ones_like(input_ids, dtype=torch.bool)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    _, attn2, _, emb_in, _ = base.prepare_inputs_labels_for_multimodal(
        input_ids, attn, None, labels, images_clip, [], [], mask_images=None, image_token_types=None)
    out = base.model(input_ids=None, attention_mask=attn2, inputs_embeds=emb_in, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = out.hidden_states[-1]
    seg_mask = base.build_seg_token_mask(input_ids)
    emb = base.model.text_hidden_fcs[0](hidden)[seg_mask][:1]
    vm = base.model.visual_model
    image_embeddings = base.get_visual_embs(images_sam)
    sparse, dense = vm.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=emb[None])
    low_res, iou_pred = vm.mask_decoder(image_embeddings=image_embeddings, image_pe=vm.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=sparse.to(emb.dtype), dense_prompt_embeddings=dense, multimask_output=False)
    m = base.postprocess_masks(low_res, input_size=resize, original_size=orig_size)
    return torch.sigmoid(m[0, 0].float()).cpu().numpy(), float(iou_pred[0, 0])


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config, args.set)
    deepspeed.init_distributed(dist_backend="nccl")
    assert dist.get_world_size() == 1, "single process"
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)
    dtype = M.DTYPES[cfg["model"].get("precision", "bf16")]
    spec = TaskSpec.from_cfg(cfg["data"])
    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    tokenizer, model = MODELS.build(cfg["model"], dtype=dtype)
    if not args.no_adapter:
        assert args.adapter, "--adapter or --no-adapter"
        model = M.apply_adapter(model, args.adapter)
    model.to(device).eval()
    for m in model.modules():
        if isinstance(m, MoE):
            m.set_deepspeed_parallelism()
    base = M.unwrap(model)

    df = load_manifest(cfg["data"], "test")
    df = df[df.image_id.isin(args.ids)].set_index("image_id").loc[args.ids].reset_index()
    vision_tower = base.get_model().get_vision_tower()
    data_args = types.SimpleNamespace(image_processor=vision_tower.image_processor, image_aspect_ratio="pad",
                                      is_multimodal=True, mm_use_im_start_end=True)
    ds = DATASETS.build(cfg["data"], mode="infer", subset="test", tokenizer=tokenizer, data_args=data_args,
                        model_cfg=cfg["model"], prompt=cfg.get("prompt", "joint"), with_seg=True, seg_only=False, df=df)
    prompts = dict(PROMPTS, joint=spec.prompt_for("joint"))
    seg_id = ds.seg_token_idx

    rows = []
    for i in range(len(ds)):
        s = ds[i]
        images_clip = s["image_clip"][None].to(device=device, dtype=dtype)
        images_sam = s["image_sam"][None].to(device=device, dtype=dtype)
        gt = s["gt_mask"].numpy().astype(bool)
        for key in args.prompts:
            prompt = prompts[key]
            ids, *_ = ds.base.encode_text(prompt, "")
            cut = (ids == COLON).nonzero()[-1].item() + 1
            input_ids = ids[:cut][None].to(device)
            gen = model.generate(input_ids, images=images_clip, attention_mask=torch.ones_like(input_ids), mask_images=None,
                                 image_token_types=None, do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True)
            new = gen[0, input_ids.shape[1]:] if gen.shape[1] > input_ids.shape[1] and torch.equal(gen[0, :input_ids.shape[1]], input_ids[0]) else gen[0]
            text = tokenizer.decode(new, skip_special_tokens=False).replace(tokenizer.eos_token, "").strip()
            rec = dict(image_id=s["image_id"], class_key=s["class_key"], plane=s["plane"], mask_type=s["mask_type"],
                       prompt_key=key, prompt=prompt, text=text, has_seg=bool((new == seg_id).any()), n_tokens=int(len(new)))
            if rec["has_seg"]:
                upto = (new == seg_id).nonzero()[0].item() + 1
                full = torch.cat([input_ids[0], new[:upto]])[None]
                prob, iou_pred = decode_mask(base, full, images_clip, images_sam, s["resize"], s["orig_size"], dtype)
                pred = prob >= float(cfg.get("seg_threshold", 0.5))
                inter = float((pred & gt).sum())
                rec.update(mask_area=float(pred.mean()), iou_pred=iou_pred,
                           dice=(2 * inter / (pred.sum() + gt.sum())) if (pred.sum() + gt.sum()) > 0 else 1.0,
                           mask_file=f"masks/{s['image_id']}__{key}.png")
                cv2.imwrite(str(out / rec["mask_file"]), prob_to_png(prob))
            rows.append(rec)
            print(f"[describe] {s['image_id']} [{key}] -> {text!r}" + (f"  mask {rec['mask_area']:.4f} dice {rec.get('dice', float('nan')):.3f}" if rec["has_seg"] else ""), flush=True)
    with open(out / "answers.json", "w") as f:
        json.dump(dict(adapter=None if args.no_adapter else args.adapter, prompts=prompts, rows=rows), f, indent=2)
    print(f"[describe] {len(rows)} answers -> {out}", flush=True)


if __name__ == "__main__":
    main()
