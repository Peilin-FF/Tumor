#!/usr/bin/env python3
"""CPU-only check of the model side (no GPU needed): load MedPLIB-7b-2e, wrap with LoRA, select
trainable modules, build optimizer groups, save/load an adapter, run the CLIP+projector image
token expansion and the SAM mask decoder on one batch."""
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medplib_bt.config import load_config  # noqa: E402
from medplib_bt.datasets import brisc  # noqa: E402,F401
from medplib_bt.datasets.brisc import collate_infer  # noqa: E402
from medplib_bt.models import medplib as M  # noqa: E402
from medplib_bt.registry import DATASETS, MODELS  # noqa: E402

t0 = time.time()
cfg = load_config("configs/train/stage_b.yaml")
tok, model = MODELS.build(cfg["model"], dtype=torch.bfloat16)
print(f"loaded in {time.time()-t0:.0f}s; vocab {len(tok)}; seg idx {model.seg_token_idx}")
n_gate = sum(isinstance(m, M.GateLinear) for m in model.modules())
print("GateLinear patched:", n_gate, "| any meta params:", any(p.device.type == "meta" for p in model.parameters()))
tot = sum(p.numel() for p in model.parameters())
print(f"total params {tot/1e9:.3f}B")

model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model, lora_cfg = M.configure_trainable(model, cfg["model"], lora_r=16, lora_alpha=32, lora_dropout=0.05)
n_moe = M.mark_moe_params(model)
tr, tot = M.count_params(model)
print(f"trainable {tr/1e6:.1f}M ({100*tr/tot:.3f}%), lora targets {len(lora_cfg.target_modules)}, moe-flagged {n_moe}")
groups = M.build_param_groups(model, 2e-5, 1e-4, 0.01)
for g in groups:
    print(f"  group {g['name']:<22} lr {g['lr']:.1e} wd {g['weight_decay']} n={len(g['params'])} params={sum(p.numel() for p in g['params'])/1e6:.2f}M moe={g.get('moe', False)}")
by_kind = {}
for n, p in model.named_parameters():
    if p.requires_grad:
        k = "lora" if "lora_" in n else n.split(".")[-3] if "text_hidden_fcs" in n else ("mask_decoder" if "mask_decoder" in n else "mm_projector" if "mm_projector" in n else "gate.wg" if "gate.wg" in n else "other")
        by_kind[k] = by_kind.get(k, 0) + p.numel()
print("trainable by kind (M):", {k: round(v / 1e6, 2) for k, v in by_kind.items()})

# adapter round trip
out = Path("/tmp/medplib_bt_adapter_check")
meta = M.save_adapter(model, out, lora_cfg, extra=dict(check=True))
sd = torch.load(out / "trainable.pt")
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.add_(1.0)
n = M.load_trainable_into(model, sd)
ok = all(torch.equal(dict(model.named_parameters())[k].cpu(), v.to(dict(model.named_parameters())[k].dtype)) for k, v in sd.items())
print(f"adapter round trip: saved {meta['n_tensors']} tensors ({meta['n_params']/1e6:.1f}M), reloaded {n}, equal={ok}")

# image token expansion + seg mask + SAM decoder on one batch (CPU, bf16)
model.eval()
base = M.unwrap(model)
vt = base.get_model().get_vision_tower()
data_args = types.SimpleNamespace(image_processor=vt.image_processor, image_aspect_ratio="pad", is_multimodal=True, mm_use_im_start_end=True)
ds = DATASETS.build(cfg["data"], mode="infer", subset="test", tokenizer=tok, data_args=data_args, model_cfg=cfg["model"], prompt="joint", with_seg=True)
batch = collate_infer([ds[0]])
from medplib_bt.engine.inference import pad_sequences
input_ids = pad_sequences(batch["input_ids"], tok.pad_token_id)
labels = pad_sequences(batch["labels"], -100)
attn = input_ids.ne(tok.pad_token_id)
images_clip = batch["images_clip"].to(torch.bfloat16)[batch["owner"]]
with torch.no_grad():
    _, attn2, _, emb, new_labels = base.prepare_inputs_labels_for_multimodal(input_ids, attn, None, labels, images_clip, [], [], mask_images=None, image_token_types=None)
    seg_mask = base.build_seg_token_mask(input_ids)
    print("input_ids", tuple(input_ids.shape), "-> inputs_embeds", tuple(emb.shape), "attn2", tuple(attn2.shape), "labels", tuple(new_labels.shape), "seg_mask", tuple(seg_mask.shape), "seg per row", seg_mask.sum(1).tolist())
    assert emb.shape[1] == seg_mask.shape[1] == new_labels.shape[1]
    img_emb = base.get_visual_embs(batch["images"].to(torch.bfloat16))
    vm = base.model.visual_model
    text_embed = torch.randn(1, 1, 256, dtype=torch.bfloat16)
    sparse, dense = vm.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=text_embed)
    low_res, iou = vm.mask_decoder(image_embeddings=img_emb[:1], image_pe=vm.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
    m = base.postprocess_masks(low_res, input_size=batch["resize_list"][0], original_size=batch["meta"][0]["orig_size"])
    print("SAM image emb", tuple(img_emb.shape), "low_res", tuple(low_res.shape), "mask", tuple(m.shape), "iou_pred", float(iou[0, 0]))
print(f"ALL OK in {time.time()-t0:.0f}s")
