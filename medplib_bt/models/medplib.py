"""Model "medplib_moe": load MedPLIB-7b-2e (Llama-7B with 2-expert MoE layers, CLIP-L/336 tower,
SAM-Med2D-B mask decoder, all included in the checkpoint), pick trainable modules, add LoRA,
save/load light-weight adapters (only the trained tensors)."""
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .. import paths
from ..registry import MODELS

paths.add_medplib_to_syspath()
from deepspeed.moe.layer import MoE  # noqa: E402
from model.MedPLIB import MedPLIBForCausalLM  # noqa: E402  (registers the config/model classes)
from model.medplib import conversation as conversation_lib  # noqa: E402
from utils.utils import ADD_OTHERS_TOKENS, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ------------------------------------------------------------------------------------------
# tokenizer / model
# ------------------------------------------------------------------------------------------
def load_tokenizer(ckpt_dir, model_max_length: int = 1024):
    tok = transformers.AutoTokenizer.from_pretrained(
        str(ckpt_dir), model_max_length=model_max_length, padding_side="right", use_fast=False, legacy=True)
    tok.pad_token = tok.unk_token
    n_added = 0
    for t in ADD_OTHERS_TOKENS + [f"<gen_{i}>" for i in range(1, 257)] + [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]:
        n_added += tok.add_tokens(t, special_tokens=True)
    assert n_added == 0, f"checkpoint tokenizer was missing {n_added} MedPLIB tokens"
    return tok


class GateLinear(nn.Linear):
    """Drop-in replacement for DeepSpeed TopKGate.wg.

    TopKGate.forward does `if self.wg.weight.dtype != torch.float32: self.wg = self.wg.float()`,
    which re-allocates the parameter storage in place. Under DeepSpeed bf16 training that
    silently detaches the router weight from the ZeRO optimizer's flat buffer, so the
    router would never actually train. This subclass keeps the parameter in the model dtype,
    makes `.float()` a no-op and computes the gate logits in fp32 (the input is already fp32).
    """

    def float(self):  # noqa: A003
        return self

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), None)


def patch_gate_linear(model) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, MoE):
            gate = m.deepspeed_moe.gate
            old = gate.wg
            if isinstance(old, GateLinear):
                continue
            new = GateLinear(old.in_features, old.out_features, bias=False)
            new.weight = old.weight  # same Parameter object (keeps its name and DeepSpeed attributes)
            gate.wg = new
            n += 1
    return n


def expert_group_name(model) -> str:
    for m in model.modules():
        if isinstance(m, MoE):
            return m.expert_group_name
    raise RuntimeError("no MoE layer found")


def set_conversation_template(name: str):
    conversation_lib.default_conversation = conversation_lib.conv_templates[name]


@MODELS.register("medplib_moe")
def build_medplib(cfg: dict, dtype=None, loss_overrides: Optional[Dict[str, float]] = None, tokenizer=None):
    """Returns (tokenizer, model). `cfg` is the model section of an experiment config."""
    dtype = dtype or DTYPES[cfg.get("precision", "bf16")]
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg["checkpoint"], cfg.get("model_max_length", 1024))
    seg_token = tokenizer("<SEG>", add_special_tokens=False).input_ids
    assert len(seg_token) == 1, seg_token
    seg_token_idx = seg_token[0]
    config = transformers.AutoConfig.from_pretrained(str(cfg["checkpoint"]))
    config.mm_vision_tower = str(cfg["clip_dir"])
    config.vision_tower = str(cfg["clip_dir"])
    config.use_cache = False
    loss = dict(cfg.get("loss", {}))
    loss.update(loss_overrides or {})
    model = MedPLIBForCausalLM.from_pretrained(
        str(cfg["checkpoint"]), config=config, torch_dtype=dtype, low_cpu_mem_usage=True,
        test_only=True, vision_pretrained=str(cfg["sam_ckpt"]) if cfg.get("sam_ckpt") else None,
        seg_token_idx=seg_token_idx,
        ce_loss_weight=float(loss.get("ce", 1.0)), bce_loss_weight=float(loss.get("bce", 2.0)),
        dice_loss_weight=float(loss.get("dice", 0.5)), iou_loss_weight=float(loss.get("iou", 0.0)),
        focal_loss_weight=float(loss.get("focal", 0.0)))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.vocab_size == len(tokenizer), (model.config.vocab_size, len(tokenizer))
    assert model.seg_token_idx == seg_token_idx
    patch_gate_linear(model)
    for n, p in model.named_parameters():
        assert p.device.type != "meta", f"parameter left on meta device: {n}"
    set_conversation_template(cfg.get("conv_template", "llava_v1"))
    return tokenizer, model


# ------------------------------------------------------------------------------------------
# trainable parameters / LoRA
# ------------------------------------------------------------------------------------------
def unwrap(model):
    """PeftModel / DeepSpeedEngine -> the underlying MedPLIBForCausalLM."""
    if hasattr(model, "module") and not isinstance(model, MedPLIBForCausalLM):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model


def find_lora_targets(model, target_substrings: Iterable[str], exclude: Iterable[str]) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(x in name for x in exclude):
            continue
        if any(t in name.split(".")[-1] for t in target_substrings):
            names.append(name)
    return sorted(names)


def configure_trainable(model, model_cfg: dict, lora_r: Optional[int] = None, lora_alpha: Optional[int] = None,
                        lora_dropout: Optional[float] = None):
    """Freeze everything, optionally wrap with LoRA, then unfreeze the configured modules (+ LoRA).
    Returns (model, LoraConfig or None)."""
    lcfg = dict(model_cfg.get("lora", {}))
    r = int(lcfg.get("r", 0) if lora_r is None else lora_r)
    alpha = int(lcfg.get("alpha", 32) if lora_alpha is None else lora_alpha)
    dropout = float(lcfg.get("dropout", 0.05) if lora_dropout is None else lora_dropout)
    for p in model.parameters():
        p.requires_grad_(False)
    lora_config = None
    if r > 0:
        from peft import LoraConfig, get_peft_model
        targets = find_lora_targets(model, lcfg.get("targets", []), lcfg.get("exclude", []))
        assert targets, "no LoRA target modules found"
        lora_config = LoraConfig(r=r, lora_alpha=alpha, target_modules=targets, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
    train_patterns = list(model_cfg.get("trainable", {}).get("new_modules", []))
    frozen = list(model_cfg.get("trainable", {}).get("frozen_always", []))
    for n, p in model.named_parameters():
        if any(pat in n for pat in train_patterns):
            p.requires_grad_(True)
        if any(pat in n for pat in frozen):
            p.requires_grad_(False)
    return model, lora_config


def enable_sam_encoder_grad(model) -> None:
    """MedPLIB runs the SAM image encoder under torch.no_grad(); replace that method so gradients
    reach trainable encoder parameters (the SAM-Med2D adapters)."""
    import types

    base = unwrap(model)

    def get_visual_embs(self, pixel_values):
        return self.model.visual_model.image_encoder(pixel_values)

    base.get_visual_embs = types.MethodType(get_visual_embs, base)


def mark_moe_params(model) -> int:
    """DeepSpeed requires at least one optimizer param group flagged as MoE when the model
    contains MoE layers. Trainable parameters living inside the MoE blocks (LoRA weights of
    the expert MLPs and the router) are flagged with the layer's expert group name; with
    ep_size=1 that group is the whole data-parallel world, i.e. plain all-reduce."""
    name = expert_group_name(unwrap(model))
    n = 0
    for pn, p in model.named_parameters():
        if p.requires_grad and "deepspeed_moe" in pn:
            p.allreduce = False
            p.group_name = name
            n += 1
    return n


def is_moe_param(p) -> bool:
    return hasattr(p, "allreduce") and not p.allreduce


def build_param_groups(model, lr_lora: float, lr_new: float, weight_decay: float):
    groups: Dict[tuple, list] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = "lora" if "lora_" in n else "new"
        decay = p.ndim > 1
        groups.setdefault((kind, is_moe_param(p), decay), []).append(p)
    out = []
    for (kind, moe, decay), ps in groups.items():
        g = dict(params=ps, lr=lr_lora if kind == "lora" else lr_new, weight_decay=weight_decay if decay else 0.0,
                 name=f"{kind}_{'moe' if moe else 'dense'}_{'wd' if decay else 'nowd'}", kind=kind)
        if moe:
            g["moe"] = True
            g["name"] = ps[0].group_name  # DeepSpeed looks the expert data-parallel group up by this name
        out.append(g)
    return out


def count_params(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot


# ------------------------------------------------------------------------------------------
# adapter checkpoints: only the trainable tensors (LoRA + new modules), ~0.1-0.5 GB
# ------------------------------------------------------------------------------------------
def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_adapter(model, out_dir, lora_config, extra: Optional[dict] = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = trainable_state_dict(model)
    torch.save(sd, out_dir / "trainable.pt")
    lora = None
    if lora_config is not None:
        lora = lora_config.to_dict()
        lora["target_modules"] = sorted(lora["target_modules"])  # PEFT keeps a set, which is not JSON-serialisable
    meta = dict(n_tensors=len(sd), n_params=int(sum(v.numel() for v in sd.values())), lora=lora)
    if extra:
        meta.update(extra)
    with open(out_dir / "adapter_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta


def load_adapter_meta(adapter_dir) -> dict:
    with open(Path(adapter_dir) / "adapter_meta.json") as f:
        return json.load(f)


def load_trainable_into(model, state_dict: Dict[str, torch.Tensor]) -> int:
    """Copy tensors by name; tolerates the `base_model.model.` prefix PEFT adds."""
    params = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for k, v in state_dict.items():
            k2 = k if k in params else None
            if k2 is None and ("base_model.model." + k) in params:
                k2 = "base_model.model." + k
            if k2 is None and k.startswith("base_model.model.") and k[len("base_model.model."):] in params:
                k2 = k[len("base_model.model."):]
            if k2 is None:
                raise KeyError(f"tensor {k} not found in model")
            params[k2].copy_(v.to(params[k2].dtype))
            n += 1
    return n


def apply_adapter(model, adapter_dir):
    """Wrap `model` with the LoRA config stored in adapter_dir (if any) and load the trained
    tensors. Returns the (possibly PEFT-wrapped) model."""
    meta = load_adapter_meta(adapter_dir)
    lora = meta.get("lora")
    if lora:
        from peft import LoraConfig, get_peft_model
        tm = lora["target_modules"]
        if isinstance(tm, str):  # adapters written before target_modules was stored as a list
            import ast
            tm = ast.literal_eval(tm)
        cfg = LoraConfig(r=lora["r"], lora_alpha=lora["lora_alpha"], target_modules=sorted(tm),
                         lora_dropout=lora["lora_dropout"], bias=lora["bias"], task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    sd = torch.load(Path(adapter_dir) / "trainable.pt", map_location="cpu")
    n = load_trainable_into(model, sd)
    assert n == meta["n_tensors"], (n, meta["n_tensors"])
    return model


def copy_adapter_dir(src, dst):
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
