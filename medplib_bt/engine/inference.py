"""Inference: class = argmax of the sequence log-likelihood of the four canonical answers under
the fixed prompt (no free generation); mask = the <SEG> hidden state of the chosen answer
decoded by SAM-Med2D into a full-resolution logit map. The mask branch always runs, also for
a non-tumorous prediction. Shared by infer.py and the demo."""
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import paths
from ..models.medplib import unwrap

paths.add_medplib_to_syspath()
from utils.utils import IGNORE_INDEX  # noqa: E402


def pad_sequences(seqs, pad_value):
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)


@torch.no_grad()
def run_candidates(model, tokenizer, batch: Dict, device, class_keys: List[str], dtype=torch.bfloat16,
                   decode_masks: bool = True) -> List[Dict]:
    """batch: output of data.brisc.collate_infer. Returns one dict per image with
    ll (C,), ll_mean (C,), pred_idx, pred_key, mask_logits (H,W float32 cpu) and iou_pred."""
    base = unwrap(model)
    B = len(batch["meta"])
    C = batch["n_candidates"]
    owner = batch["owner"].to(device)
    input_ids = pad_sequences(batch["input_ids"], tokenizer.pad_token_id).to(device)
    score = pad_sequences(batch["score_masks"], False).to(device)
    score_labels = torch.where(score, input_ids, torch.full_like(input_ids, IGNORE_INDEX))
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    images_clip = batch["images_clip"].to(device=device, dtype=dtype)[owner]
    images_sam = batch["images"].to(device=device, dtype=dtype)

    # LLaVA-style image-token expansion (the same routine the training forward uses)
    _, attn2, _, inputs_embeds, new_labels = base.prepare_inputs_labels_for_multimodal(
        input_ids, attention_mask, None, score_labels, images_clip, [], [], mask_images=None, image_token_types=None)
    out = base.model(input_ids=None, attention_mask=attn2, inputs_embeds=inputs_embeds,
                     output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = out.hidden_states[-1]  # (B*C, L, 4096) after the final norm

    # class log-likelihood: sum of log p(token) over the answer tokens (excluding <SEG> and </s>)
    tgt = new_labels[:, 1:]
    valid = tgt != IGNORE_INDEX
    rows, cols = valid.nonzero(as_tuple=True)
    ll = torch.zeros(B * C, device=device, dtype=torch.float32)
    n_tok = torch.zeros(B * C, device=device, dtype=torch.float32)
    if rows.numel() > 0:
        h = hidden[rows, cols]
        logp = F.log_softmax(base.lm_head(h).float(), dim=-1)
        tok_lp = logp.gather(-1, tgt[rows, cols].unsqueeze(-1)).squeeze(-1)
        ll.index_add_(0, rows, tok_lp)
        n_tok.index_add_(0, rows, torch.ones_like(tok_lp))
    ll = ll.view(B, C)
    ll_mean = ll / n_tok.view(B, C).clamp(min=1)
    pred_idx = ll.argmax(dim=1)

    mask_logits = [None] * B
    iou_preds = [None] * B
    if decode_masks:
        seg_mask = base.build_seg_token_mask(input_ids)
        assert seg_mask.shape[1] == hidden.shape[1], (seg_mask.shape, hidden.shape)
        counts = seg_mask.sum(1)
        assert bool((counts == 1).all()), f"expected exactly one <SEG> per candidate row, got {counts.tolist()}"
        emb = base.model.text_hidden_fcs[0](hidden)[seg_mask]                      # (B*C, 256)
        emb = emb.view(B, C, -1)[torch.arange(B, device=device), pred_idx]          # (B, 256)
        image_embeddings = base.get_visual_embs(images_sam)                         # (B, 256, 16, 16)
        vm = base.model.visual_model
        for i in range(B):
            sparse, dense = vm.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=emb[i][None, None, :])
            low_res, iou_pred = vm.mask_decoder(
                image_embeddings=image_embeddings[i:i + 1], image_pe=vm.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse.to(emb.dtype), dense_prompt_embeddings=dense, multimask_output=False)
            m = base.postprocess_masks(low_res, input_size=batch["resize_list"][i], original_size=batch["meta"][i]["orig_size"])
            mask_logits[i] = m[0, 0].float().cpu()
            iou_preds[i] = float(iou_pred[0, 0])
    results = []
    for i in range(B):
        results.append(dict(meta=batch["meta"][i], ll=ll[i].float().cpu().numpy(), ll_mean=ll_mean[i].float().cpu().numpy(),
                            pred_idx=int(pred_idx[i]), pred_key=class_keys[int(pred_idx[i])],
                            mask_logits=mask_logits[i], iou_pred=iou_preds[i], gt_mask=batch["gt_masks"][i]))
    return results


# ------------------------------------------------------------------------------------------
# post-processing + metrics (numpy)
# ------------------------------------------------------------------------------------------
def postprocess_mask(prob: np.ndarray, threshold: float, min_component_ratio: float) -> np.ndarray:
    """prob in [0,1] -> binary uint8 mask; drop connected components smaller than
    min_component_ratio * (H*W) pixels."""
    m = (prob > threshold).astype(np.uint8)
    if min_component_ratio > 0 and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component_ratio * m.size
        m = keep[lab].astype(np.uint8)
    return m


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    denom = 2 * tp + fp + fn
    union = tp + fp + fn
    return dict(dice=(2 * tp / denom) if denom > 0 else 1.0, iou=(tp / union) if union > 0 else 1.0,
                sensitivity=(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
                specificity=(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
                pred_area_ratio=float(pred.mean()), gt_area_ratio=float(gt.mean()), tp=tp, fp=fp, fn=fn, tn=tn)


def softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def prob_to_png(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8)


def png_to_prob(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert m is not None, path
    return m.astype(np.float32) / 255.0
