"""Gradio demo: upload a T1 brain MRI -> predicted class with confidence, tumour mask,
overlay, lesion area ratio and a class/mask consistency warning; mask and JSON download.

    python -m medplib_bt.demo_app --adapter outputs/runs/stageB_seed42/final --port 7860
"""
import argparse
import json
import os
import tempfile
import types

import cv2
import gradio as gr
import numpy as np
import torch

from . import paths
from .config import load_config
from .datasets import brisc  # noqa: F401
from .datasets.brisc import BriscInferenceDataset, collate_infer
from .datasets.task import TaskSpec
from .engine.inference import mask_metrics, postprocess_mask, run_candidates, softmax_np
from .models import medplib as M
from .registry import MODELS

paths.add_medplib_to_syspath()
from deepspeed.moe.layer import MoE  # noqa: E402


def init_single_process_dist():
    import deepspeed
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29799")
    deepspeed.init_distributed(dist_backend="nccl")


class Predictor:
    def __init__(self, cfg, adapter, device):
        self.cfg = cfg
        self.spec = TaskSpec.from_cfg(cfg["data"])
        self.device = device
        self.dtype = M.DTYPES[cfg["model"].get("precision", "bf16")]
        self.tokenizer, model = MODELS.build(cfg["model"], dtype=self.dtype)
        if adapter:
            model = M.apply_adapter(model, adapter)
        self.model = model.to(device).eval()
        for m in self.model.modules():
            if isinstance(m, MoE):
                m.set_deepspeed_parallelism()
        vt = M.unwrap(self.model).get_model().get_vision_tower()
        self.data_args = types.SimpleNamespace(image_processor=vt.image_processor, image_aspect_ratio="pad",
                                               is_multimodal=True, mm_use_im_start_end=True)
        import pandas as pd
        self.template_df = pd.DataFrame([dict(image_id="upload", class_key=self.spec.class_keys[0], plane="na", subset="demo",
                                              mask_type="derived_empty", image_path="", mask_path="")])
        self.ds = BriscInferenceDataset(self.template_df, self.tokenizer, self.data_args, self.spec, root=".",
                                        sam_img_size=int(cfg["model"].get("sam_img_size", 256)))

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray):
        img_rgb = np.ascontiguousarray(img_rgb[..., :3])
        image_sam, image_clip, resize = self.ds.base.encode_image(img_rgb)
        ids, labs, scores = [], [], []
        for cand in self.spec.candidates():
            input_ids, labels, *_ = self.ds.base.encode_text(self.ds.prompt, f"{cand} {self.spec.seg_token}")
            ids.append(input_ids)
            labs.append(labels)
            scores.append((labels != -100) & (labels != self.ds.seg_token_idx) & (labels != self.ds.eos))
        sample = dict(image_id="upload", class_key=self.spec.class_keys[0], class_idx=0, plane="na", subset="demo",
                      mask_type="derived_empty", image_path="", image_sam=image_sam, image_clip=image_clip, resize=resize,
                      gt_mask=torch.zeros(img_rgb.shape[:2], dtype=torch.uint8), orig_size=tuple(img_rgb.shape[:2]),
                      input_ids=ids, labels=labs, score_masks=scores)
        r = run_candidates(self.model, self.tokenizer, collate_infer([sample]), self.device, self.spec.class_keys, self.dtype)[0]
        prob = torch.sigmoid(r["mask_logits"]).numpy()
        mask = postprocess_mask(prob, float(self.cfg.get("seg_threshold", 0.5)), float(self.cfg.get("min_component_ratio", 0.0)))
        probs = softmax_np(r["ll"][None])[0]
        return r["pred_key"], probs, mask, prob


def build_ui(pred: Predictor):
    def run(image):
        if image is None:
            return None, None, "", None, None
        key, probs, mask, prob = pred.predict(image)
        nt = "non_tumorous"
        area = float(mask.mean())
        conflict = (key == nt and mask.any()) or (key != nt and not mask.any())
        overlay = image[..., :3].copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 0), 1)
        result = dict(canonical_class=key, class_probabilities={k: float(p) for k, p in zip(pred.spec.class_keys, probs)},
                      confidence=float(probs.max()), lesion_area_ratio=area, consistency_flag=bool(conflict),
                      seg_threshold=float(pred.cfg.get("seg_threshold", 0.5)))
        text = f"**{key}**  (confidence {probs.max():.3f})\n\nlesion area ratio: {area:.4f}"
        if conflict:
            text += "\n\n⚠️ class and mask disagree, please review"
        tmp = tempfile.mkdtemp()
        mask_path = os.path.join(tmp, "mask.png")
        json_path = os.path.join(tmp, "result.json")
        cv2.imwrite(mask_path, mask * 255)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        return mask * 255, overlay, text, mask_path, json_path

    with gr.Blocks(title="MedPLIB brain tumour demo") as demo:
        gr.Markdown("# MedPLIB brain tumour classification + segmentation (BRISC 2025)\nResearch prototype, not for clinical use.")
        with gr.Row():
            inp = gr.Image(type="numpy", label="T1 brain MRI")
            mask_out = gr.Image(label="predicted mask")
            over_out = gr.Image(label="overlay")
        txt = gr.Markdown()
        with gr.Row():
            f1 = gr.File(label="mask.png")
            f2 = gr.File(label="result.json")
        btn = gr.Button("Run")
        btn.click(run, inputs=inp, outputs=[mask_out, over_out, txt, f1, f2])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infer/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    init_single_process_dist()
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    demo = build_ui(Predictor(cfg, args.adapter, device))
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
