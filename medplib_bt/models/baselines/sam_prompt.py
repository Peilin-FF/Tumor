"""Model "sam_med2d_prompt": the SAM-Med2D ViT-B checkpoint driven by a box or point prompt derived
from the expert mask (an oracle-prompt upper bound for prompted segmentation). No training."""
import numpy as np
import torch
import torch.nn.functional as F

from ... import paths
from ...registry import MODELS

paths.add_medplib_to_syspath()
from model.segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402
from model.segment_anything_med2d import build_sam_vit_b  # noqa: E402

PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


class SamPrompter(torch.nn.Module):
    def __init__(self, checkpoint, image_size: int = 256, prompt: str = "box", box_jitter: float = 0.0, seed: int = 42):
        super().__init__()
        self.sam = build_sam_vit_b(str(checkpoint), image_size=image_size, encoder_adapter=True)
        self.image_size, self.prompt, self.box_jitter = image_size, prompt, box_jitter
        self.transform = ResizeLongestSide(image_size)
        self.rng = np.random.default_rng(seed)

    def preprocess(self, img_rgb: np.ndarray):
        resized = self.transform.apply_image(img_rgb)
        h, w = resized.shape[:2]
        x = (torch.from_numpy(resized).permute(2, 0, 1).float() - PIXEL_MEAN) / PIXEL_STD
        pad = torch.zeros(3, self.image_size, self.image_size)
        top, left = (self.image_size - h) // 2, (self.image_size - w) // 2   # MedPLIB pads centred
        pad[:, top:top + h, left:left + w] = x
        return pad, (h, w), (top, left)

    def prompt_from_mask(self, mask: np.ndarray, orig_size, resize, offset):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return None
        sy, sx = resize[0] / orig_size[0], resize[1] / orig_size[1]
        if self.prompt == "box":
            x0, x1, y0, y1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
            if self.box_jitter > 0:
                bw, bh = x1 - x0, y1 - y0
                j = self.rng.uniform(-self.box_jitter, self.box_jitter, 4)
                x0, x1, y0, y1 = x0 + j[0] * bw, x1 + j[1] * bw, y0 + j[2] * bh, y1 + j[3] * bh
            box = torch.tensor([[x0 + offset[1], y0 + offset[0], x1 + offset[1], y1 + offset[0]]], dtype=torch.float32)
            return dict(boxes=box)
        cy, cx = ys.mean() * sy + offset[0], xs.mean() * sx + offset[1]
        return dict(points=(torch.tensor([[[cx, cy]]], dtype=torch.float32), torch.tensor([[1]], dtype=torch.int64)))

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray, mask: np.ndarray, device):
        x, resize, offset = self.preprocess(img_rgb)
        pr = self.prompt_from_mask(mask, img_rgb.shape[:2], resize, offset)
        if pr is None:
            return np.zeros(img_rgb.shape[:2], np.float32)
        emb = self.sam.image_encoder(x[None].to(device))
        sparse, dense = self.sam.prompt_encoder(points=(pr["points"][0].to(device), pr["points"][1].to(device)) if "points" in pr else None,
                                                boxes=pr["boxes"].to(device) if "boxes" in pr else None, masks=None, text_embeds=None)
        low_res, _ = self.sam.mask_decoder(image_embeddings=emb, image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                           sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        # decoder output is a 4x-downsampled grid: upsample to the padded input size, crop the image
        # region, then resize to the original image (the standard SAM post-processing)
        m = F.interpolate(low_res, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        m = m[:, :, offset[0]:offset[0] + resize[0], offset[1]:offset[1] + resize[1]]
        m = F.interpolate(m, img_rgb.shape[:2], mode="bilinear", align_corners=False)
        return torch.sigmoid(m[0, 0]).float().cpu().numpy()


@MODELS.register("sam_med2d_prompt")
def build_sam_prompter(cfg: dict, **kwargs):
    return SamPrompter(cfg["checkpoint"], image_size=int(cfg.get("image_size", 256)), prompt=cfg.get("prompt", "box"),
                       box_jitter=float(cfg.get("box_jitter", 0.0)))
