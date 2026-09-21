"""Dataset "brisc2025": BRISC manifest rows -> MedPLIB samples (MedPLIB's own SAM/CLIP
preprocessing and LLaVA-v1 tokenisation). Training samples mix joint / classification-only /
segmentation-only prompts, use an all-zero mask for non-tumorous images and joint image+mask
augmentation; inference samples carry the prompt with every candidate answer."""
import copy
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DistributedSampler, Sampler

from .. import paths
from ..registry import DATASETS
from .task import TaskSpec

paths.add_medplib_to_syspath()
from datasets.LazySupervisedDataset import LazySupervisedDataset, preprocess, preprocess_multimodal  # noqa: E402
from datasets.DataCollatorForSupervisedDataset import DataCollatorForSupervisedDataset  # noqa: E402
from model.segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402
from utils.utils import IGNORE_INDEX  # noqa: E402


def load_image_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"cannot read {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, shape, bin_thresh: int = 128) -> np.ndarray:
    """Binary uint8 mask; an empty/None path gives the derived all-zero mask."""
    if not path:
        return np.zeros(tuple(shape[:2]), dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert m is not None, f"cannot read {path}"
    assert m.shape == tuple(shape[:2]), f"mask {path} {m.shape} != image {tuple(shape[:2])}"
    return (m >= bin_thresh).astype(np.uint8)


def augment_pair(img: np.ndarray, mask: np.ndarray, rng: random.Random, a: dict):
    """Geometry shared by image and mask (mask: nearest neighbour); photometry image-only."""
    h, w = img.shape[:2]
    if rng.random() < a.get("hflip_p", 0.5):
        img = np.ascontiguousarray(img[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    rot = float(a.get("rotate_deg", 10))
    s_lo, s_hi = a.get("scale", [0.9, 1.1])
    shift = float(a.get("shift_frac", 0.05))
    angle = rng.uniform(-rot, rot)
    scale = rng.uniform(s_lo, s_hi)
    tx = rng.uniform(-shift, shift) * w
    ty = rng.uniform(-shift, shift) * h
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bc = float(a.get("brightness_contrast", 0.1))
    out = img.astype(np.float32) * rng.uniform(1 - bc, 1 + bc) + rng.uniform(-bc, bc) * 255.0
    if rng.random() < a.get("noise_p", 0.2):
        out = out + np.random.default_rng(rng.getrandbits(32)).normal(0.0, float(a.get("noise_sigma", 4.0)), size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), mask


class BriscMedPLIBDataset(LazySupervisedDataset):
    """One manifest row -> the dict MedPLIB's DataCollatorForSupervisedDataset expects."""

    def __init__(self, df: pd.DataFrame, tokenizer, data_args, spec: TaskSpec, root, sam_img_size: int = 256,
                 clip_img_size: int = 336, augmentation: Optional[dict] = None, task_mix: Optional[Dict[str, float]] = None,
                 fixed_task: Optional[str] = None, seed: int = 42, mask_bin_thresh: int = 128):
        Dataset.__init__(self)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.spec = spec
        self.root = Path(root)
        self.sam_img_size = sam_img_size
        self.transform = ResizeLongestSide(sam_img_size)
        self.clip_img_size = clip_img_size
        self.transform_clip = ResizeLongestSide(clip_img_size)
        self.rows = df.reset_index(drop=True).to_dict("records")
        self.augmentation = augmentation if (augmentation and augmentation.get("enabled", True)) else None
        self.task_mix = task_mix or spec.task_mix
        self.fixed_task = fixed_task
        self.seed = seed
        self.epoch = 0
        self.mask_bin_thresh = mask_bin_thresh

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def mask_path(self, row):
        return (self.root / row["mask_path"]) if isinstance(row.get("mask_path"), str) and row["mask_path"] else None

    # ---- image side -------------------------------------------------------------
    def encode_image(self, img_rgb: np.ndarray):
        image_resize = self.transform.apply_image(img_rgb)
        resize = image_resize.shape[:2]
        image_sam = self.preprocess(torch.from_numpy(image_resize).permute(2, 0, 1).contiguous(), self.sam_img_size)
        image_clip = self.transform_clip.apply_image(img_rgb)
        image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_img_size, normalize=False)
        image_clip = self.data_args.image_processor.preprocess(image_clip, return_tensors="pt")["pixel_values"][0]
        return image_sam, image_clip, resize

    # ---- text side --------------------------------------------------------------
    def encode_text(self, prompt: str, answer: str):
        conv = [{"from": "human", "value": "<image>\n" + prompt}, {"from": "gpt", "value": answer}]
        sources = preprocess_multimodal(copy.deepcopy([conv]), self.data_args)
        d = preprocess(sources, self.tokenizer, has_image=True)
        return d["input_ids"][0], d["labels"][0], d["conversations"], d["question"], d["gt"]

    def sample_task(self, i: int) -> str:
        if self.fixed_task:
            return self.fixed_task
        rng = random.Random((self.seed * 1000003 + self.epoch) * 1000003 + i)
        r, acc = rng.random(), 0.0
        for task, p in self.task_mix.items():
            acc += p
            if r < acc:
                return task
        return list(self.task_mix)[-1]

    def __getitem__(self, key):
        # key is either an index or (index, task) from TaskBatchSampler
        i, task = (key if isinstance(key, tuple) else (key, None))
        row = self.rows[i]
        task = task or self.sample_task(i)
        img = load_image_rgb(self.root / row["image_path"])
        mask = load_mask(self.mask_path(row), img.shape, self.mask_bin_thresh)
        if self.augmentation is not None:
            rng = random.Random((self.seed * 7919 + self.epoch) * 7919 + i)
            img, mask = augment_pair(img, mask, rng, self.augmentation)
        image_sam, image_clip, resize = self.encode_image(img)
        input_ids, labels, conversations, question, gt = self.encode_text(self.spec.prompt_for(task), self.spec.answer_for(task, row["class_key"]))
        d = dict(input_ids=input_ids, labels=labels, conversations=conversations, question=question, gt=gt,
                 image_clip=image_clip, image_sam=image_sam, region_masks=[],
                 image_path=str(self.root / row["image_path"]), inference=False, tokenizer=self.tokenizer, answer_type=None,
                 image_id=row["image_id"], class_key=row["class_key"], task=task)
        if task in ("joint", "seg"):
            m = torch.from_numpy(mask).float()
            d["masks"] = [m]
            d["label"] = [torch.ones(m.shape[0], m.shape[1]) * self.ignore_label]
            d["resize"] = [resize]
        else:
            d["masks"] = []
        return d


def collate_train(batch):
    return DataCollatorForSupervisedDataset(batch, inference=False)


class TaskBatchSampler(Sampler):
    """Batches of (index, task) with ONE task per micro-batch, drawn from the task mix with a
    (seed, epoch) RNG so every rank draws the same task sequence. With DeepSpeed ZeRO every rank
    must build the same computation graph per micro-step; mixing tasks inside a batch or across
    ranks would leave the mask branch without gradients on some ranks and hang the reduction."""

    def __init__(self, sampler: DistributedSampler, batch_size: int, task_mix: Dict[str, float],
                 seed: int = 42, fixed_task: Optional[str] = None):
        self.sampler = sampler
        self.batch_size = batch_size
        self.task_mix = task_mix
        self.seed = seed
        self.fixed_task = fixed_task
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def _draw(self, rng: random.Random) -> str:
        if self.fixed_task:
            return self.fixed_task
        r, acc = rng.random(), 0.0
        for task, p in self.task_mix.items():
            acc += p
            if r < acc:
                return task
        return list(self.task_mix)[-1]

    def __iter__(self):
        rng = random.Random(self.seed * 1000003 + self.epoch)
        batch = []
        for idx in self.sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                task = self._draw(rng)
                yield [(i, task) for i in batch]
                batch = []

    def __len__(self):
        return len(self.sampler) // self.batch_size


# ----------------------------------------------------------------------------------------
# Inference: one image -> the prompt with each candidate answer
# ----------------------------------------------------------------------------------------
class BriscInferenceDataset(Dataset):
    """Per image, tensors for the prompt with every candidate answer ("{label} <SEG>") so one
    batched forward gives class log-likelihoods and the <SEG> hidden state of every candidate.
    Positions scored for the class likelihood are the answer tokens excluding <SEG> and </s>."""

    def __init__(self, df: pd.DataFrame, tokenizer, data_args, spec: TaskSpec, root, sam_img_size: int = 256,
                 clip_img_size: int = 336, prompt: Optional[str] = None, with_seg: bool = True, mask_bin_thresh: int = 128,
                 seg_only: bool = False):
        self.base = BriscMedPLIBDataset(df, tokenizer, data_args, spec, root, sam_img_size, clip_img_size,
                                        augmentation=None, fixed_task="joint", mask_bin_thresh=mask_bin_thresh)
        self.spec = spec
        self.seg_only = seg_only  # segmentation-only control: single answer "<SEG>", no class scoring
        self.prompt = prompt or spec.prompt_for("seg" if seg_only else ("joint" if with_seg else "cls"))
        self.candidates = [""] if seg_only else spec.candidates()
        self.with_seg = with_seg
        self.seg_token_idx = tokenizer(spec.seg_token, add_special_tokens=False).input_ids[0]
        self.eos = tokenizer.eos_token_id

    def __len__(self):
        return len(self.base.rows)

    def __getitem__(self, i):
        row = self.base.rows[i]
        img = load_image_rgb(self.base.root / row["image_path"])
        mask = load_mask(self.base.mask_path(row), img.shape, self.base.mask_bin_thresh)
        image_sam, image_clip, resize = self.base.encode_image(img)
        ids, labs, score_masks = [], [], []
        for cand in self.candidates:
            answer = (self.spec.seg_token if self.seg_only else f"{cand} {self.spec.seg_token}") if self.with_seg else cand
            input_ids, labels, *_ = self.base.encode_text(self.prompt, answer)
            score = (labels != IGNORE_INDEX) & (labels != self.seg_token_idx) & (labels != self.eos)
            ids.append(input_ids)
            labs.append(labels)
            score_masks.append(score)
        return dict(image_id=row["image_id"], class_key=row["class_key"], class_idx=self.spec.idx(row["class_key"]),
                    plane=row["plane"], subset=row["subset"], mask_type=row["mask_type"],
                    image_path=str(self.base.root / row["image_path"]),
                    image_sam=image_sam, image_clip=image_clip, resize=resize,
                    gt_mask=torch.from_numpy(mask), orig_size=tuple(img.shape[:2]),
                    input_ids=ids, labels=labs, score_masks=score_masks)


def collate_infer(batch):
    """Flatten (image, candidate) pairs; `owner[j]` is the image index of flattened row j."""
    all_ids, all_labels, all_score, owner = [], [], [], []
    for b_idx, b in enumerate(batch):
        for ids, labs, sm in zip(b["input_ids"], b["labels"], b["score_masks"]):
            all_ids.append(ids)
            all_labels.append(labs)
            all_score.append(sm)
            owner.append(b_idx)
    keys = ("image_id", "class_key", "class_idx", "plane", "subset", "mask_type", "image_path", "orig_size")
    return dict(
        meta=[{k: b[k] for k in keys} for b in batch],
        images=torch.stack([b["image_sam"] for b in batch], 0),
        images_clip=torch.stack([b["image_clip"] for b in batch], 0),
        resize_list=[b["resize"] for b in batch],
        gt_masks=[b["gt_mask"] for b in batch],
        input_ids=all_ids, labels=all_labels, score_masks=all_score, owner=torch.tensor(owner),
        n_candidates=len(batch[0]["input_ids"]),
    )


# ----------------------------------------------------------------------------------------
# plain tensors for the conventional baselines (U-Net, CNN classifier, prompted SAM)
# ----------------------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class BriscTensorDataset(Dataset):
    """Square-resized image (ImageNet-normalised, 3xSxS) with either the binary mask (task "seg")
    or the class index (task "cls"); the original image/mask are kept for evaluation."""

    def __init__(self, df: pd.DataFrame, spec: TaskSpec, root, task: str = "seg", size: int = 512,
                 augmentation: Optional[dict] = None, seed: int = 42, mask_bin_thresh: int = 128, keep_original: bool = False):
        self.rows = df.reset_index(drop=True).to_dict("records")
        self.spec, self.root, self.task, self.size = spec, Path(root), task, size
        self.augmentation = augmentation if (augmentation and augmentation.get("enabled", True)) else None
        self.seed, self.epoch, self.mask_bin_thresh, self.keep_original = seed, 0, mask_bin_thresh, keep_original

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def mask_path(self, row):
        return (self.root / row["mask_path"]) if isinstance(row.get("mask_path"), str) and row["mask_path"] else None

    def __getitem__(self, i):
        row = self.rows[i]
        img = load_image_rgb(self.root / row["image_path"])
        mask = load_mask(self.mask_path(row), img.shape, self.mask_bin_thresh)
        if self.augmentation is not None:
            img, mask = augment_pair(img, mask, random.Random((self.seed * 7919 + self.epoch) * 7919 + i), self.augmentation)
        x = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA if img.shape[0] > self.size else cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        d = dict(image=torch.from_numpy(x).permute(2, 0, 1).contiguous(), image_id=row["image_id"],
                 class_idx=self.spec.idx(row["class_key"]), class_key=row["class_key"], plane=row["plane"],
                 subset=row["subset"], mask_type=row["mask_type"], orig_size=tuple(img.shape[:2]))
        if self.task == "seg":
            d["mask"] = torch.from_numpy(cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)).float()
        if self.keep_original:
            d["orig_image"] = img
            d["orig_mask"] = mask
        return d


def collate_tensor(batch):
    out = {k: [b[k] for b in batch] for k in batch[0]}
    out["image"] = torch.stack(out["image"], 0)
    if "mask" in out:
        out["mask"] = torch.stack(out["mask"], 0)
    out["class_idx"] = torch.tensor(out["class_idx"])
    return out


# ----------------------------------------------------------------------------------------
# registered builder
# ----------------------------------------------------------------------------------------
def load_manifest(cfg: dict, subset: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(cfg["manifest"])
    if subset and subset != "all":
        df = df[df.subset == subset]
    return df.sort_values("image_id").reset_index(drop=True)


@DATASETS.register("brisc2025")
def build_brisc(cfg: dict, mode: str, subset: str, tokenizer=None, data_args=None, model_cfg: Optional[dict] = None, seed: int = 42,
                augment: Optional[bool] = None, fixed_task: Optional[str] = None, max_samples: int = 0,
                shard: tuple = (0, 1), prompt: Optional[str] = None, with_seg: bool = True, df: Optional[pd.DataFrame] = None,
                seg_only: bool = False, size: int = 512, keep_original: bool = False):
    """mode: "train" -> BriscMedPLIBDataset (task mix, augmentation); "infer" -> BriscInferenceDataset;
    "seg" / "cls" -> BriscTensorDataset for the conventional baselines (tumour images only for "seg").
    shard=(rank, world) selects every world-th image (used by inference and validation)."""
    spec = TaskSpec.from_cfg(cfg)
    model_cfg = model_cfg or {}
    if df is None:
        df = load_manifest(cfg, subset)
        if mode == "seg":
            df = df[df.mask_type == "expert"].reset_index(drop=True)
        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=seed).sort_values("image_id").reset_index(drop=True)
        df = df.iloc[shard[0]::shard[1]].reset_index(drop=True)
    if mode in ("seg", "cls"):
        aug = cfg.get("augmentation", {})
        if augment is not None:
            aug = dict(aug, enabled=bool(augment))
        return BriscTensorDataset(df, spec, cfg["root"], task=mode, size=size, augmentation=aug, seed=seed,
                                  mask_bin_thresh=int(cfg.get("mask_bin_threshold", 128)), keep_original=keep_original)
    common = dict(tokenizer=tokenizer, data_args=data_args, spec=spec, root=cfg["root"],
                  sam_img_size=int(model_cfg.get("sam_img_size", 256)), clip_img_size=int(model_cfg.get("clip_img_size", 336)),
                  mask_bin_thresh=int(cfg.get("mask_bin_threshold", 128)))
    if mode == "train":
        aug = cfg.get("augmentation", {})
        if augment is not None:
            aug = dict(aug, enabled=bool(augment))
        return BriscMedPLIBDataset(df, augmentation=aug, task_mix=spec.task_mix, fixed_task=fixed_task, seed=seed, **common)
    if mode == "infer":
        p = spec.prompt_for(prompt) if prompt in spec.prompts else prompt
        return BriscInferenceDataset(df, prompt=p, with_seg=with_seg, seg_only=seg_only, **common)
    raise ValueError(mode)
