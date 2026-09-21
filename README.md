# MedPLIB brain tumour classification + segmentation (BRISC 2025)

Course project: fine-tune **MedPLIB-7b-2e** (Llama-7B MoE + CLIP-L/336 + SAM-Med2D-B) so one
model answers the fixed prompt with a class (glioma / meningioma / pituitary tumor /
non-tumorous) and a `<SEG>` token that is decoded into a tumour mask.

## Layout

```
configs/
  paths.yaml                 server paths (models, data, outputs)
  model/                     medplib_7b_2e.yaml + baselines (efficientnet_b0, unet_resnet34, sam_med2d_box)
  dataset/brisc2025.yaml     classes/answers, prompts, task mix, augmentation
  train/                     stage_a.yaml, stage_b.yaml, baseline_*.yaml, control_*.yaml, overfit64.yaml
  infer/                     default.yaml (threshold, bootstrap), sam_med2d_box.yaml
medplib_bt/
  registry.py, config.py     MODELS / DATASETS registries, YAML configs with --set overrides
  models/medplib.py          "medplib_moe": load checkpoint, LoRA, trainable modules, adapters
  models/baselines/          "timm_classifier", "smp_unet", "sam_med2d_prompt"
  datasets/brisc.py          "brisc2025": training / inference datasets
  datasets/manifest.py       build data/manifest.csv from the BRISC folders
  engine/inference.py        log-likelihood classification + <SEG> mask decoding, metrics
  train.py, infer.py, evaluate.py        MedPLIB training, inference, metrics
  train_baseline.py, infer_baseline.py   baselines, same prediction layout as infer.py
  aggregate.py, consistency.py           result tables, class/mask decision consistency
  describe.py, demo_app.py               free-text generation, interactive demo
scripts/                     setup_env.sh, download_data.sh, download_models.sh, run_*.sh
third_party/MedPLIB/         vendored upstream code (unmodified, media stripped)
```

## Setup (server)

```bash
scripts/setup_env.sh                 # conda env `medplib` (torch 2.1.2, transformers 4.31, deepspeed 0.13.1, peft 0.10)
scripts/download_data.sh             # BRISC 2025 from Kaggle -> data/brisc2025
scripts/download_models.sh           # MedPLIB-7b-2e, CLIP-L/336, SAM-Med2D-B -> /mnt/data/peilin/HF_MODEL
python scripts/make_safetensors_index.py   # index for the safetensors shards (the HF repo only ships a .bin index)
python -m medplib_bt.datasets.manifest   # data/manifest.csv
```

## Train / evaluate

```bash
scripts/run_train.sh 0,1,2,3,4,5,6,7 configs/train/overfit64.yaml overfit64          # pre-flight
scripts/run_train.sh 0,1,2,3,4,5,6,7 configs/train/stage_a.yaml stageA_seed42
scripts/run_train.sh 0,1,2,3,4,5,6,7 configs/train/stage_b.yaml stageB_seed42 --init-from outputs/runs/stageA_seed42/final
scripts/run_infer.sh 0,1,2,3 outputs/preds/stageB_seed42 --adapter outputs/runs/stageB_seed42/final   # test set + metrics
scripts/run_infer.sh 0,1,2,3 outputs/preds/zeroshot --no-adapter                                     # pretrained control
scripts/run_pipeline.sh 0,1,2,3,4,5,6,7 42                                                            # stage A -> B -> test, one seed

# baselines (one GPU each), scored by the same evaluate.py
scripts/run_baseline.sh 0 configs/train/baseline_efficientnet.yaml effnet_b0_seed42
scripts/run_baseline.sh 1 configs/train/baseline_unet.yaml unet_r34_seed42
scripts/run_nnunet.sh 2 nnunet_2d
scripts/run_baseline_infer.sh 3 configs/infer/sam_med2d_box.yaml outputs/preds/sam_med2d_oracle_box

# tables
python -m medplib_bt.aggregate --seeds outputs/preds/stageB_seed42 --zeroshot outputs/preds/zeroshot \
    --baseline "EfficientNet-B0=outputs/preds/effnet_b0_seed42" --baseline "U-Net R34=outputs/preds/unet_r34_seed42" \
    --baseline "nnU-Net 2D=outputs/preds/nnunet_2d" --baseline "SAM-Med2D oracle box=outputs/preds/sam_med2d_oracle_box" --out outputs/results
python -m medplib_bt.consistency --model "MedPLIB=outputs/preds/stageB_seed42" \
    --pipeline "EfficientNet-B0 + U-Net=outputs/preds/effnet_b0_seed42,outputs/preds/unet_r34_seed42" \
    --gated "EfficientNet-B0 gate + U-Net=outputs/preds/effnet_b0_seed42,outputs/preds/unet_r34_seed42" --out outputs/results
python -m medplib_bt.demo_app --adapter outputs/runs/stageB_seed42/final --port 7860
```

Any config value can be overridden on the command line, e.g. `--set seed=123 --set model.lora.r=8`.
Results land in `outputs/preds/<run>/eval/` (summary_metrics.json, confusion_matrix.png,
dice_by_class.png, qualitative_grid.png, failure_cases.html, masks/).

## Recipe

* Stage A (1 epoch, 43.1M trainable): expert router, image projector, `<SEG>` projector and mask
  decoder; the language model, CLIP tower and SAM-Med2D encoder are frozen.
  Loss 1.0 CE + 2.0 BCE + 0.5 Dice.
* Stage B (12 epochs, 286.8M of 11.75B trainable): the stage-A modules, plus LoRA r=16 / alpha=32 /
  dropout 0.05 on attention and both experts' MLPs in all 32 layers, plus the SAM-Med2D encoder
  adapters. Loss 1.0 CE + 2.0 BCE + 2.0 Dice + 1.0 IoU.
* Samples: 50 % class + `<SEG>` + mask, 25 % class only, 25 % `<SEG>` + mask; non-tumorous
  images use an all-zero mask.
* AdamW, LoRA lr 2e-5, new modules 1e-4, wd 0.01, 3 % warm-up + cosine; bf16, DeepSpeed ZeRO-2,
  micro-batch 2 x grad-accum 4 x GPUs (effective batch 64).
* Inference: class = argmax sequence log-likelihood of the four answers; mask from the `<SEG>`
  state of that answer, threshold 0.5; the mask branch always runs (no hard gating).
* Test: official BRISC test split (1000 images, 860 with expert masks). Reported separately:
  4-class metrics, Dice/IoU on tumour images and over all images, false-positive masks on non-tumour images,
  class/mask conflicts; bootstrap 95 % CIs.

## Results (official BRISC test split, 1000 images, mask threshold 0.5)

Every model is trained on the official 5000-image training split and scored by the same
`evaluate.py` on the same 1000 test images (`outputs/results/controls.csv`).

| Model | Accuracy | Macro-F1 | Tumour Dice (860) | All-image Dice (1000) | Tumour IoU | Dice glioma / meningioma / pituitary | False masks on non-tumour |
|---|---|---|---|---|---|---|---|
| **MedPLIB-7b-2e, fine-tuned (ours)** | **0.980** | **0.982** | 0.854 | **0.874** | 0.774 | 0.763 / 0.919 / 0.865 | **0.0%** |
| EfficientNet-B0 (classification only) | 0.993 | 0.994 | - | - | - | - | - |
| U-Net ResNet-34 (segmentation only) | - | - | **0.881** | 0.812 | **0.818** | 0.784 / 0.952 / 0.892 | 61.4% |
| nnU-Net 2D, 100 epochs (segmentation only) | - | - | 0.877 | 0.807 | 0.813 | 0.777 / 0.950 / 0.886 | 62.1% |
| SAM-Med2D-B with boxes from the expert masks (upper bound) | - | - | 0.788 | 0.818 | 0.686 | 0.646 / 0.880 / 0.815 | 0.0% (no box on non-tumour scans) |
| MedPLIB-7b-2e, no fine-tuning | 0.294 | 0.123 | 0.040 | 0.085 | 0.027 | 0.055 / 0.067 / 0.000 | 63.6% |

The fine-tuned model's Dice by lesion size is 0.728 / 0.859 / 0.892 (small <= 0.5 % of the image,
medium, large > 1.5 %); balanced accuracy 0.982; class/mask conflict rate 0.

The tumour-only Dice ignores the 140 non-tumour test scans. The dedicated segmenters are trained on
tumour images only and paint a mask on ~60% of those scans, so on the full test set (a non-tumour
scan scores 1 for an empty mask and 0 for any mask) the fine-tuned MedPLIB leads: 0.874 vs 0.812
(U-Net) and 0.807 (nnU-Net). EfficientNet-B0 does include the no-tumour class, so its accuracy is
comparable as is. Pixel-pooled weighted mIoU in the BRISC paper's definition: ours 0.785, U-Net
0.823, nnU-Net 0.819 (paper: U-Net 0.757, best model 0.806).

### Decision consistency (`medplib_bt/consistency.py`)

One image, one decision: does the class answer agree with the mask, is a healthy scan left alone,
and are both answers right at once (class correct and mask correct: Dice >= 0.5 on a tumour image,
empty on a healthy one)? A pipeline pairs EfficientNet-B0's answer with a segmenter's mask on the
same image; "gated" drops the mask when the classifier says non-tumorous. `outputs/results/consistency.csv`.

| System | Contradictions | Mask on healthy scan | Tumour with no mask | All-image Dice | Both answers right |
|---|---|---|---|---|---|
| MedPLIB, fine-tuned (joint) | 0.0% | 0.0% | 0.0% | 0.874 | 93.9% |
| EfficientNet-B0 + U-Net R34, no gate | 9.0% | 61.4% | 0.5% | 0.812 | 87.0% |
| EfficientNet-B0 gate + U-Net R34 | 0.4% | 0.0% | 0.5% | 0.898 | 95.6% |
| EfficientNet-B0 gate + nnU-Net 2D | 0.5% | 0.0% | 0.6% | 0.894 | 95.3% |

The joint model never contradicts itself; an ungated pipeline does on 9% of images (86 healthy
scans painted, 4 gliomas named but not segmented). A one-line class gate removes the first kind and
the gated pair then edges the joint model on both-answers-right (95.6 vs 93.9%, overlapping bootstrap CIs), so
consistency is built in but not unique to the joint model.

### Language interface (`medplib_bt/describe.py`)

Greedy free-text generation on 7 test images (`outputs/report/describe_*/answers.json`).
The fine-tuned model writes the trained format ("glioma <SEG>") and names the class 7/7; the mask decoded from the
generated <SEG> matches the scored inference. A yes/no "is there a tumor" prompt it never saw is
answered correctly 7/7 (incl. "No" on a healthy scan), but free descriptions collapse to one word
and an untrained imaging-plane question is mostly wrong (2/7; pretrained model 4/7, which answers in
its pretraining vocabulary, e.g. "non-enhancing tumor in the head and neck region"). Narrow
fine-tuning shrank the language interface; mixing in templated report text would be the fix.
