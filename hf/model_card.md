---
license: cc-by-nc-4.0
library_name: timm
pipeline_tag: image-classification
tags:
  - dermoscopy
  - skin-lesion
  - ham10000
  - convnext
datasets:
  - HAM10000
---

# skin-lesion-convnext-v1 (final_v1)

7-class dermoscopic skin lesion classifier (akiec, bcc, bkl, df, mel, nv, vasc) trained on HAM10000 as a
student learning project. **Research and education only. Not a medical device and not for diagnosis.**

Code, experiment history and limitations: https://github.com/yunus345/skin-cancer-detection
Demo: https://huggingface.co/spaces/emreozkn/skin-lesion-demo

## Model

- Two ConvNeXt-Tiny models (timm `convnext_tiny.fb_in22k_ft_in1k_384`, 384 px), same recipe, seeds 42 and 2.
- Each is temperature-scaled on the validation set (T = 0.93 and 0.89); the prediction is the equal-weight
  mean of the calibrated probabilities.
- Two operating points: `normal` (argmax) and `tarama` (screening: predict melanoma when its mean probability
  is ≥ 0.28).
- Preprocessing: RGB, Gray-World color constancy, resize to 384×384, ImageNet normalization. It must match
  `src.training.train.build_transforms(384, train=False)` in the code repository.

Files: `seed42.pt`, `seed2.pt` (state_dict only, loadable with `torch.load(weights_only=True)`), `config.json`
(architecture, class order, temperatures, thresholds, MD5s, measured metrics). Use the `v1` tag.

## Results (single measurement, 2026-11-12)

| Set | macro F1 | balanced accuracy | melanoma recall | macro AUC |
|---|---|---|---|---|
| ISIC 2018 Task 3 test (1512, external) | 0.795 | 0.796 | 0.737 | 0.972 |
| HAM10000 internal test (986) | 0.829 | 0.832 | 0.720 | 0.973 |

The external number is the one to report. The internal test set was looked at several times during
development and is optimistic. The external set had been used once before, for an earlier model.

## Limitations

- Trained only on dermoscopy images from two centres (Austria, Australia), mostly light skin. Performance on
  phone photos, other devices or darker skin types is unknown and likely worse.
- HAM10000 has no patient IDs; the split is lesion-level, not patient-level.
- Even confident predictions can be wrong: on validation, 8 of 102 melanomas were missed with confidence ≥ 0.85.
- Full list: "Sınırlamalar" in the GitHub README.

## Data and license

Trained on HAM10000 (Tschandl, Rosendahl, Kittler, 2018, https://arxiv.org/abs/1803.10417),
© ViDIR Group, Medical University of Vienna, licensed CC BY-NC 4.0. These weights are released under the same
non-commercial license.
