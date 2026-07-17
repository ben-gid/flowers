---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
tags:
- image-classification
- flowers
- oxford-102
- torchvision
- efficientnet
- transfer-learning
metrics:
- accuracy
- f1
model-index:
- name: EfficientNetV2-S Flower Classifier
  results:
  - task:
      type: image-classification
    dataset:
      name: Oxford-102 Flowers
      type: oxford-102-flowers
      split: validation
    metrics:
    - type: accuracy
      value: 0.9682
    - type: f1
      value: 0.9468
  - task:
      type: image-classification
    dataset:
      name: Oxford-102 Flowers
      type: oxford-102-flowers
      split: test
    metrics:
    - type: accuracy
      value: 0.9642
    - type: f1
      value: 0.9364
---

# EfficientNetV2-S Flower Classifier

Fine-tuned [`torchvision.models.efficientnet_v2_s`](https://docs.pytorch.org/vision/main/models/efficientnetv2.html) (ImageNet-1K pretrained) for 102-class flower classification on the Oxford-102 Flowers dataset, with the full backbone unfrozen during fine-tuning. Achieves **0.9642 accuracy / 0.9364 macro-F1** on a held-out test split.

> **Note:** earlier versions of this card reported 0.9997 accuracy / 0.9995 F1. Those numbers were produced by a data leak in the training pipeline's split logic and have been corrected — see [Metrics correction](#metrics-correction--earlier-scores-were-invalid) below. The weights are unaffected; only the measurement was.

**Recommended when** you need strong accuracy at a fraction of the size and latency — still the default choice for serving. At ~82MB and ~30ms mean latency it is ~4x smaller and ~3x faster than [ViT-B/16 Flower Classifier](vit-b16-readme.md), which costs it ~2.7pp of test macro-F1.

That trade-off is a real one now. Under the earlier (leaked) metrics both models looked tied at ~1.0, making this model a free win; on corrected data ViT-B/16 is measurably more accurate. This card still recommends EfficientNetV2-S as the serving default — 4x the size and 3x the latency is a steep price for 2.7pp, and most deployments should pay the accuracy rather than the resources — but if you are running offline batch work with no latency budget, ViT-B/16 is now the better pick on merit, not a rounding error.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import models

weights_path = hf_hub_download(repo_id="bengid/efficientnetv2-s-flower-classifier", filename="efficientnetv2-s-flower-classifier.safetensors")

model = models.efficientnet_v2_s(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 102)
model.load_state_dict(load_file(weights_path, device="cpu"))
model.eval()

# preprocessing: resize(256) -> center-crop(224) -> normalize with dataset mean/std
# see src/utils.py:get_transforms() in the training repo for the exact pipeline
# (https://github.com/ben-gid/flowers/blob/main/src/utils.py)
```

## Training Data

[Oxford-102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) — 8,189 images across 102 flower species, downloaded via `torchvision.datasets.Flowers102`. Class-weighted `CrossEntropyLoss` was used to correct for the dataset's uneven per-class image counts.

## Training Procedure

Single-stage fine-tune with a **two-phase backbone unfreeze** callback (`BackboneFinetuning`): the EfficientNetV2 backbone starts frozen (only the classification head trains), then unfreezes at a fixed epoch with its own, lower learning rate and a separate parameter group — unlike this project's original (v1) EfficientNet-B0 model, which only ever unfroze its *last 3 backbone blocks*.

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR scheduler | Cosine annealing (T_max=30, eta_min=1e-06) |
| Head LR (before unfreeze) | 1e-3 |
| Head LR (after unfreeze) | 1e-3 |
| Backbone LR (after unfreeze) | 1e-5 |
| Unfreeze epoch | 5 |
| Max epochs | 30 |
| Batch size | 32 |
| Effective batch size | 32 |
| Gradient accumulation | 1 |
| Precision | 16-mixed |
| Weight decay | 0.01 |
| Early stopping patience | 5 |

## Evaluation

Oxford-102 split 70/15/15 by `random_split(seed=42)` → 5,733 train / 1,228 val / 1,228 test. Val and test are disjoint from train and from each other.

| Metric | Validation | Test |
|---|---|---|
| **Accuracy** | **0.9682** | **0.9642** |
| **Macro F1** | **0.9468** | **0.9364** |
| Loss | 0.1246 | 0.1206 |

| Property | Value |
|---|---|
| Parameters | 20,308,150 |
| Model size | 81.8 MB |
| Checkpoint size | 245.0 MB |
| Mean latency | 29.6 ms |
| p95 latency | 30.9 ms |

Latency measured on ryzen 5600x cpu at batch size 1.

Macro F1 sits ~2.8pp below accuracy on test, a wider spread than ViT-B/16's ~1.6pp. Oxford-102's class counts range from 40 to 258 images, so this gap says errors concentrate in the rare classes — and that this model's long-tail performance is meaningfully weaker than ViT's, more so than the headline accuracy difference alone suggests. Per-class F1 is computed but not currently exported; see `test_per_class_f1` in `src/classifier.py`.

## Metrics correction — earlier scores were invalid

An earlier version of this card claimed **0.9997 accuracy / 0.9995 F1**. That was a measurement bug, not a real result. `FlowerDataModule.setup()` in `src/data.py` built all three splits from the same subset:

```python
train_subset, val_subset, test_subset = random_split(full_ds, (0.7, 0.15, 0.15), generator=generator)

self.train_set = SubsetWithTransform(train_subset, self.transform_train)
self.val_set   = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be val_subset
self.test_set  = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be test_subset
```

`val_subset` and `test_subset` were built and then discarded. Validation and test scored the model against the exact images it had trained on — total leakage, not partial. A ~1.0 score was the expected outcome of memorization and carried no information about generalization.

**The weights are not contaminated.** Training only ever read `train_subset`, so no held-out image ever reached a gradient. What the bug corrupted was *measurement* — and, through it, model selection: `ModelCheckpoint` and `EarlyStopping` were both driven by `val_acc`, which was really train accuracy. This checkpoint was therefore chosen on a signal that couldn't distinguish good epochs from overfit ones, and is unlikely to be the best epoch of its run.

The split logic is now fixed. The metrics above come from re-scoring this same published checkpoint against genuinely held-out data via `notebooks/reevaluate_published_models.ipynb`.

**A full retrain on the corrected splits is planned.** Expect these numbers to improve: this is a leak-era checkpoint re-measured honestly, not a model whose training was ever guided by a real validation signal. Early stopping never had a reason to fire at the right time, and no hyperparameter choice in this run was validated against held-out data.

## Strengths & Weaknesses

**Strengths:**
- ~4x smaller (81.8MB vs 343.5MB) and ~3x faster (29.6ms vs 86.5ms mean latency) than ViT-B/16 for ~2.7pp of test macro-F1 — still the best accuracy-per-cost tradeoff of the four models trained, and the reason this remains the serving default.
- Convolutional inductive bias generalizes well from a modest fine-tuning dataset (~8k images), needing less data than attention-based architectures to reach its ceiling.
- Small enough to fit comfortably in the Docker image and serve cheaply (this is the architecture family the v2 API's original `helpers.py` defaulted to).

**Weaknesses:**
- Genuinely less accurate than ViT-B/16 — ~1.5pp test accuracy and ~2.7pp test macro-F1 behind. The earlier claim that this gap was "negligible" / "within noise" was an artifact of the leaked metrics, which pinned both models at ~1.0 and hid the difference entirely.
- Weaker on rare classes specifically: its accuracy-to-macro-F1 spread (~2.8pp) is nearly double ViT's (~1.6pp), so the deficit is concentrated in exactly the long-tail classes an imbalanced dataset makes hardest.
- EfficientNetV2's published training recipe (progressive resizing, adaptive regularization) is more sensitive to schedule/hyperparameter choices than a straightforward ViT fine-tune; deviating far from a tuned recipe can cost more accuracy than it would for ViT.
- Selected by a checkpoint callback that was reading a leaked metric (see above), so this is likely not the best epoch this recipe can produce.

## Limitations

- **Closed-set, single-label**: trained on exactly 102 Oxford flower species; will confidently misclassify any other flower species, non-flower image, or multi-flower image into one of the 102 known classes — there is no out-of-distribution rejection.
- **Fixed input pipeline**: expects a 224×224 center-cropped, normalized input (resize-then-crop). Unusual aspect ratios or off-center subjects can crop the flower out of frame.
- **No adversarial robustness or calibration guarantees** — confidence scores are not calibrated probabilities.
- Reported metrics are on held-out Oxford-102 val/test splits; real-world images (different lighting, backgrounds, camera quality) may perform worse.
- **Macro F1 (~0.936) is the number to trust, not accuracy (~0.964)** — Oxford-102 is class-imbalanced (40–258 images per class), and the ~2.8pp gap between the two means errors concentrate in rare classes. If your use case cares about the long tail, budget for the F1 figure, and consider ViT-B/16, whose long-tail deficit is smaller.
- This checkpoint predates the split fix and was selected on a leaked validation signal — a retrain is planned (see [Metrics correction](#metrics-correction--earlier-scores-were-invalid)).

## Intended Use

**Intended uses:**
- Flower species identification within the 102 Oxford-102 classes (gardening/botany apps, educational tools, dataset labeling).
- Default backend model for this project's v2 `/classify` API endpoint — chosen for its accuracy/latency/size balance.

**Out-of-scope uses:**
- General-purpose plant, object, or scene classification outside the 102 trained species.
- Medical, toxicity, or safety-related plant identification.
- Any use where a wrong classification has safety or financial consequences without human review.

## Model Comparison

This project trained four models in total, in this order. All figures are **test-split** accuracy on held-out data:

| Model | Test Acc | Test F1 | Params | Size (MB) | Best For |
|---|---|---|---|---|---|
| [SimpleCNN (scratch)](https://huggingface.co/bengid/flower-classifier/blob/main/flower_model_weights.pth) | ~0.63 | - | - | - | historical baseline only |
| [EfficientNet-B0 (v1, partial unfreeze)](https://huggingface.co/bengid/flower-classifier/blob/main/ft_EfficientNet-B0.pth) | >0.93 | - | - | - | historical baseline only |
| **EfficientNetV2-S (this model)** | **0.9642** | **0.9364** | 20,308,150 | 81.8 | efficient production serving |
| [ViT-B/16](https://huggingface.co/bengid/vit-flower-classifier) | 0.9796 | 0.9637 | 85,877,094 | 343.5 | maximum accuracy |

The two v1 models were trained by the older, pre-Lightning pipeline (`api/app/v1/flowers/train_scratch.py`), which split train/val/test correctly — their numbers were never affected by the leak and are directly comparable to the corrected v2 figures above.

### Why the earlier models underperformed

- **SimpleCNN (scratch)** was trained from randomly initialized weights with no ImageNet pretraining, on a 6-block custom CNN — too little capacity and too little prior visual knowledge to learn 102 fine-grained flower classes from ~8k images alone.
- **EfficientNet-B0 (v1)** started from ImageNet-pretrained weights but only ever unfroze its *last 3 backbone blocks* during fine-tuning (see this project's root `README.md` for the original two-stage recipe) — the earlier backbone layers, tuned for general ImageNet features, never adapted to flower-specific low/mid-level features.
- Both **EfficientNetV2-S** (this model) and **ViT-B/16** unfreeze the *entire* backbone during fine-tuning, which drives the improvement from ~93% to ~96–98% test accuracy.

Note that this last gain is **~3–5 points, not the ~7 points the leaked metrics implied**. Full-backbone unfreezing is a real improvement over partial unfreezing, but a far more modest one than a jump from 93% to "100%" suggested. The leaked numbers made a good architectural decision look like a spectacular one.

## License

Apache 2.0, consistent with this project's license.

## Citation

**Base model (EfficientNetV2):**
```bibtex
@inproceedings{tan2021efficientnetv2,
  title={EfficientNetV2: Smaller Models and Faster Training},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```

**Training dataset:**
```bibtex
@inproceedings{nilsback2008automated,
  title={Automated flower classification over a large number of classes},
  author={Nilsback, Maria-Elena and Zisserman, Andrew},
  booktitle={2008 Sixth Indian Conference on Computer Vision, Graphics \& Image Processing},
  pages={722--729},
  year={2008},
  organization={IEEE}
}
```
