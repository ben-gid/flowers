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
      value: 0.9699
    - type: f1
      value: 0.9498
  - task:
      type: image-classification
    dataset:
      name: Oxford-102 Flowers
      type: oxford-102-flowers
      split: test
    metrics:
    - type: accuracy
      value: 0.9691
    - type: f1
      value: 0.9433
---

# EfficientNetV2-S Flower Classifier

Fine-tuned [`torchvision.models.efficientnet_v2_s`](https://docs.pytorch.org/vision/main/models/efficientnetv2.html) (ImageNet-1K pretrained) for 102-class flower classification on the Oxford-102 Flowers dataset, with the full backbone unfrozen during fine-tuning. Achieves **0.9691 accuracy / 0.9433 macro-F1** on a held-out test split.

> **Note:** this card describes a **new checkpoint**, retrained after a data leak in the training pipeline's split logic was fixed. The weights previously published under this repo were leak-era and were selected on a corrupted validation signal — see [Training history](#training-history). If you pinned an earlier revision, re-download.

**Recommended when raw weight size is the binding constraint.** At 81.8 MB and 20,308,150 parameters it is the smallest of the three models published here. It is *not* the fastest and it is not the most accurate: on GPU at batch size 1 it is the slowest of the three, and its macro-F1 (0.9433) trails both [ConvNeXt-Tiny](https://huggingface.co/bengid/convnext-tiny-flower-classifier) (0.9694) and [ViT-B/16](https://huggingface.co/bengid/vit-flower-classifier) (0.9722). If you are deploying to GPU, ConvNeXt-Tiny has both lower latency and higher accuracy for a modest increase in weight size — pick this model when download size or host memory is what you are optimizing.

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

# preprocessing: Resize((256, 256)) -> CenterCrop(224) -> ToTensor -> Normalize(dataset mean/std)
# note: the resize is to a fixed 256x256 square, NOT shortest-side-256 -- aspect ratio is not preserved
# see src/utils.py:get_transforms() in the training repo for the exact pipeline
# (https://github.com/ben-gid/flowers/blob/main/src/utils.py)
```

## Training Data

[Oxford-102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) — 8,189 images across 102 flower species, downloaded via `torchvision.datasets.Flowers102`. Class-weighted `CrossEntropyLoss` was used to correct for the dataset's uneven per-class image counts.

## Training Procedure

Single-stage fine-tune with a **two-phase backbone unfreeze** callback (`BackboneFinetuning`): the EfficientNetV2 backbone starts frozen (only the classification head trains), then unfreezes at a fixed epoch with its own, lower learning rate and a separate parameter group — unlike this project's original (v1) EfficientNet-B0 model, which only ever unfroze its *last 3 backbone blocks*.

Trained under the `corrected-split-retrain` MLflow experiment, i.e. after the split bug described in [Training history](#training-history) was fixed. Checkpoint selection and early stopping were therefore driven by a genuine held-out validation signal.

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR scheduler | Cosine annealing (T_max=30, eta_min=1e-06) |
| Head LR (before unfreeze) | 0.001 |
| Head LR (after unfreeze) | 0.001 |
| Backbone LR (after unfreeze) | 1e-05 |
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
| **Accuracy** | **0.9699** | **0.9691** |
| **Macro F1** | **0.9498** | **0.9433** |
| Loss | 0.1263 | 0.1110 |

| Property | Value |
|---|---|
| Parameters | 20,308,150 |
| Model size | 81.8 MB |
| Checkpoint size | 245.0 MB |
| Mean latency | 11.33 ms |
| p95 latency | 12.84 ms |

Latency measured on an NVIDIA RTX 5070 (`cuda:0`) at batch size 1, with `torch.cuda.synchronize()` after every forward pass. **This is the number most likely to surprise you.** EfficientNetV2-S has the fewest parameters of the three models here and is nonetheless the slowest on GPU at batch size 1 — its long chain of small depthwise blocks leaves the GPU launch-bound at batch size 1, where ConvNeXt-Tiny's fewer, larger operations do not. Do not infer latency from parameter count. **These figures do not transfer to CPU.** An earlier CPU benchmark of the same architectures put ViT-B/16 at roughly 3x the latency of the other two, while ConvNeXt-Tiny and EfficientNetV2-S landed within a few milliseconds of each other — close enough that two runs of that benchmark disagreed on which was faster. Benchmark on your own serving hardware before treating any of this as a ranking.

Macro F1 sits well below accuracy on test, a wider spread than either other model here. Oxford-102's class counts range from 40 to 258 images, so this gap says errors concentrate in the rare classes — and that this model's long-tail performance is meaningfully weaker than ConvNeXt-Tiny's or ViT-B/16's, more so than the headline accuracy difference alone suggests. Per-class F1 is computed but not currently exported; see `test_per_class_f1` in `src/classifier.py`.

## Training history

An earlier version of this card claimed **0.9997 accuracy / 0.9995 F1**. That was a measurement bug, not a real result. `FlowerDataModule.setup()` in `src/data.py` built all three splits from the same subset:

```python
train_subset, val_subset, test_subset = random_split(full_ds, (0.7, 0.15, 0.15), generator=generator)

self.train_set = SubsetWithTransform(train_subset, self.transform_train)
self.val_set   = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be val_subset
self.test_set  = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be test_subset
```

`val_subset` and `test_subset` were built and then discarded. Validation and test scored the model against the exact images it had trained on — total leakage, not partial. A ~1.0 score was the expected outcome of memorization and carried no information about generalization.

**The weights were never contaminated.** Training only ever read `train_subset`, so no held-out image ever reached a gradient. What the bug corrupted was *measurement* — and, through it, model selection: `ModelCheckpoint` and `EarlyStopping` were both driven by `val_acc`, which was really train accuracy.

**This checkpoint replaces the leak-era one.** It comes from the `corrected-split-retrain` experiment, trained after the fix, so every metric above was measured on genuinely held-out data and the checkpoint was selected on a real validation signal.

One finding worth recording: the retrain recovered less than expected. The leak-era EfficientNetV2-S checkpoint, re-scored honestly on the same held-out data, reached 0.9364 test macro-F1 against this model's 0.9433. The bug inflated the *reported* numbers substantially; it cost the *weights* comparatively little.

## Strengths & Weaknesses

**Strengths:**
- Smallest of the three published models — 81.8 MB of weights and 20,308,150 parameters, the cheapest to download, store, and hold in host memory.
- Convolutional inductive bias generalizes well from a modest fine-tuning dataset (~8k images), needing less data than attention-based architectures to reach its ceiling.
- Its GPU latency penalty does not carry over to CPU, where it runs neck-and-neck with ConvNeXt-Tiny — so a CPU-only deployment pays much less for choosing it than the GPU figures suggest.

**Weaknesses:**
- Least accurate of the three: 0.9433 test macro-F1 against ConvNeXt-Tiny's 0.9694 and ViT-B/16's 0.9722. This is the largest gap in the lineup, not a rounding difference.
- Slowest of the three on GPU at batch size 1 despite having the fewest parameters — see the note under [Evaluation](#evaluation). Its earlier cards claimed a latency advantage that only holds on CPU.
- Weakest on rare classes specifically: its accuracy-to-macro-F1 spread is the widest of the three, so the deficit concentrates in exactly the long-tail classes an imbalanced dataset makes hardest.
- EfficientNetV2's published training recipe (progressive resizing, adaptive regularization) is more sensitive to schedule/hyperparameter choices than a straightforward ConvNeXt or ViT fine-tune; deviating far from a tuned recipe can cost more accuracy than it would for the other two.

## Limitations

- **Closed-set, single-label**: trained on exactly 102 Oxford flower species; will confidently misclassify any other flower species, non-flower image, or multi-flower image into one of the 102 known classes — there is no out-of-distribution rejection.
- **Fixed input pipeline**: expects a 224×224 input produced by resizing to a fixed 256×256 square and center-cropping. The resize does not preserve aspect ratio, so non-square images are distorted before the crop, and off-center subjects can be cropped out of frame.
- **No adversarial robustness or calibration guarantees** — confidence scores are not calibrated probabilities.
- Reported metrics are on held-out Oxford-102 val/test splits; real-world images (different lighting, backgrounds, camera quality) may perform worse.
- **Macro F1 (0.9433) is the number to trust, not accuracy (0.9691)** — Oxford-102 is class-imbalanced (40–258 images per class), and the gap between the two means errors concentrate in rare classes. If your use case cares about the long tail, budget for the F1 figure, and consider ConvNeXt-Tiny, whose long-tail deficit is smaller.

## Intended Use

**Intended uses:**
- Flower species identification within the 102 Oxford-102 classes (gardening/botany apps, educational tools, dataset labeling).
- Backend model for this project's v2 `/classify` API endpoint when download size or host memory is the constraint.

**Out-of-scope uses:**
- General-purpose plant, object, or scene classification outside the 102 trained species.
- Medical, toxicity, or safety-related plant identification.
- Any use where a wrong classification has safety or financial consequences without human review.

## Model Comparison

All figures are **test-split** metrics on held-out data. The three current models were trained on the corrected splits; the two v1 baselines are listed for historical context.

| Model | Test Acc | Test F1 | Params | Size (MB) | Best For |
|---|---|---|---|---|---|
| [SimpleCNN (scratch)](https://huggingface.co/bengid/flower-classifier/blob/main/flower_model_weights.pth) | ~0.63 | - | - | - | historical baseline only |
| [EfficientNet-B0 (v1, partial unfreeze)](https://huggingface.co/bengid/flower-classifier/blob/main/ft_EfficientNet-B0.pth) | >0.93 | - | - | - | historical baseline only |
| [ViT-B/16](https://huggingface.co/bengid/vit-flower-classifier) | 0.9837 | 0.9722 | 85,877,094 | 343.5 | maximum macro-F1 |
| [ConvNeXt-Tiny](https://huggingface.co/bengid/convnext-tiny-flower-classifier) | 0.9821 | 0.9694 | 27,898,566 | 111.6 | default serving choice |
| **[EfficientNetV2-S](https://huggingface.co/bengid/efficientnetv2-s-flower-classifier) (this model)** | **0.9691** | **0.9433** | **20,308,150** | **81.8** | smallest weights |

The two v1 models were trained by the older, pre-Lightning pipeline (`api/app/v1/flowers/train_scratch.py`), which split train/val/test correctly — their numbers were never affected by the leak and are directly comparable to the corrected figures above.

### Why the earlier models underperformed

- **SimpleCNN (scratch)** was trained from randomly initialized weights with no ImageNet pretraining, on a 6-block custom CNN — too little capacity and too little prior visual knowledge to learn 102 fine-grained flower classes from ~8k images alone.
- **EfficientNet-B0 (v1)** started from ImageNet-pretrained weights but only ever unfroze its *last 3 backbone blocks* during fine-tuning (see this project's root `README.md` for the original two-stage recipe) — the earlier backbone layers, tuned for general ImageNet features, never adapted to flower-specific low/mid-level features.
- All three current models unfreeze the *entire* backbone during fine-tuning, which drives the improvement from ~93% to ~96–98% test accuracy.

Note that this gain is **~3–5 points, not the ~7 points the leaked metrics implied**. Full-backbone unfreezing is a real improvement over partial unfreezing, but a far more modest one than a jump from 93% to "100%" suggested.

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
