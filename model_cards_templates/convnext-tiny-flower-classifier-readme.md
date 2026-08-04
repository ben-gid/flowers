---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
tags:
- image-classification
- flowers
- oxford-102
- torchvision
- convnext
- transfer-learning
metrics:
- accuracy
- f1
model-index:
- name: ConvNeXt-Tiny Flower Classifier
  results:
  - task:
      type: image-classification
    dataset:
      name: Oxford-102 Flowers
      type: oxford-102-flowers
      split: validation
    metrics:
    - type: accuracy
      value: $val_acc
    - type: f1
      value: $val_f1
  - task:
      type: image-classification
    dataset:
      name: Oxford-102 Flowers
      type: oxford-102-flowers
      split: test
    metrics:
    - type: accuracy
      value: $test_acc
    - type: f1
      value: $test_f1
---

# ConvNeXt-Tiny Flower Classifier

Fine-tuned [`torchvision.models.convnext_tiny`](https://docs.pytorch.org/vision/main/models/convnext.html) (ImageNet-1K pretrained) for 102-class flower classification on the Oxford-102 Flowers dataset, with the full backbone unfrozen during fine-tuning. Achieves **$test_acc accuracy / $test_f1 macro-F1** on a held-out test split.

**Recommended as the default choice for this project.** It lands within a fraction of a point of [ViT-B/16](https://huggingface.co/bengid/vit-flower-classifier)'s macro-F1 ($test_f1 vs $vit_test_f1) at $model_size_mb MB against ViT's $vit_model_size_mb MB, and it is the fastest of the three at batch size 1 on GPU. If you are serving this model, start here; reach for ViT-B/16 only when you want the last fraction of a point of accuracy and can pay roughly three times the weights.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import models

weights_path = hf_hub_download(repo_id="bengid/convnext-tiny-flower-classifier", filename="convnext-tiny-flower-classifier.safetensors")

model = models.convnext_tiny(weights=None)
model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 102)
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

Single-stage fine-tune with a **two-phase backbone unfreeze** callback (`BackboneFinetuning`): the ConvNeXt backbone starts frozen (only the classification head trains), then unfreezes at a fixed epoch with its own, lower learning rate and a separate parameter group — unlike this project's original (v1) EfficientNet-B0 model, which only ever unfroze its *last 3 backbone blocks*.

Trained under the `corrected-split-retrain` MLflow experiment, i.e. after the split bug described in [Training history](#training-history) was fixed. Checkpoint selection and early stopping were therefore driven by a genuine held-out validation signal.

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | $optimizer |
| LR scheduler | $scheduler |
| Head LR (before unfreeze) | $lr_head_stage_1 |
| Head LR (after unfreeze) | $lr_head_stage_2 |
| Backbone LR (after unfreeze) | $lr_backbone |
| Unfreeze epoch | $unfreeze_at_epoch |
| Max epochs | $max_epochs |
| Batch size | $batch_size |
| Effective batch size | $effective_batch_size |
| Gradient accumulation | $accumulate_grad_batches |
| Precision | $precision |
| Weight decay | $weight_decay |
| Early stopping patience | $early_stopping_patience |

## Evaluation

Oxford-102 split 70/15/15 by `random_split(seed=42)` → 5,733 train / 1,228 val / 1,228 test. Val and test are disjoint from train and from each other.

| Metric | Validation | Test |
|---|---|---|
| **Accuracy** | **$val_acc** | **$test_acc** |
| **Macro F1** | **$val_f1** | **$test_f1** |
| Loss | $val_loss | $test_loss |

| Property | Value |
|---|---|
| Parameters | $num_parameters |
| Model size | $model_size_mb MB |
| Checkpoint size | $checkpoint_size_mb MB |
| Mean latency | $latency_ms_mean ms |
| p95 latency | $latency_ms_p95 ms |

Latency measured on an NVIDIA RTX 5070 (`cuda:0`) at batch size 1, with `torch.cuda.synchronize()` after every forward pass. **These figures do not transfer to CPU.** An earlier CPU benchmark of the same architectures put ViT-B/16 at roughly 3x the latency of the other two, while ConvNeXt-Tiny and EfficientNetV2-S landed within a few milliseconds of each other — close enough that two runs of that benchmark disagreed on which was faster. Benchmark on your own serving hardware before treating any of this as a ranking.

Macro F1 sits below accuracy on both splits, which is the signature of uneven per-class performance on a dataset whose class counts range from 40 to 258 images — the rare classes are where the misses are. Per-class F1 is computed but not currently exported; see `test_per_class_f1` in `src/classifier.py`.

## Training history

This project's earlier published checkpoints reported near-perfect scores (~1.0 accuracy / ~1.0 F1). Those were a measurement bug, not a real result. `FlowerDataModule.setup()` in `src/data.py` built all three splits from the same subset:

```python
train_subset, val_subset, test_subset = random_split(full_ds, (0.7, 0.15, 0.15), generator=generator)

self.train_set = SubsetWithTransform(train_subset, self.transform_train)
self.val_set   = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be val_subset
self.test_set  = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be test_subset
```

`val_subset` and `test_subset` were built and then discarded, so validation and test scored the model against the exact images it had trained on. Weights were never contaminated — training only ever read `train_subset` — but *measurement* was, and through it model selection: `ModelCheckpoint` and `EarlyStopping` were both driven by `val_acc`, which was really train accuracy.

**This checkpoint is not affected.** It comes from the `corrected-split-retrain` experiment, trained after the fix, so every metric above was measured on genuinely held-out data and the checkpoint was selected on a real validation signal.

One finding worth recording: retraining on the corrected splits recovered very little. The leak-era ConvNeXt-Tiny checkpoint, re-scored honestly on the same held-out data, reaches $convnext_leak_test_f1 test macro-F1 against this model's $test_f1. The bug inflated the *reported* numbers substantially; it cost the *weights* almost nothing.

## Strengths & Weaknesses

**Strengths:**
- Best accuracy-to-size balance of the three: near-ViT macro-F1 ($test_f1 vs $vit_test_f1) at roughly a third of the weights ($model_size_mb MB vs $vit_model_size_mb MB). EfficientNetV2-S is smaller still in absolute terms, but gives up far more accuracy to get there.
- Fastest of the three at batch size 1 on GPU ($latency_ms_mean ms), despite having more parameters than EfficientNetV2-S — it runs fewer, larger operations per forward pass, where EfficientNetV2-S's long chain of small depthwise blocks leaves the GPU launch-bound.
- Modern convolutional design retains the inductive bias that helps on a modest (~8k image) fine-tuning set, without ViT's data appetite.
- Small enough to ship inside this project's Docker image comfortably.

**Weaknesses:**
- Still measurably behind ViT-B/16 on macro-F1 ($test_f1 vs $vit_test_f1). The gap is small, and ViT's slightly narrower accuracy-to-F1 spread suggests it sits in the long tail — but the margin there is thin enough not to lean on.
- Roughly a third larger in parameter count than EfficientNetV2-S ($num_parameters vs $effnet_num_parameters), so if raw download size is the binding constraint rather than throughput, EfficientNetV2-S is smaller.
- Like every model here, tuned on a single fine-tuning recipe rather than a hyperparameter search — the reported figures are one good run, not a demonstrated ceiling.

## Limitations

- **Closed-set, single-label**: trained on exactly 102 Oxford flower species; will confidently misclassify any other flower species, non-flower image, or multi-flower image into one of the 102 known classes — there is no out-of-distribution rejection.
- **Fixed input pipeline**: expects a 224×224 input produced by resizing to a fixed 256×256 square and center-cropping. The resize does not preserve aspect ratio, so non-square images are distorted before the crop, and off-center subjects can be cropped out of frame.
- **No adversarial robustness or calibration guarantees** — confidence scores are not calibrated probabilities.
- Reported metrics are on held-out Oxford-102 val/test splits; real-world images (different lighting, backgrounds, camera quality) may perform worse.
- **Macro F1 ($test_f1) is the number to trust, not accuracy ($test_acc)** — Oxford-102 is class-imbalanced (40–258 images per class), and the gap between the two means errors concentrate in rare classes. If your use case cares about the long tail, budget for the F1 figure.

## Intended Use

**Intended uses:**
- Flower species identification within the 102 Oxford-102 classes (gardening/botany apps, educational tools, dataset labeling).
- Default backend model for this project's v2 `/classify` API endpoint.

**Out-of-scope uses:**
- General-purpose plant, object, or scene classification outside the 102 trained species.
- Medical, toxicity, or safety-related plant identification.
- Any use where a wrong classification has safety or financial consequences without human review.

## Model Comparison

All figures are **test-split** metrics on held-out data. The three current models were trained on the corrected splits; the two v1 baselines are listed for historical context.

$comparison_table

The two v1 models were trained by the older, pre-Lightning pipeline (`api/app/v1/flowers/train_scratch.py`), which split train/val/test correctly — their numbers were never affected by the leak and are directly comparable to the corrected figures above.

### Why the earlier models underperformed

- **SimpleCNN (scratch)** was trained from randomly initialized weights with no ImageNet pretraining, on a 6-block custom CNN — too little capacity and too little prior visual knowledge to learn 102 fine-grained flower classes from ~8k images alone.
- **EfficientNet-B0 (v1)** started from ImageNet-pretrained weights but only ever unfroze its *last 3 backbone blocks* during fine-tuning (see this project's root `README.md` for the original two-stage recipe) — the earlier backbone layers, tuned for general ImageNet features, never adapted to flower-specific low/mid-level features.
- All three current models unfreeze the *entire* backbone during fine-tuning, which drives the improvement from ~93% to ~96–98% test accuracy.

Note that this gain is **~3–5 points, not the ~7 points the leaked metrics implied**. Full-backbone unfreezing is a real improvement over partial unfreezing, but a far more modest one than a jump from 93% to "100%" suggested.

## License

Apache 2.0, consistent with this project's license.

## Citation

**Base model (ConvNeXt):**
```bibtex
@inproceedings{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11976--11986},
  year={2022}
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
