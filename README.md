# Classification-of-Food-101-datasets-using-Deep-learning
Deep analysis of Food-101 paper and development of Deep learning models 
# Applied Deep Learning Lab Portfolio — Food-101 Classification

**Module:** Deep Learning (MOD006565) | **Level:** 6 | **Student:** Dheeraj Kodwani

A deep learning portfolio exploring food image classification on the [Food-101 dataset](https://www.tensorflow.org/datasets/catalog/food101) (101 categories, 101,000 images). The project progresses from a paper-replication baseline through to custom CNN architectures and unsupervised clustering.

---

## Results summary

| Model | Top-1 Accuracy | Top-5 Accuracy | Loss |
|---|---|---|---|
| GoogLeNet (paper baseline) | 72.3% | 91.0% | — |
| GoogLeNet (Part A — replicated) | 70.89% | — | 1.1276 |
| ResNet50 | 59.73% | — | 1.6248 |
| Vision Transformer (ViT-B/16) | 68.15% | — | 1.2845 |
| **EfficientNet-B2 + custom CNN (Part B)** | **76.9%** | **93.72%** | **0.8827** |

---

## Project structure

```
├── deep_learning.ipynb                  # Main notebook — all parts
├── Deep_Learning_Coursework_Report.pdf  # Technical report
└── README.md
```

---

## What's covered

### Part A — Baseline model (GoogLeNet)
- Replication of the GoogLeNet architecture from the Food-101 paper (Bossard et al., 2014)
- SGD optimiser (lr=0.01, momentum=0.9), 0.7 dropout, ImageNet normalisation
- Dataset split: 68,175 train / 7,575 val / 25,250 test
- Achieved **70.89% test accuracy**, closely matching the paper's 72.3%

### Part B — Custom CNN (EfficientNet-B2)
- EfficientNet-B2 pretrained backbone with two custom Conv2D refinement layers added
- Two-stage fine-tuning: frozen backbone (8 epochs) → top 3 blocks unfrozen (12 epochs)
- Data augmentation: random crop, horizontal flip, rotation, colour jitter, affine transforms
- AdamW optimiser with weight decay, label smoothing, early stopping
- Achieved **76.9% Top-1 / 93.72% Top-5** — outperforming the original paper
- Also trained ResNet50 and ViT-B/16 for comparison

### Part C — Self-Organising Map (SOM)
- 256-dimensional feature vectors extracted from EfficientNet-B2
- 15×15 SOM grid, sigma=1.0, lr=0.5, 5,000 iterations on 5,000 samples
- Visualised U-Matrix, feature map, dominant class per node, and sample distribution
- Demonstrates topology-preserving unsupervised clustering of food image features without labels

---

## Setup & requirements

```bash
pip install torch torchvision scikit-learn matplotlib minisom tqdm
```

GPU recommended (trained on NVIDIA RTX 3060 Laptop GPU). The notebook runs end-to-end — simply run all cells in order.

```python
# Dataset is downloaded automatically via torchvision
import torchvision.datasets as datasets
datasets.Food101(root='./data', split='train', download=True)
```

---

## Key techniques

- Transfer learning with ImageNet pretrained weights
- Two-stage fine-tuning (head training → selective unfreezing)
- Data augmentation pipeline for food image variability
- Multi-metric evaluation: accuracy, precision, recall, F1, sensitivity, specificity, Top-5
- Unsupervised feature clustering with Self-Organising Maps

---

## Reference

Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 — Mining Discriminative Components with Random Forests. *ECCV 2014*.
