# CIFAR10-Classification-and-BatchNorm-Analysis

**Neural Network and Deep Learning — Project 2**
**Author:** Huang Jichuan

This repository contains my implementation and experimental analysis for Project 2 of *Neural Network and Deep Learning*. The project has two main parts:

1. **CIFAR-10 image classification**, including controlled CNN ablations, optimizer comparisons, model interpretation, and a high-performance PreAct-ResNet-18 training pipeline.
2. **Batch Normalization analysis**, including VGG-A vs. VGG-A-BN comparison and a deeper study of how BN improves optimization.

The best CIFAR-10 result obtained in this project is **96.45% test accuracy** using **PreAct-ResNet-18 + Mixup + Stochastic Weight Averaging (SWA)**.


---

## Highlights

* Built a configurable **BasicCNN** for systematic ablation experiments.
* Implemented and compared different model widths, loss functions, activations, and optimizers.
* Implemented two custom optimizers: **MyMomentumSGD** and **MyAdamW**.
* Trained a strong **PreAct-ResNet-18** model with Mixup, EMA, AMP, cosine learning-rate scheduling, and SWA.
* Generated multiple interpretation and visualization results, including first-layer filters, feature maps, Grad-CAM, confusion matrix, and loss landscapes.
* Compared **VGG-A** and **VGG-A-BN** under the same training setting.
* Analyzed Batch Normalization from several perspectives: loss landscape, gradient predictiveness, effective smoothness, learning-rate tolerance, noisy BN, internal activation distribution, and BN-vs-Dropout comparison.

---

## Main Results

### Part 1: CIFAR-10 Classification

| Experiment                   |                          Model / Method | Best Accuracy |
| ---------------------------- | --------------------------------------: | ------------: |
| Best controlled CNN ablation | BasicCNN with channels `(96, 192, 384)` |    **92.79%** |
| Final model                  |          PreAct-ResNet-18 + Mixup + SWA |    **96.45%** |

The final model improves the best controlled BasicCNN ablation by **3.66 percentage points**, showing the effectiveness of residual architecture, stronger augmentation, Mixup, and model weight averaging.

### Part 2: Batch Normalization Analysis

| Model             | Best Validation Accuracy | Final Validation Accuracy |
| ----------------- | -----------------------: | ------------------------: |
| VGG-A             |               **74.64%** |                    74.55% |
| VGG-A + BatchNorm |               **77.66%** |                    77.66% |

VGG-A-BN learns faster and reaches better validation performance than the original VGG-A model. The additional optimization diagnostics show that Batch Normalization improves training not only by regularization, but more importantly by making the optimization landscape smoother and more stable.

---

## Project Structure

```text
.
├── requirements.txt
├── README.md
├── shared/
│   ├── utils.py
│   └── data_cifar.py
├── part1_cifar10/
│   ├── models.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── trainer.py
│   ├── visualize.py
│   ├── tta.py
│   ├── run_ablation.py
│   ├── run_best.py
│   └── run_visualize.py
├── part2_batchnorm/
│   ├── compare_bn.py
│   ├── VGG_Loss_Landscape.py
│   ├── plot_landscape.py
│   ├── lr_range_experiment.py
│   ├── noisy_bn_experiment.py
│   ├── ics_visualization.py
│   ├── compare_dropout.py
│   ├── models/
│   ├── data/
│   └── utils/
├── figures/
│   ├── part1/
│   └── part2/

```

---

## Environment Setup

The project is implemented in PyTorch. A CUDA-enabled GPU is recommended for the full training pipeline.

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24
matplotlib>=3.7
tqdm>=4.65
```

---

## Dataset

This project uses **CIFAR-10 only**. The dataset can be downloaded automatically through `torchvision.datasets.CIFAR10` when running the scripts.

CIFAR-10 contains 60,000 color images of size `32 x 32` from 10 classes:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

---

## How to Run

### Part 1: CIFAR-10 Classification

Run the ablation experiments:

```bash
cd part1_cifar10
python run_ablation.py
```

Train the best model:

```bash
python run_best.py
```

Generate visualization results:

```bash
python run_visualize.py
```

### Part 2: Batch Normalization

Run VGG-A vs. VGG-A-BN comparison:

```bash
cd ../part2_batchnorm
python compare_bn.py
```

Run BN optimization-landscape experiments:

```bash
python VGG_Loss_Landscape.py
python plot_landscape.py
```

Run additional BN analysis experiments:

```bash
python lr_range_experiment.py
python noisy_bn_experiment.py
python ics_visualization.py
python compare_dropout.py
```

---

## Visualizations

The repository includes scripts for producing the following visual analysis results:

### CIFAR-10 Model Interpretation

* first convolutional-layer filters;
* intermediate feature maps;
* Grad-CAM visualization;
* confusion matrix;
* 1D loss landscape;
* 2D loss landscape.

### Batch Normalization Analysis

* VGG-A vs. VGG-A-BN training curves;
* loss-landscape bands across different learning rates;
* gradient predictiveness;
* effective beta-smoothness;
* trainable learning-rate range;
* noisy BatchNorm comparison;
* internal activation distribution evolution;
* BN-vs-Dropout comparison.


---

## Trained Weights and Results

```text
Code repository: https://github.com/hank-aa11/CIFAR10-Classification-and-BatchNorm-Analysis
Trained weights and result files: https://drive.google.com/drive/folders/1nXk15pEtj5kobGgNZ14suTcF0Sc5H0PH?usp=drive_link
```

---

## Reproducibility Notes

* The CIFAR-10 dataset is automatically downloaded by the training scripts.
* Random seeds are set in the main scripts for more stable reproduction.
* Full training of the final PreAct-ResNet-18 model may take several hours depending on GPU speed.
* Result logs are saved as JSON files and can be used to reproduce the tables and curves in the report.
* Figures are saved to the corresponding `figures/` directories.

---

## Conclusion

This project demonstrates a complete deep-learning workflow: model construction, controlled ablation, optimizer implementation, high-performance CIFAR-10 training, visualization-based interpretation, and Batch Normalization analysis. The final results show both strong engineering performance and clear conceptual understanding of convolutional networks, optimization, and normalization methods.
