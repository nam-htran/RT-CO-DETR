# RT-CO-DETR: Boosting Real-Time Object Detection with Knowledge Distillation

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project enhances the performance of the real-time object detector **RT-DETR** on the **TACO (Trash Annotations in Context)** dataset by leveraging **Knowledge Distillation**. We use a powerful but slower teacher model, **Conditional DETR**, to guide the training of a faster student model, **RT-DETR-L**, significantly improving its accuracy without compromising its real-time inference speed.

## 📋 Table of Contents

1.  [Overview](#1-overview)
2.  [Key Features](#2-key-features)
3.  [Performance Benchmark](#3-performance-benchmark)
4.  [Methodology](#4-methodology)
5.  [Project Structure](#5-project-structure)
6.  [Setup and Usage](#6-setup-and-usage)
7.  [License](#7-license)
8.  [Acknowledgements](#8-acknowledgements)

## 1. Overview

The primary goal of this project is to improve the accuracy of the RT-DETR model for trash detection on the TACO dataset. Standard fine-tuning can be limited, so we employ a teacher-student knowledge distillation strategy.

-   **Student Model:** **RT-DETR-L**, a fast and efficient real-time object detector.
-   **Teacher Model:** **Conditional DETR** (`microsoft/conditional-detr-resnet-50`), a larger, more accurate, but slower detector.
-   **Core Idea:** The student model learns not only from the ground-truth labels but also from the rich, "soft" knowledge provided by the teacher model. This includes learning from the teacher's intermediate feature representations and its final prediction distributions.

The final distilled model is benchmarked against a standard fine-tuned RT-DETR baseline and a powerful YOLOv11l baseline to demonstrate the effectiveness of our approach.

## 2. Key Features

-   **Knowledge Distillation Pipeline:** Implements both feature-level and prediction-level distillation.
    -   **Feature Distillation:** The student learns to mimic the feature maps from the teacher's ResNet-50 backbone using an MSE loss.
    -   **Prediction Distillation:** The student learns the teacher's output distribution for both classification (using KL Divergence loss) and bounding box regression (using L1 loss).
-   **Automated Training Orchestration:** A master script (`train.py`) manages the entire workflow, from data preparation to distillation and final fine-tuning experiments.
-   **Environment-Aware Configuration:** The central `config.py` file automatically detects whether the code is running in a local or Kaggle environment and adjusts data paths accordingly.
-   **Multi-GPU Support:** Leverages `torchrun` for efficient, distributed training on multiple GPUs.
-   **Comprehensive Benchmarking:** Compares three models (Distilled RT-DETR, Baseline RT-DETR, and Baseline YOLO) across accuracy (mAP), complexity (Parameters, FLOPs), and inference speed.

## 3. Performance Benchmark

All models were evaluated on the TACO validation set. The benchmark was conducted on a **Tesla T4** GPU.

| Model | mAP@.50-.95 | mAP@.50 | Speed (ms) | Params (M) | FLOPs (G) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RT-DETR (Distilled)** | **0.2610** | **0.3100** | 59.06 | 40.92 | 136.06 |
| RT-DETR (Baseline) | 0.2390 | 0.3000 | 59.86 | 40.92 | 136.06 |
| YOLOv11l (Baseline) | 0.2566 | 0.2960 | **28.74** | **25.36** | **87.53** |

### Analysis of Results

1.  **Knowledge Distillation is Highly Effective:**
    -   The **Distilled RT-DETR** significantly outperforms the **Baseline RT-DETR** in accuracy, achieving a **9.2% relative improvement** in mAP@.50-.95 (`0.2610` vs. `0.2390`).
    -   This accuracy gain comes at **no extra cost** to inference speed or model complexity, as both RT-DETR models have virtually identical parameters and FLOPs. This is the core success of the project.

2.  **Comparison with a Strong Baseline (YOLO):**
    -   The **YOLOv11l** model is significantly faster and more efficient (2x faster, ~62% of the parameters). This highlights YOLO's strength in speed-optimized architectures.
    -   However, the **Distilled RT-DETR** achieves **higher accuracy** across both mAP metrics, demonstrating its superior capability in learning complex representations, especially after being guided by a powerful teacher.

**Conclusion:** The project successfully demonstrates that knowledge distillation is a powerful technique to enhance a real-time detector like RT-DETR, allowing it to achieve state-of-the-art accuracy that can even surpass other highly optimized models like YOLO, albeit with a trade-off in inference speed.

## 4. Methodology

The project follows a three-stage pipeline: Data Preparation, Knowledge Distillation, and Comparative Fine-tuning.
<img width="2776" height="2712" alt="Mermaid Chart - Create complex, visual diagrams with text -2025-11-01-072938" src="https://github.com/user-attachments/assets/69c689b1-6f36-478b-b281-86a6acf72cb7" />

## 5. Project Structure

The codebase is organized into a modular and reusable structure.

```
.
├── config.py               # Central configuration for all paths and settings
├── train.py                # Master script to orchestrate the entire training pipeline
├── requirements.txt        # Project dependencies
├── scripts/                # Helper scripts for data preparation and config generation
│   ├── convert_coco_to_yolo.py
│   └── generate_rtdetr_configs.py
├── src/
│   ├── distillation/       # Core logic for the knowledge distillation process
│   │   ├── trainer_codetr.py # Main distillation training loop
│   │   ├── models.py       # Teacher model wrapper
│   │   └── dataset.py      # Custom dataset for distillation
│   └── finetune/           # Scripts for fine-tuning baseline models
│       ├── trainer_yolo.py
│       └── ...
├── rtdetr/                 # Submodule containing the RT-DETR source code
├── templates/              # Template files for generating experiment configs
└── output/                 # Default directory for all generated outputs
    ├── DISTILL/            # Checkpoints from the distillation phase
    ├── FINETUNE_BASELINE/  # Checkpoints for the baseline RT-DETR model
    ├── FINETUNE_DISTILLED/ # Checkpoints for the distilled RT-DETR model
    └── YOLO/               # Checkpoints and data for the YOLO model
```

## 6. Setup and Usage

### Prerequisites

-   Python 3.11+
-   PyTorch 2.0+ and `torchvision`
-   Numpy < 2.0
-   Git
-   An NVIDIA GPU with CUDA for training

### Step 1: Clone the Repository and Install Dependencies

```bash
# Clone this repository
git clone https://github.com/nam-htran/RT-CO-DETR.git
cd RT-CO-DETR

# Install required packages
pip install -r requirements.txt
```
The RT-DETR submodule will be automatically cloned by the training script if it's not found.

### Step 2: Prepare the Dataset

Place your `processed_taco_coco` dataset according to the structure defined in `config.py`. For a local setup, the expected structure is:

```
.
├── data_input/
│   └── processed_taco_coco/
│       ├── train2017/
│       ├── val2017/
│       └── annotations/
│           ├── instances_train2017.json
│           └── instances_val2017.json
└── ... (rest of the project files)
```
*Note: If running on Kaggle, the script automatically uses the `/kaggle/input/` path.*

### Step 3: Run the Training Pipeline

The `train.py` script orchestrates the entire process. You can run all steps at once or individually.

#### Option A: Run the Full Pipeline (Recommended)

This command will execute data preparation, knowledge distillation, and all fine-tuning experiments sequentially.

```bash
python train.py --all
```

#### Option B: Run Steps Individually

This is useful for debugging or re-running specific parts of the pipeline.

```bash
# 1. Prepare data formats (creates YOLO data)
python train.py --prepare-data

# 2. Run knowledge distillation to create the distilled checkpoint
# This step is GPU-intensive and may take a long time.
python train.py --distill

# 3. Run fine-tuning experiments for all three models
python train.py --finetune
```

### Step 4: Run the Final Benchmark

After all models have been trained, a `best.pth` (or `best.pt`) checkpoint will be available in their respective `output/` subdirectories. Run the benchmark script to generate the final comparison table and plot.

```bash
# (This script is provided in the repository)
python final_benchmark.py
```
The results will be printed to the console and saved to `benchmark_output/benchmark_summary.csv` and `benchmark_output/benchmark_plot.png`.

## 7. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 8. Acknowledgements

-   This project is built upon the official implementation of [RT-DETR](https://github.com/lyuwenyu/RT-DETR).
-   The teacher model, [Conditional-DETR](https://huggingface.co/microsoft/conditional-detr-resnet-50), is provided by Microsoft via the Hugging Face Hub.
-   The baseline model, [YOLO](https://github.com/ultralytics/ultralytics), is provided by Ultralytics.
-   The [TACO dataset](https://tacodataset.org/) is the foundation for this trash detection task.
```
