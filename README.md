# RT-CO-DETR: Boosting Real-Time Object Detection with Knowledge Distillation

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Reproducible-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project enhances the performance of the real-time object detector **RT-DETR** on the **TACO (Trash Annotations in Context)** dataset by leveraging **Knowledge Distillation**. We use a powerful but slower teacher model, **Conditional DETR**, to guide the training of a faster student model, **RT-DETR-L**, significantly improving its accuracy without compromising its real-time inference speed.

The entire training and benchmarking pipeline is designed to be executed within a **Kaggle Notebook** environment.

## 📋 Table of Contents

1.  [Overview](#1-overview)
2.  [Key Features](#2-key-features)
3.  [Performance Benchmark](#3-performance-benchmark)
4.  [Methodology](#4-methodology)
5.  [Project Structure](#5-project-structure)
6.  [Usage and Reproduction](#6-usage-and-reproduction)
7.  [License](#7-license)
8.  [Acknowledgements](#8-acknowledgements)

## 1. Overview

The primary goal is to improve the accuracy of the RT-DETR model for trash detection on the TACO dataset. Standard fine-tuning can be limited, so we employ a teacher-student knowledge distillation strategy.

-   **Student Model:** **RT-DETR-L**, a fast and efficient real-time object detector.
-   **Teacher Model:** **Conditional DETR** (`microsoft/conditional-detr-resnet-50`), a larger, more accurate, but slower detector.
-   **Core Idea:** The student model learns not only from ground-truth labels but also from the rich knowledge provided by the teacher model, including intermediate feature representations and final prediction distributions.

The final distilled model is benchmarked against a standard fine-tuned RT-DETR and a powerful YOLOv11l baseline to demonstrate the effectiveness of our approach.

## 2. Key Features

-   **Knowledge Distillation Pipeline:** Implements both feature-level and prediction-level distillation.
-   **Automated Training Orchestration:** A master script (`train.py`) manages the entire workflow, from data preparation to distillation and final fine-tuning.
-   **Environment-Aware Configuration:** A central `config.py` automatically detects local vs. Kaggle environments.
-   **Multi-GPU Support:** Leverages `torchrun` for efficient, distributed training.
-   **Comprehensive Benchmarking:** A dedicated script within our Kaggle notebook compares models across accuracy (mAP), complexity (Parameters, FLOPs), and inference speed.

## 3. Performance Benchmark

All models were evaluated on the TACO validation set. The benchmark was conducted on a **Tesla T4** GPU. The full analysis, including the generation of these results, can be found in the analysis notebook mentioned in the "How to Reproduce" section.

| Model | mAP@.50-.95 | mAP@.50 | Speed (ms) | Params (M) | FLOPs (G) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RT-DETR (Distilled)** | **0.2610** | **0.3100** | 59.06 | 40.92 | 136.06 |
| RT-DETR (Baseline) | 0.2390 | 0.3000 | 59.86 | 40.92 | 136.06 |
| YOLOv11l (Baseline) | 0.2566 | 0.2960 | **28.74** | **25.36** | **87.53** |

### Analysis of Results

1.  **Knowledge Distillation is Highly Effective:** The **Distilled RT-DETR** significantly outperforms the **Baseline RT-DETR**, achieving a **9.2% relative improvement** in mAP@.50-.95. This accuracy gain comes at **no extra cost** to inference speed or model complexity.

2.  **Comparison with a Strong Baseline (YOLO):** While the **YOLOv11l** is significantly faster and more efficient, the **Distilled RT-DETR** achieves **higher accuracy**, demonstrating its superior capability in learning complex representations after being guided by a powerful teacher.

**Conclusion:** Knowledge distillation is a powerful technique to enhance a real-time detector like RT-DETR, allowing it to achieve state-of-the-art accuracy.

## 4. Methodology

The project follows a three-stage pipeline: Data Preparation, Knowledge Distillation, and Comparative Fine-tuning.
<img width="2776" height="2712" alt="Mermaid Chart - Create complex, visual diagrams with text -2025-11-01-072938" src="https://github.com/user-attachments/assets/f042d817-f103-4226-9be8-f441ce9a5471" />


## 5. Project Structure

The codebase is organized into a modular and reusable structure.

```
.
├── config.py               # Central configuration for all paths and settings
├── train.py                # Master script to orchestrate the entire training pipeline
├── requirements.txt        # Project dependencies
├── benchmark/
│   └── aftertrain-analysis-rt-codetr.ipynb # Kaggle notebook for final analysis
├── scripts/                # Helper scripts for data prep and config generation
├── src/
│   ├── distillation/       # Core logic for knowledge distillation
│   └── finetune/           # Scripts for fine-tuning baselines
├── rtdetr/                 # Submodule with the RT-DETR source code
├── templates/              # Template files for experiment configs
└── output/                 # Default directory for all generated outputs
```

## 6. Usage and Reproduction

There are two ways to engage with this project:

-   **Option 1 (Recommended):** Reproduce the final benchmark results quickly using our prepared Kaggle Notebook.
-   **Option 2 (Advanced):** Run the entire training pipeline from scratch on your own machine.

---

### Option 1: Reproduce Benchmark Results (Kaggle)

This is the easiest way to verify our findings. The entire analysis is encapsulated in a single, self-contained Kaggle Notebook.

#### Step 1: Access the Analysis Notebook

The notebook containing the complete benchmark code is located in the repository at:
-   `./benchmark/aftertrain-analysis-rt-codetr.ipynb`

This notebook is the single source of truth for reproducing the results. All generated benchmark outputs (`.csv`, `.png`) can also be found in this folder.

#### Step 2: Configure the Notebook Environment

When you open the notebook, attach the following Kaggle datasets as input:

-   **TACO Dataset:** `/kaggle/input/dsp-pre-final`
-   **Pre-trained Models:** `/kaggle/input/rt-co-detr-trained`

#### Step 3: Run All Cells

Execute all cells in the notebook sequentially. The notebook is designed to be fully automated:
1.  It installs all necessary dependencies.
2.  It clones the required `RT-CO-DETR` repository.
3.  It runs the `final_benchmark.py` script, which performs all evaluation tasks.
4.  Finally, it generates and displays the summary table and comparison plot, which are saved to `/kaggle/working/benchmark_output/`.

---

### Option 2: Run the Full Training Pipeline (Local/Advanced)

Follow these steps if you want to run the entire training process from scratch on your own machine.

#### Prerequisites

-   Python 3.11+
-   PyTorch 2.0+ and `torchvision`
-   Numpy < 2.0
-   Git
-   An NVIDIA GPU with CUDA for training
-   All other dependencies are listed in `requirements.txt`.

#### Step 1: Clone the Repository and Install Dependencies

```bash
# Clone this repository
git clone https://github.com/nam-htran/RT-CO-DETR.git
cd RT-CO-DETR

# Install required packages
pip install -r requirements.txt
```
The `train.py` script will automatically clone the required `RT-DETR` submodule if it is not found.

#### Step 2: Prepare the Dataset

Place your `processed_taco_coco` dataset according to the structure defined in `config.py`. For a local setup, the expected structure is:

```
.
├── data_input/
│   └── processed_taco_coco/
│       ├── train2017/
│       ├── val2017/
│       └── annotations/
└── ... (rest of the project files)
```

#### Step 3: Run the Training Pipeline

The `train.py` script orchestrates the entire process.

**Run the Full Pipeline (Recommended):**
This command executes data preparation, knowledge distillation, and all fine-tuning experiments sequentially.
```bash
python train.py --all
```

**Run Steps Individually:**
This is useful for debugging or re-running specific parts of the pipeline.
```bash
# 1. Prepare data formats (creates YOLO data)
python train.py --prepare-data

# 2. Run knowledge distillation (GPU-intensive)
python train.py --distill

# 3. Run fine-tuning experiments for all three models
python train.py --finetune
```

## 7. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 8. Acknowledgements

-   This project is built upon the official implementation of [RT-DETR](https://github.com/lyuwenyu/RT-DETR).
-   The teacher model, [Conditional-DETR](https://huggingface.co/microsoft/conditional-detr-resnet-50), is provided by Microsoft via the Hugging Face Hub.
-   The baseline model, [YOLO](https://github.com/ultralytics/ultralytics), is provided by Ultralytics.
-   The [TACO dataset](https://tacodataset.org/) is the foundation for this trash detection task.
```
