# RT-CO-DETR: Boosting Real-Time Object Detection with Knowledge Distillation

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project enhances the performance of the real-time object detector **RT-DETR** on the **TACO (Trash Annotations in Context)** dataset by leveraging **Knowledge Distillation**. We use a powerful but slower teacher model, **Conditional DETR**, to guide the training of a faster student model, **RT-DETR-L**, significantly improving its accuracy without compromising its real-time inference speed.

The entire training and benchmarking pipeline is designed to be executed within a **Kaggle Notebook** environment.

## 📋 Table of Contents

1.  [Overview](#1-overview)
2.  [Key Features](#2-key-features)
3.  [Performance Benchmark](#3-performance-benchmark)
4.  [Methodology](#4-methodology)
5.  [Project Structure](#5-project-structure)
6.  [How to Reproduce (Kaggle Notebook)](#6-how-to-reproduce-kaggle-notebook)
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
-   **Automated Training Orchestration:** A master script (`train.py`) manages the entire workflow, from data preparation to distillation and final fine-tuning experiments.
-   **Kaggle-Optimized:** The entire workflow is designed to run seamlessly in Kaggle Notebooks, automatically handling data paths and dependencies.
-   **Multi-GPU Support:** Leverages `torchrun` for efficient, distributed training.
-   **Comprehensive Benchmarking:** A dedicated script within our Kaggle notebook compares models across accuracy (mAP), complexity (Parameters, FLOPs), and inference speed.

## 3. Performance Benchmark

All models were evaluated on the TACO validation set. The benchmark was conducted on a **Tesla T4** GPU within a Kaggle Notebook. The full analysis, including the generation of these results, can be found in the analysis notebook mentioned in the "How to Reproduce" section.

| Model | mAP@.50-.95 | mAP@.50 | Speed (ms) | Params (M) | FLOPs (G) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RT-DETR (Distilled)** | **0.2610** | **0.3100** | 59.06 | 40.92 | 136.06 |
| RT-DETR (Baseline) | 0.2390 | 0.3000 | 59.86 | 40.92 | 136.06 |
| YOLOv11l (Baseline) | 0.2566 | 0.2960 | **28.74** | **25.36** | **87.53** |

### Analysis of Results

1.  **Knowledge Distillation is Highly Effective:**
    -   The **Distilled RT-DETR** significantly outperforms the **Baseline RT-DETR**, achieving a **9.2% relative improvement** in mAP@.50-.95.
    -   This accuracy gain comes at **no extra cost** to inference speed or model complexity. This is the core success of the project.

2.  **Comparison with a Strong Baseline (YOLO):**
    -   The **YOLOv11l** model is significantly faster and more efficient (2x faster, ~62% of the parameters).
    -   However, the **Distilled RT-DETR** achieves **higher accuracy** across both mAP metrics, demonstrating its superior capability in learning complex representations after being guided by a powerful teacher.

**Conclusion:** Knowledge distillation is a powerful technique to enhance a real-time detector like RT-DETR, allowing it to achieve state-of-the-art accuracy that can even surpass other highly optimized models like YOLO, albeit with a trade-off in inference speed.

## 4. Methodology

The project follows a three-stage pipeline: Data Preparation, Knowledge Distillation, and Comparative Fine-tuning.
<img width="2776" height="2712" alt="Mermaid Chart - Create complex, visual diagrams with text -2025-11-01-072938" src="https://github.com/user-attachments/assets/b98318fd-2c66-4bac-a0e5-7399999e6f29" />


## 5. Project Structure

The codebase is organized into a modular and reusable structure.

```
.
├── config.py               # Central configuration for all paths and settings
├── train.py                # Master script to orchestrate the entire training pipeline
├── requirements.txt        # Project dependencies
├── scripts/                # Helper scripts for data preparation and config generation
├── src/
│   ├── distillation/       # Core logic for the knowledge distillation process
│   └── finetune/           # Scripts for fine-tuning baseline models
├── rtdetr/                 # Submodule containing the RT-DETR source code
├── templates/              # Template files for generating experiment configs
└── output/                 # Default directory for all generated outputs
```

## 6. How to Reproduce (Kaggle Notebook)

The entire analysis and benchmarking process is encapsulated in a single, self-contained Kaggle Notebook. This notebook uses the pre-trained models and datasets to generate the final results.

### Step 1: Access the Analysis Notebook

The notebook containing the complete benchmark code is located at:
-   **`/kaggle/aftertrain-analysis-rt-codetr.ipynb`**

This notebook is the single source of truth for reproducing the results.

### Step 2: Configure the Notebook Environment

When you open the notebook, ensure the following datasets are attached:

-   **TACO Dataset:** Attach the dataset containing the trash images and annotations. The expected input path is `/kaggle/input/dsp-pre-final`.
-   **Pre-trained Models:** Attach the dataset containing the final trained checkpoints and config templates. The expected input path is `/kaggle/input/rt-co-detr-trained`.

### Step 3: Run All Cells

Execute all cells in the notebook sequentially from top to bottom. The notebook is designed to be fully automated:
1.  It installs all necessary dependencies.
2.  It clones the required `RT-CO-DETR` repository.
3.  It runs the `final_benchmark.py` script, which:
    -   Automatically locates pre-trained models and data.
    -   Performs accuracy evaluation, complexity analysis, and speed measurement.
4.  Finally, it generates and displays the summary table and comparison plot, which are also saved to the `/kaggle/working/benchmark_output/` directory.

By following these steps, you can precisely replicate the benchmark results presented in this document.

## 7. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 8. Acknowledgements

-   This project is built upon the official implementation of [RT-DETR](https://github.com/lyuwenyu/RT-DETR).
-   The teacher model, [Conditional-DETR](https://huggingface.co/microsoft/conditional-detr-resnet-50), is provided by Microsoft via the Hugging Face Hub.
-   The baseline model, [YOLO](https://github.com/ultralytics/ultralytics), is provided by Ultralytics.
-   The [TACO dataset](https://tacodataset.org/) is the foundation for this trash detection task.
