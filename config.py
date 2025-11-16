import os
from pathlib import Path
import torch

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

ROOT_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.absolute()))
PROJECT_OUTPUT_DIR = ROOT_DIR / 'output'

if IS_KAGGLE:
    print("Kaggle environment detected. Using input data from /kaggle/input/...")
    DATA_INPUT_DIR = Path("/kaggle/input/dsp-pre-final")
else:
    print("Local environment detected. Using input data from 'data_input' folder.")
    DATA_INPUT_DIR = ROOT_DIR / 'data_input'

COCO_INPUT_DIR = DATA_INPUT_DIR / 'processed_taco_coco'
COCO_TRAIN_IMAGES = COCO_INPUT_DIR / 'train2017'
COCO_VAL_IMAGES = COCO_INPUT_DIR / 'val2017'
COCO_TRAIN_ANNOTATIONS = COCO_INPUT_DIR / 'annotations/instances_train2017.json'
COCO_VAL_ANNOTATIONS = COCO_INPUT_DIR / 'annotations/instances_val2017.json'

YOLO_GROUP_DIR = PROJECT_OUTPUT_DIR / 'YOLO'
YOLO_DATA_DIR = YOLO_GROUP_DIR / 'taco_yolo'
YOLO_CONFIG_FILE = YOLO_DATA_DIR / 'taco.yaml'
YOLO_FINETUNE_OUTPUT_DIR = YOLO_GROUP_DIR / 'yolo_checkpoints'
YOLO_TRAIN_IMAGES = YOLO_DATA_DIR / 'images/train'
YOLO_VAL_IMAGES = YOLO_DATA_DIR / 'images/val'
YOLO_TRAIN_LABELS = YOLO_DATA_DIR / 'labels/train'
YOLO_VAL_LABELS = YOLO_DATA_DIR / 'labels/val'

DISTILL_DIR = PROJECT_OUTPUT_DIR / 'DISTILL'
DISTILLED_BEST_WEIGHTS = DISTILL_DIR / 'distilled_rtdetr_teacher_BEST.pth'

RTDETR_SOURCE_DIR = ROOT_DIR / 'rtdetr'
RTDETR_PYTORCH_DIR = RTDETR_SOURCE_DIR
RTDETR_CONFIG_DIR = RTDETR_PYTORCH_DIR / 'configs/rtdetrv2'
RTDETR_TOOLS_DIR = RTDETR_PYTORCH_DIR / 'tools'

FINETUNE_BASELINE_OUTPUT_DIR = PROJECT_OUTPUT_DIR / 'FINETUNE_BASELINE'
FINETUNE_DISTILLED_OUTPUT_DIR = PROJECT_OUTPUT_DIR / 'FINETUNE_DISTILLED'
RTDETR_FINETUNE_CONFIG_DISTILLED = RTDETR_CONFIG_DIR / 'rtdetrv2_taco_finetune_distilled.yml'
RTDETR_FINETUNE_CONFIG_BASELINE = RTDETR_CONFIG_DIR / 'rtdetrv2_taco_finetune_BASELINE.yml'

SCRIPTS_DIR = ROOT_DIR / 'scripts'
SRC_DIR = ROOT_DIR / 'src'
TEMPLATES_DIR = ROOT_DIR / 'templates'

WANDB_PROJECT_DISTILL = "Distill-RTDETR-Teacher"
WANDB_PROJECT_YOLO_FINETUNE = "YOLO-TACO-Benchmark"

NUM_GPUS_PER_NODE = int(os.getenv("NUM_GPUS_PER_NODE", torch.cuda.device_count() if torch.cuda.is_available() else 1))