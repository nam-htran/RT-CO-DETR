import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

def train_yolo_baseline():
    output_path = config.YOLO_FINETUNE_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        print("Logged in to W&B successfully.")
    except Exception as e:
        print(f"W&B login failed. Training will proceed without logging. Error: {e}")
    
    wandb.init(
        project=config.WANDB_PROJECT_YOLO_FINETUNE,
        name='yolo11l_taco_finetune_baseline',
        job_type='fine-tuning'
    )

    model_weights = 'yolo11l.pt'
    print(f"Initializing YOLO model with weights: {model_weights}")
    model = YOLO(model_weights)

    device = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # --- Training ---
    print(f"Starting YOLO fine-tuning with data from: {config.YOLO_CONFIG_FILE}")
    print(f"Checkpoints will be saved to: {output_path}")

    model.train(
        data=str(config.YOLO_CONFIG_FILE),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(output_path),
        name='yolo11l_finetune_baseline',
        exist_ok=True,
        device=device
    )
    
    wandb.finish()
    print("\nYOLO fine-tuning complete.")

if __name__ == '__main__':
    train_yolo_baseline()