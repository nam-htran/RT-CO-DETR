import os
import subprocess
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
import config

from scripts.convert_coco_to_yolo import run_conversion as run_coco_to_yolo_conversion
from scripts.generate_rtdetr_configs import run_config_generation
from src.finetune.trainer_yolo import train_yolo_baseline

def run_command(command: str, cwd: Path):
    """Helper function to run a shell command."""
    print(f"\n{'='*30}\n Executing in '{cwd}':\n   $ {command}\n{'='*30}")
    try:
        subprocess.run(command, shell=True, check=True, text=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Command failed with exit code {e.returncode} ---", file=sys.stderr)
        raise e

def run_distillation():
    """STEP 2: Run knowledge distillation."""
    print("### STEP 2: Running Knowledge Distillation... ###")
    script_path = str(config.SRC_DIR / "distillation/trainer_codetr.py") 
    command = f"torchrun --nproc_per_node={config.NUM_GPUS_PER_NODE} {script_path}"
    run_command(command, cwd=config.ROOT_DIR)

def run_finetuning():
    """STEP 3: Run fine-tuning experiments."""
    print("### STEP 3.1: Generating latest RT-DETR config files... ###")
    run_config_generation()
    
    rtdetr_train_script = str(config.RTDETR_TOOLS_DIR / "train.py")
    
    relative_config_distilled = config.RTDETR_FINETUNE_CONFIG_DISTILLED.relative_to(config.RTDETR_PYTORCH_DIR)
    relative_config_baseline = config.RTDETR_FINETUNE_CONFIG_BASELINE.relative_to(config.RTDETR_PYTORCH_DIR)

    launcher = f"torchrun --nproc_per_node={config.NUM_GPUS_PER_NODE}"

    print("\n### STEP 3.2: Fine-tuning RT-DETR (Distilled)... ###")
    command_distilled = f"{launcher} {rtdetr_train_script} -c {relative_config_distilled} --use-amp --seed=0"
    run_command(command_distilled, cwd=config.RTDETR_PYTORCH_DIR)

    print("\n### STEP 3.3: Fine-tuning RT-DETR (Baseline)... ###")
    command_baseline = f"{launcher} {rtdetr_train_script} -c {relative_config_baseline} --use-amp --seed=0"
    run_command(command_baseline, cwd=config.RTDETR_PYTORCH_DIR)

    print("\n### STEP 3.4: Fine-tuning YOLO (Baseline)... ###")
    train_yolo_baseline()

def main():
    """Main entry point to orchestrate the full training pipeline."""
    parser = argparse.ArgumentParser(description="Master training orchestrator for the RT-DETR project.")
    parser.add_argument('--all', action='store_true', help='Run all steps: prepare-data, distill, finetune.')
    parser.add_argument('--prepare-data', action='store_true', help='Run only the data preparation step.')
    parser.add_argument('--distill', action='store_true', help='Run only the knowledge distillation step.')
    parser.add_argument('--finetune', action='store_true', help='Run all fine-tuning experiments.')
    args = parser.parse_args()

    run_all = not any([args.prepare_data, args.distill, args.finetune]) or args.all

    if not config.RTDETR_SOURCE_DIR.exists():
        print(f"RT-DETR repository not found. Cloning...")
        run_command(f"git clone https://github.com/lyuwenyu/RT-DETR.git {config.RTDETR_SOURCE_DIR}", cwd=config.ROOT_DIR)

    if run_all or args.prepare_data:
        print("### STEP 1: Preparing dataset... ###")
        run_coco_to_yolo_conversion()
    if run_all or args.distill:
        run_distillation()
    if run_all or args.finetune:
        run_finetuning()

    print("\nAll selected processes completed successfully.")

if __name__ == "__main__":
    main()