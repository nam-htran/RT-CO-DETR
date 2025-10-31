# ===== train.py (Final Version Based on User's Proven Workaround) =====
import os
import subprocess
import argparse
import sys
from pathlib import Path

# Add the project root to sys.path to ensure 'config' can be imported
sys.path.append(str(Path(__file__).parent.absolute()))
import config

# Import pipeline step functions
from scripts.convert_coco_to_yolo import run_conversion as run_coco_to_yolo_conversion
from scripts.generate_rtdetr_configs import run_config_generation
from src.finetune.trainer_yolo import train_yolo_baseline

def run_manual_ddp_script(script_command: str, cwd: Path):
    """
    WORKAROUND for Windows DDP based on the user's original, working code.
    This function bypasses 'torchrun' and manually sets the necessary DDP
    environment variables for a single-GPU process.
    """
    # Create a copy of the current environment to modify
    env = os.environ.copy()

    # --- Manually set DDP environment variables to simulate a single-process DDP environment ---
    env['MASTER_ADDR'] = 'localhost'
    env['MASTER_PORT'] = '29500'  # An arbitrary free port
    env['RANK'] = '0'
    env['WORLD_SIZE'] = str(config.NUM_GPUS_PER_NODE) # Should be '1' for this setup
    env['LOCAL_RANK'] = '0' 
    
    # --- CRITICAL FIX for the 'libuv' error on Windows ---
    env["USE_LIBUV"] = "0"
    
    # Add project root to PYTHONPATH for robust module importing
    env["PYTHONPATH"] = str(config.ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    # Construct the final command to be executed
    py_executable = sys.executable
    command = f'"{py_executable}" {script_command}'
    
    print(f"\n{'='*30}")
    print(f"🚀 Executing MANUAL DDP command in '{cwd}':")
    print(f"   (Bypassing torchrun, setting DDP ENV VARS manually)")
    print(f"   $ {command}")
    print(f"{'='*30}")

    try:
        # Run the python script with the custom-built environment
        subprocess.run(command, shell=True, check=True, text=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Manual DDP Command failed with exit code {e.returncode} ---", file=sys.stderr)
        raise e

def run_distillation():
    """STEP 2: Run knowledge distillation."""
    print("### STEP 2: Running Knowledge Distillation... ###")
    script_path = str(config.SRC_DIR / "distillation/trainer_codetr.py") 
    run_manual_ddp_script(script_path, cwd=config.ROOT_DIR)

def run_finetuning():
    """STEP 3: Run fine-tuning experiments."""
    print("### STEP 3.1: Generating latest RT-DETR config files... ###")
    run_config_generation()
    
    rtdetr_train_script = str(config.RTDETR_TOOLS_DIR / "train.py")
    
    relative_config_distilled = config.RTDETR_FINETUNE_CONFIG_CODETR.relative_to(config.RTDETR_PYTORCH_DIR)
    relative_config_baseline = config.RTDETR_FINETUNE_CONFIG_BASELINE.relative_to(config.RTDETR_PYTORCH_DIR)

    print("\n### STEP 3.2: Fine-tuning RT-DETR (Distilled)... ###")
    command_distilled = f"{rtdetr_train_script} -c {relative_config_distilled} --use-amp --seed=0"
    run_manual_ddp_script(command_distilled, cwd=config.RTDETR_PYTORCH_DIR)

    print("\n### STEP 3.3: Fine-tuning RT-DETR (Baseline)... ###")
    command_baseline = f"{rtdetr_train_script} -c {relative_config_baseline} --use-amp --seed=0"
    run_manual_ddp_script(command_baseline, cwd=config.RTDETR_PYTORCH_DIR)

    print("\n### STEP 3.4: Fine-tuning YOLO (Baseline)... ###")
    train_yolo_baseline() # This script does not need the DDP wrapper

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
        subprocess.run(f"git clone https://github.com/lyuwenyu/RT-DETR.git {config.RTDETR_SOURCE_DIR}", shell=True, check=True, cwd=config.ROOT_DIR)

    if run_all or args.prepare_data:
        print("### STEP 1: Preparing dataset... ###")
        run_coco_to_yolo_conversion()
    if run_all or args.distill:
        run_distillation()
    if run_all or args.finetune:
        run_finetuning()

    print("\n✅ All selected processes completed successfully.")

if __name__ == "__main__":
    main()