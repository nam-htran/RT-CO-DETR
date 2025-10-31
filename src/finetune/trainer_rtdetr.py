import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

def run_command(command):
    print(f"Executing: {command}")
    process = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    print(process.stdout)
    if process.stderr:
        print(process.stderr)

def main():
    RTDETR_PYTORCH_DIR = str(config.RTDETR_PYTORCH_DIR)

    run_command(f'cp {config.CONVNEXT_BEST_WEIGHTS} {RTDETR_PYTORCH_DIR}/')
    
    print("\n--- Starting RT-DETR fine-tuning (ConvNeXt teacher) ---")
    config_convnext = str(config.RTDETR_FINETUNE_CONFIG_CONVNEXT)
    run_command(f'cd {RTDETR_PYTORCH_DIR} && torchrun --nproc_per_node={config.NUM_GPUS_PER_NODE} tools/train.py -c {config_convnext} --use-amp --seed=0')

    print("\n--- Converting ViT-distilled checkpoint ---")
    convert_script_path = config.SCRIPTS_DIR / 'convert_lightly_checkpoint.py'
    run_command(f'python {convert_script_path}')

    print("\n--- Starting RT-DETR fine-tuning (ViT teacher) ---")
    config_vit = str(config.RTDETR_FINETUNE_CONFIG_VIT)
    run_command(f'cd {RTDETR_PYTORCH_DIR} && torchrun --nproc_per_node={config.NUM_GPUS_PER_NODE} tools/train.py -c {config_vit} --use-amp --seed=0')
    
    print("\n--- Starting RT-DETR fine-tuning (Baseline) ---")
    config_baseline = str(config.RTDETR_FINETUNE_CONFIG_BASELINE)
    run_command(f'cd {RTDETR_PYTORCH_DIR} && torchrun --nproc_per_node={config.NUM_GPUS_PER_NODE} tools/train.py -c {config_baseline} --use-amp --seed=0')

if __name__ == '__main__':
    main()