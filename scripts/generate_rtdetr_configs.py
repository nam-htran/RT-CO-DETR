# ===== scripts/generate_rtdetr_configs.py (Corrected Version) =====
import sys
from pathlib import Path
import yaml

# Add project root to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def _generate_config_file(template_filename: str, output_filename: Path, replacements: dict):
    template_path = config.TEMPLATES_DIR / template_filename
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, 'r') as f:
        content = f.read()

    for key, value in replacements.items():
        # Ensure paths are absolute and use forward slashes for cross-platform compatibility
        replacement_value = str(Path(value).absolute()).replace('\\', '/')
        content = content.replace(f"{{{key}}}", replacement_value)

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, 'w') as f:
        f.write(content)
    print(f"Generated config file: {output_filename}")

def run_config_generation():
    """Generates all necessary RT-DETR config files for fine-tuning."""
    print("--- Generating RT-DETR config files from templates... ---")
    
    common_replacements = {
        "TRAIN_IMG_FOLDER": config.COCO_TRAIN_IMAGES,
        "VAL_IMG_FOLDER": config.COCO_VAL_IMAGES,
        "TRAIN_ANN_FILE": config.COCO_TRAIN_ANNOTATIONS,
        "VAL_ANN_FILE": config.COCO_VAL_ANNOTATIONS,
    }

    # Config for the distilled model
    # FIX: Changed variable names to match those defined in config.py
    _generate_config_file(
        "rtdetrv2_taco_finetune_distilled.yml.template",  # Using a consistent template name
        config.RTDETR_FINETUNE_CONFIG_DISTILLED,
        {
            **common_replacements,
            "OUTPUT_DIR": config.FINETUNE_DISTILLED_OUTPUT_DIR,
            "TUNING_CHECKPOINT": config.DISTILLED_BEST_WEIGHTS,
        }
    )
    
    # Config for the Baseline model (no pre-distillation)
    _generate_config_file(
        "rtdetrv2_taco_finetune_baseline.yml.template",
        config.RTDETR_FINETUNE_CONFIG_BASELINE,
        {
            **common_replacements,
            "OUTPUT_DIR": config.FINETUNE_BASELINE_OUTPUT_DIR,
        }
    )
    print("--- Config generation complete. ---")

if __name__ == "__main__":
    run_config_generation()