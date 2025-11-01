import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def _generate_config_file(template_filename: str, output_filename: Path, replacements: dict):
    template_path = config.TEMPLATES_DIR / template_filename
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, 'r') as f:
        content = f.read()

    for key, value in replacements.items():
        replacement_value = str(Path(value).absolute()).replace('\\', '/')
        content = content.replace(f"{{{key}}}", replacement_value)

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, 'w') as f:
        f.write(content)
    print(f"Generated config file: {output_filename}")

def run_config_generation():
    print("--- Generating RT-DETR config files from templates... ---")
    
    common_replacements = {
        "TRAIN_IMG_FOLDER": config.COCO_TRAIN_IMAGES,
        "VAL_IMG_FOLDER": config.COCO_VAL_IMAGES,
        "TRAIN_ANN_FILE": config.COCO_TRAIN_ANNOTATIONS,
        "VAL_ANN_FILE": config.COCO_VAL_ANNOTATIONS,
    }

    _generate_config_file(
        "rtdetrv2_taco_finetune_distilled.yml.template", 
        config.RTDETR_FINETUNE_CONFIG_DISTILLED,
        {
            **common_replacements,
            "OUTPUT_DIR": config.FINETUNE_DISTILLED_OUTPUT_DIR,
            "TUNING_CHECKPOINT": config.DISTILLED_BEST_WEIGHTS,
        }
    )
    
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