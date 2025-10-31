import torch
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def run_conversion():
    print(f"Searching for checkpoints in: {config.VIT_DISTILL_CHECKPOINTS_DIR}")
    
    ckpt_files = list(config.VIT_DISTILL_CHECKPOINTS_DIR.glob('*.ckpt'))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No .ckpt files found in '{config.VIT_DISTILL_CHECKPOINTS_DIR}'. "
            "Please ensure ViT distillation has been run successfully."
        )
        
    latest_checkpoint_path = max(ckpt_files, key=os.path.getctime)
    print(f"Found latest checkpoint: {latest_checkpoint_path}")

    lightly_checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
    original_state_dict = lightly_checkpoint['state_dict']

    PREFIX_TO_REMOVE = "student_embedding_model.wrapped_model."
    clean_state_dict = {}
    
    print("Converting state dictionary keys...")
    for key, value in original_state_dict.items():
        if key.startswith(PREFIX_TO_REMOVE):
            temp_key = key[len(PREFIX_TO_REMOVE):]
            if temp_key.startswith('_backbone') or temp_key.startswith('_encoder'):
                new_key = temp_key.lstrip('_')
                clean_state_dict[new_key] = value

    if not clean_state_dict:
        raise ValueError(
            "Could not extract any valid keys from the checkpoint. "
            "Please check the model structure and the prefix to remove."
        )

    final_structure = {'model': clean_state_dict}
    
    config.VIT_CONVERTED_WEIGHTS.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving converted weights to: {config.VIT_CONVERTED_WEIGHTS}")
    torch.save(final_structure, config.VIT_CONVERTED_WEIGHTS)
    print("Checkpoint conversion successful!")

if __name__ == '__main__':
    run_conversion()