import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch import Tensor

class HuggingFaceTeacherWrapper(nn.Module):
    """
    A flexible wrapper for teacher models from Hugging Face (supports both ConvNeXT and ViT).
    It standardizes the output to a list of feature maps for consistent distillation loss calculation.
    """
    def __init__(self, model_id: str, token: str = None):
        super().__init__()
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Loading teacher model '{model_id}' from Hugging Face...")
        
        config = AutoConfig.from_pretrained(model_id, token=token, output_hidden_states=True)
        self._model = AutoModel.from_pretrained(model_id, config=config, token=token)
        
        self.is_vit = "vit" in config.model_type.lower()
        
        if self.is_vit:
            self.feature_dims = [self._model.config.hidden_size]
        else:
            all_hidden_sizes = self._model.config.hidden_sizes
            self.feature_dims = [all_hidden_sizes[i] for i in [1, 2, 3]]
        
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Architecture detected: {'ViT' if self.is_vit else 'ConvNeXT'}.")
            print(f"Extracting features with dimensions: {self.feature_dims}")

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = self._model(pixel_values=x)
        
        if self.is_vit:
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            b, s, d = patch_tokens.shape
            h = w = int(s**0.5)
            feature_map = patch_tokens.permute(0, 2, 1).reshape(b, d, h, w)
            return [feature_map]
        else:
            return [outputs.hidden_states[i] for i in [2, 3, 4]]