# ===== src/distillation/trainer_detr.py (Definitive Final Version) =====
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import wandb
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import sys

# Correct, robust import from the transformers library
from transformers import AutoModelForObjectDetection

# Add project root to sys.path to resolve local imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config as project_config
from src.distillation.dataset import CocoDetectionForDistill

def _setup_ddp_if_needed():
    """Initializes DDP if environment variables are set."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        backend = 'gloo' if sys.platform == 'win32' else 'nccl'
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 1

def _cleanup_ddp():
    """Cleans up the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

class DETRTeacherWrapper(nn.Module):
    """
    This wrapper correctly loads a DETR-family model from the Hugging Face Hub.
    Uses the verified 'microsoft/conditional-detr-resnet-50' model ID.
    """
    def __init__(self, model_name: str = "microsoft/conditional-detr-resnet-50"):
        super().__init__()
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"Loading teacher model '{model_name}' from Hugging Face Hub...")

        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        # This will download and cache the model specified by the correct ID
        self._model = AutoModelForObjectDetection.from_pretrained(model_name, token=hf_token)
        
        # Standard feature dimensions for a ResNet-50 backbone
        self.feature_dims = [512, 1024, 2048]

    def forward_features(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Extracts multi-scale features from the model's backbone."""
        backbone_output = self._model.model.backbone(pixel_values, output_hidden_states=True)
        # Return feature maps from stages 2, 3, and 4 (strides 8, 16, 32)
        return [backbone_output.hidden_states[i] for i in [1, 2, 3]]

    def forward_preds(self, pixel_values: torch.Tensor) -> dict:
        """Performs a full forward pass to get final predictions."""
        outputs = self._model(pixel_values=pixel_values)
        return {'pred_logits': outputs.logits, 'pred_boxes': outputs.pred_boxes}

def main_training_function(rank, world_size, cfg):
    device = rank
    is_main_process = (rank == 0)

    if is_main_process:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_name = f"distill_conditional_detr_{timestamp}"
        try:
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key)
                wandb.init(project=cfg["wandb_project"], config=cfg, name=run_name)
        except Exception as e:
            print(f"W&B login failed: {e}. No remote logging will be performed.")
        Path(cfg["best_weights_filename"]).parent.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1: dist.barrier()

    teacher_model = DETRTeacherWrapper(cfg["teacher_model"]).to(device)
    teacher_model.eval()

    if is_main_process:
        torch.hub.load(str(project_config.RTDETR_SOURCE_DIR), "rtdetrv2_l", source='local', pretrained=True, trust_repo=True)
    if world_size > 1: dist.barrier()

    student_hub = torch.hub.load(str(project_config.RTDETR_SOURCE_DIR), "rtdetrv2_l", source='local', pretrained=True, trust_repo=True)
    student_model = student_hub.model.to(device)
    
    student_channels = [512, 1024, 2048]
    projection_layers = nn.ModuleList([
        nn.Conv2d(student_channels[i], teacher_model.feature_dims[i], kernel_size=1) 
        for i in range(len(student_channels))
    ]).to(device)
    
    if world_size > 1:
        student_model = DDP(student_model, device_ids=[device], find_unused_parameters=True)
        projection_layers = DDP(projection_layers, device_ids=[device], find_unused_parameters=True)

    data_transforms = T.Compose([T.Resize((640, 640)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = CocoDetectionForDistill(cfg["train_images_dir"], cfg["train_ann_file"], data_transforms)
    val_dataset = CocoDetectionForDistill(cfg["val_images_dir"], cfg["val_ann_file"], data_transforms)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size_per_gpu"], num_workers=cfg["num_workers"], pin_memory=True, sampler=train_sampler, shuffle=(train_sampler is None))
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size_per_gpu"], num_workers=cfg["num_workers"], pin_memory=True, sampler=val_sampler)

    student_module = student_model.module if world_size > 1 else student_model
    projection_module = projection_layers.module if world_size > 1 else projection_layers
    
    params_to_train = list(student_module.parameters()) + list(projection_module.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg['scheduler_factor'], patience=cfg['scheduler_patience'])
    
    mse_loss_fn = nn.MSELoss()
    kld_loss_fn = nn.KLDivLoss(reduction='batchmean')
    l1_loss_fn = nn.L1Loss(reduction='mean')
    
    if is_main_process and wandb.run:
        wandb.watch((student_model, projection_layers), log="all", log_freq=100)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(cfg["epochs"]):
        if train_sampler: train_sampler.set_epoch(epoch)
        start_time = time.time()
        student_model.train(); projection_layers.train()
        total_train_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", disable=not is_main_process)
        
        for images, _ in train_iterator:
            images = images.to(device, non_blocking=True)
            
            with torch.no_grad():
                teacher_features = teacher_model.forward_features(images)
                teacher_preds = teacher_model.forward_preds(images)
            
            student_preds = student_module(images)
            student_features = student_module.encoder(student_module.backbone(images))

            loss_feat = sum(mse_loss_fn(projection_module[i](student_features[i]), F.interpolate(teacher_features[i], size=student_features[i].shape[-2:], mode="bilinear", align_corners=False)) for i in range(len(student_features)))
            
            T = cfg['distill_temperature']
            loss_cls = kld_loss_fn(F.log_softmax(student_preds['pred_logits']/T, -1), F.softmax(teacher_preds['pred_logits']/T, -1)) * (T*T)
            loss_box = l1_loss_fn(student_preds['pred_boxes'], teacher_preds['pred_boxes'])
            
            total_loss = (cfg['loss_weights']['feat'] * loss_feat) + (cfg['loss_weights']['cls'] * loss_cls) + (cfg['loss_weights']['box'] * loss_box)
            
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()
            total_train_loss += total_loss.item()

        student_model.eval(); projection_layers.eval()
        total_val_loss = 0.0
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Val]", disable=not is_main_process)
        with torch.no_grad():
            for images, _ in val_iterator:
                images = images.to(device, non_blocking=True)
                teacher_features = teacher_model.forward_features(images)
                teacher_preds = teacher_model.forward_preds(images)
                student_preds = student_module(images)
                student_features = student_module.encoder(student_module.backbone(images))
                
                loss_feat_val = sum(mse_loss_fn(projection_module[i](student_features[i]), F.interpolate(teacher_features[i], size=student_features[i].shape[-2:], mode="bilinear", align_corners=False)) for i in range(len(student_features)))
                loss_cls_val = kld_loss_fn(F.log_softmax(student_preds['pred_logits']/T, -1), F.softmax(teacher_preds['pred_logits']/T, -1)) * (T*T)
                loss_box_val = l1_loss_fn(student_preds['pred_boxes'], teacher_preds['pred_boxes'])
                total_loss_val = (cfg['loss_weights']['feat'] * loss_feat_val) + (cfg['loss_weights']['cls'] * loss_cls_val) + (cfg['loss_weights']['box'] * loss_box_val)
                total_val_loss += total_loss_val.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        if world_size > 1:
            train_tensor = torch.tensor(avg_train_loss, device=device); dist.all_reduce(train_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = train_tensor.item()
            val_tensor = torch.tensor(avg_val_loss, device=device); dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            avg_val_loss = val_tensor.item()

        if is_main_process:
            duration = time.time() - start_time
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Duration: {duration:.2f}s")
            if wandb.run: wandb.log({"epoch": epoch + 1, "train/total_loss": avg_train_loss, "val/total_loss": avg_val_loss})
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss, early_stopping_counter = avg_val_loss, 0
                print(f"Validation loss improved to {avg_val_loss:.4f}. Saving best model...")
                torch.save({'model': student_module.state_dict()}, cfg["best_weights_filename"])
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve. Counter: {early_stopping_counter}/{cfg['early_stopping_patience']}")
        
        should_stop_tensor = torch.tensor(int(early_stopping_counter >= cfg['early_stopping_patience']), device=device)
        if world_size > 1: dist.broadcast(should_stop_tensor, src=0)
        if should_stop_tensor.item() == 1:
            if is_main_process: print("Early stopping triggered.")
            break

    if is_main_process:
        print("\nDistillation finished successfully.")
        if wandb.run: wandb.finish()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(project_config.ROOT_DIR / '.env')
    
    rank, world_size = _setup_ddp_if_needed()

    try:
        cfg = {
            "learning_rate": 5e-5, "epochs": 100, "batch_size_per_gpu": 8,
            "num_workers": 2, "weight_decay": 1e-4, 
            "teacher_model": "microsoft/conditional-detr-resnet-50",
            "train_images_dir": str(project_config.COCO_TRAIN_IMAGES), "val_images_dir": str(project_config.COCO_VAL_IMAGES),
            "train_ann_file": str(project_config.COCO_TRAIN_ANNOTATIONS), "val_ann_file": str(project_config.COCO_VAL_ANNOTATIONS),
            "scheduler_patience": 5, "scheduler_factor": 0.2, "early_stopping_patience": 12,
            "distill_temperature": 4.0, "loss_weights": {"feat": 1.0, "cls": 0.8, "box": 1.2},
            "best_weights_filename": str(project_config.CODETR_BEST_WEIGHTS), # Should rename this path later
            "wandb_project": "Distill-RTDETR-ConditionalDETR",
        }
        main_training_function(rank, world_size, cfg)
    finally:
        _cleanup_ddp()