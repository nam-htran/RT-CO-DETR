import os
import json
from tqdm import tqdm
import yaml
import sys
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def _setup_directories():
    print("Setting up directories for YOLO format...")
    train_images_dst = config.YOLO_TRAIN_IMAGES
    val_images_dst = config.YOLO_VAL_IMAGES

    train_images_dst.mkdir(parents=True, exist_ok=True)
    val_images_dst.mkdir(parents=True, exist_ok=True)
    config.YOLO_TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
    config.YOLO_VAL_LABELS.mkdir(parents=True, exist_ok=True)

    print("Copying training images if they do not exist...")
    for src_file in tqdm(list(config.COCO_TRAIN_IMAGES.glob('*')), desc="Copying train images"):
        dst_file = train_images_dst / src_file.name
        if not dst_file.exists():
            shutil.copy(str(src_file), str(dst_file))

    print("Copying validation images if they do not exist...")
    for src_file in tqdm(list(config.COCO_VAL_IMAGES.glob('*')), desc="Copying val images"):
        dst_file = val_images_dst / src_file.name
        if not dst_file.exists():
            shutil.copy(str(src_file), str(dst_file))

def _convert_annotations(json_file: Path, output_labels_dir: Path):
    with open(json_file) as f:
        data = json.load(f)
    
    images_map = {img['id']: (img['file_name'], img['width'], img['height']) for img in data['images']}
    
    for ann in tqdm(data['annotations'], desc=f"Converting {json_file.name}"):
        image_id, class_id = ann['image_id'], ann['category_id']
        
        if image_id not in images_map: continue
            
        file_name, img_w, img_h = images_map[image_id]
        x, y, w, h = ann['bbox']
        
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        
        label_file_path = output_labels_dir / f"{Path(file_name).stem}.txt"
        with open(label_file_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def _create_yolo_yaml_file():
    print(f"Creating YOLO YAML config file at {config.YOLO_CONFIG_FILE}")
    with open(config.COCO_TRAIN_ANNOTATIONS) as f:
        coco_data = json.load(f)
    
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    
    yolo_yaml_content = {
        'path': str(config.YOLO_DATA_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(config.YOLO_CONFIG_FILE, 'w') as f:
        yaml.dump(yolo_yaml_content, f, sort_keys=False, indent=2)

def run_conversion():
    """Main function to perform the entire COCO to YOLO conversion process."""
    _setup_directories()
    print("\nConverting training annotations...")
    _convert_annotations(config.COCO_TRAIN_ANNOTATIONS, config.YOLO_TRAIN_LABELS)
    print("\nConverting validation annotations...")
    _convert_annotations(config.COCO_VAL_ANNOTATIONS, config.YOLO_VAL_LABELS)
    _create_yolo_yaml_file()
    print("\nConversion to YOLO format complete.")

if __name__ == '__main__':
    run_conversion()