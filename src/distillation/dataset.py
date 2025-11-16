import os
from PIL import Image
import torch
from pycocotools.coco import COCO

class CocoDetectionForDistill(torch.utils.data.Dataset):
    """
    Custom COCO dataset that only loads images, as labels are not needed for feature distillation.
    """
    def __init__(self, root, ann_file, transforms):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert("RGB")
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, 0

    def __len__(self):
        return len(self.ids)