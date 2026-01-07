import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class PetDataset(Dataset):
    def __init__(self, df, task="binary", transform=None):
        """
        task = "binary" (0=cat, 1=dog) (binary classification)
        task = "multiclass"   (0..36) (multiclass classification)
        task = "segmentation" (returns mask + image)
        """
        self.df = df
        self.task = task
        self.transform = transform
        
        # To remove the images without trimaps when doing segmentation
        if task == "segmentation":
            self.df = self.df[self.df["trimap_path"].notna()].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["img_path"]).convert("RGB")
        filename = os.path.basename(row["img_path"])

        if self.task == "binary":
            label = row["species"]    # 0=cat, 1=dog

        elif self.task == "multiclass":
            label = row["id"]   # 0..36

        elif self.task == "segmentation":
            mask = Image.open(row["trimap_path"])
            class_id = row["class_id"]

            if self.transform is not None :
                image, mask = self.transform(image, mask)
                return image, mask, class_id # (C,H,W) float, (H,W) long
            
            #if no transform
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=IMAGE_NET_MEAN, std = IMAGE_NET_STD)
            mask = torch.from_numpy(np.array(mask)).long() - 1
            return image, mask, class_id
    
        # Transformation (for classification only)
        if self.transform:
            image = self.transform(image)

        return image, label, filename