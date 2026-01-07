import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

IMG_SIZE = 224
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

def seg_train_transform(image, mask):
    # Images/masks resized to IMG_SIZE x IMG_SIZE
    image = TF.resize(image, (IMG_SIZE, IMG_SIZE))
    mask  = TF.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=InterpolationMode.NEAREST  # important for labels
    )

    # Augmentations
    # Horizontal flip with p=0.5
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    # Small random rotation with p=0.3
    if random.random() < 0.3:
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

    # Conversion image to tensor + normalize
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

    # Conversion mask to tensor of class indices
    mask_np = np.array(mask).astype(np.int64)

    # Conversion of trimaps {1, 2, 3} → {0, 1, 2}
    mask_np = mask_np - 1
    mask = torch.from_numpy(mask_np).long()

    return image, mask

def seg_val_transform(image, mask):
    # Images/masks resized to IMG_SIZE x IMG_SIZE
    image = TF.resize(image, (IMG_SIZE, IMG_SIZE))
    mask  = TF.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=InterpolationMode.NEAREST
    )
    
    # To tensor + normalize (image only)
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

    # Mask to class indices
    mask_np = np.array(mask).astype(np.int64)
    mask_np = mask_np - 1
    mask = torch.from_numpy(mask_np).long()

    return image, mask

