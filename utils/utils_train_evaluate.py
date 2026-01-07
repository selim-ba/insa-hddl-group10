import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import os
import json
import pandas as pd

from utils.utils_segmentation import multiclass_dice_coefficient, multiclass_iou


## Classifcation ##
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels, _ in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def train_one_epoch_for_pretrained(model, loader, optimizer, criterion, device, freeze_bn=False):
    """
    Training function specifically for pre-trained models with BatchNorm handling.
    """
    model.train()
    
    # Re-freeze BN layers after model.train() if needed (for frozen backbone training)
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    
    total_loss, correct, total = 0, 0, 0
    for imgs, labels, _ in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)

            probs = F.softmax(outputs, dim=1)[:, 1]  # probability of class "1" (dog)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # print("model.training:", model.training)  # should be False
            # print("dataset len:", len(loader.dataset))
            # print("total seen:", total)               # should equal dataset len


    avg_loss = total_loss / total
    acc = correct / total

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    return avg_loss, acc, all_labels, all_probs

def evaluate_multiclass(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)

            # multiclass
            probs = F.softmax(outputs, dim=1)   # shape [B, num_classes]
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total

    all_labels = np.concatenate(all_labels)     # shape [N]
    all_probs = np.concatenate(all_probs)       # shape [N, num_classes]

    return avg_loss, acc, all_labels, all_probs

## Segmentation ##
def train_one_epoch_segmentation(model, loader, optimizer, criterion, device, num_classes=3):
    model.train()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    total_dice = 0.0
    total_iou  = 0.0

    for imgs, masks, _ in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(imgs)           # (N, C, H, W)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)   # (N, H, W)

        correct_pixels += (preds == masks).sum().item()
        total_pixels   += masks.numel()

        batch_dice = multiclass_dice_coefficient(preds, masks, num_classes)
        batch_iou  = multiclass_iou(preds, masks, num_classes)

        total_dice += batch_dice * batch_size
        total_iou  += batch_iou  * batch_size

    avg_loss  = total_loss / len(loader.dataset)
    pixel_acc = correct_pixels / total_pixels
    avg_dice  = total_dice / len(loader.dataset)
    avg_iou   = total_iou  / len(loader.dataset)

    return avg_loss, pixel_acc, avg_dice, avg_iou


def evaluate_segmentation(model, loader, criterion, device, num_classes=3):
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    total_dice = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Validating", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device, dtype=torch.long)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)

            correct_pixels += (preds == masks).sum().item()
            total_pixels   += masks.numel()

            batch_dice = multiclass_dice_coefficient(preds, masks, num_classes)
            batch_iou  = multiclass_iou(preds, masks, num_classes)

            total_dice += batch_dice * batch_size
            total_iou  += batch_iou  * batch_size

    avg_loss  = total_loss / len(loader.dataset)
    pixel_acc = correct_pixels / total_pixels
    avg_dice  = total_dice / len(loader.dataset)
    avg_iou   = total_iou  / len(loader.dataset)

    return avg_loss, pixel_acc, avg_dice, avg_iou


## Save and Load Model (Classification) ##
def save_model_and_results(
    save_dir: str,
    model: torch.nn.Module,
    history: dict,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: dict,
    model_filename: str = "model_weights.pth",
):
    os.makedirs(save_dir, exist_ok=True)

    # model weights
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    # training history
    pd.DataFrame(history).to_csv(
        os.path.join(save_dir, "training_history.csv"),
        index=False
    )

    # prediction arrays
    np.save(os.path.join(save_dir, "y_true.npy"), y_true)
    np.save(os.path.join(save_dir, "y_prob.npy"), y_prob)

    # confusion matrix
    np.save(
        os.path.join(save_dir, "confusion_matrix.npy"),
        metrics["confusion_matrix"]
    )

    # metrics.json
    avg_prec = metrics["average_precision"]
    if isinstance(avg_prec, dict):
        # multiclass: already per-class, JSON-serializable
        avg_prec_json = avg_prec
    else:
        # binary: single float
        avg_prec_json = float(avg_prec)

    metrics_for_json = {
        "roc_auc": float(metrics["roc_auc"]),
        "classification_report": metrics["classification_report"],
        "average_precision": avg_prec_json,
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_for_json, f, indent=4)

    print(f"\nArtifacts saved to: {save_dir}")

def load_model_and_results(
    save_dir: str,
    model: torch.nn.Module,
    device: str,
    model_filename: str = "model_weights.pth",
):
    # load model weights
    model_path = os.path.join(save_dir, model_filename)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # load history
    history_df = pd.read_csv(os.path.join(save_dir, "training_history.csv"))

    # load prediction arrays
    y_true = np.load(os.path.join(save_dir, "y_true.npy"))
    y_prob = np.load(os.path.join(save_dir, "y_prob.npy"))

    # load confusion matrix
    cm = np.load(os.path.join(save_dir, "confusion_matrix.npy"))

    # load metrics.json
    with open(os.path.join(save_dir, "metrics.json"), "r") as f:
        metrics_json = json.load(f)

    return model, history_df, y_true, y_prob, cm, metrics_json

## Save and Load Model (Segmentation) ##
def save_segmentation_experiment(
    save_dir: str,
    model: torch.nn.Module,
    history: dict,
    metrics: dict,
    model_filename: str = "unet_weights.pth",
):
    """
    save_dir: folder where everything goes
    model:    trained U-Net
    history:  training history dict (lists)
    metrics:  dict with test metrics (loss, pixel_acc, dice, iou)
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Save model weights ---
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    # --- Save training history ---
    pd.DataFrame(history).to_csv(
        os.path.join(save_dir, "training_history.csv"),
        index=False
    )

    # --- Save metrics.json ---
    # ensure floats are JSON-serializable
    metrics_for_json = {k: float(v) for k, v in metrics.items()}
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_for_json, f, indent=4)

    print(f"\nSegmentation artifacts saved to: {save_dir}")

def load_segmentation_experiment(
    save_dir: str,
    model: torch.nn.Module,
    device: str,
    model_filename: str = "unet_weights.pth",
):
    # --- Load model ---
    model_path = os.path.join(save_dir, model_filename)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Load history ---
    history_df = pd.read_csv(os.path.join(save_dir, "training_history.csv"))

    # --- Load metrics.json ---
    with open(os.path.join(save_dir, "metrics.json"), "r") as f:
        metrics_json = json.load(f)

    return model, history_df, metrics_json