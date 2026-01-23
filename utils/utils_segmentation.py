import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


def multiclass_dice_coefficient(preds, targets, num_classes=3, eps=1e-6):
    """
    preds:   (N, H, W) class predictions
    targets: (N, H, W) class indices (ground-truth)
    Returns mean Dice over all classes.
    """
    preds_one_hot   = F.one_hot(preds, num_classes).permute(0, 3, 1, 2).float() # from (N,H,W) to (N,H,W,C), each pixel becomes a length-C vector with a single 1 at its class index, after permuteation we have (N,C,H,W)
    # with one-hot we can compute all classes at once as 'preds_one_hot * targets_one_hot' gives a tensor where a pixel is 1 only if predict and ground-truth match for that class
    # summing over (N,H,W) gives intersection for each class as a vector of length C
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3) #N,H,W
    intersection = (preds_one_hot * targets_one_hot).sum(dims) #summing over all images and all pixels, not over the class dimension
    # intersection[0] is the nb of pixels predicted as class 0 and truly class 0 (pet)
    # intersection[1], intersection[2] are same from class 1 (background) and class 2 (border)
    somme = preds_one_hot.sum(dims) + targets_one_hot.sum(dims) 

    dice = (2 * intersection + eps) / (somme + eps)
    return dice.mean().item()

def multiclass_iou(preds, targets, num_classes=3, eps=1e-6):
    """
    preds:   (N, H, W) predicted class indices
    targets: (N, H, W) ground truth class indices
    Returns mean IoU over all classes.
    """
    preds_one_hot   = F.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (preds_one_hot * targets_one_hot).sum(dims)
    union        = preds_one_hot.sum(dims) + targets_one_hot.sum(dims) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def plot_dice_iou(history):
    epochs = range(1, len(history["train_dice"]) + 1)

    # Dice
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, history["train_dice"], label="Train Dice")
    plt.plot(epochs, history["val_dice"],   label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.title("Dice Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # IoU
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, history["train_iou"], label="Train IoU")
    plt.plot(epochs, history["val_iou"],   label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.title("IoU")
    plt.legend()
    plt.tight_layout()
    plt.show()


## Per image metrics
def dice_iou_single_class(pred_hw: torch.Tensor, gt_hw: torch.Tensor, cls: int, eps: float = 1e-6):
    """
    pred_hw, gt_hw: (H, W) integer tensors
    cls: class id to evaluate 
    """
    pred_c = (pred_hw == cls)
    gt_c   = (gt_hw == cls)

    inter = (pred_c & gt_c).sum().item()
    pred_sum = pred_c.sum().item()
    gt_sum   = gt_c.sum().item()

    dice = (2 * inter + eps) / (pred_sum + gt_sum + eps)
    union = pred_sum + gt_sum - inter
    iou  = (inter + eps) / (union + eps)
    return dice, iou

@torch.no_grad()
def compute_per_image_metrics_seg(model, loader, device):
    """
    Expects loader to yield: (imgs, masks, class_ids)
    Returns a DataFrame with one row per image.
    """
    model.eval()
    rows = []

    for imgs, masks, class_ids in tqdm(loader, desc="Per-image test metrics", leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device, dtype=torch.long)

        logits = model(imgs)              # (B, C, H, W)
        preds  = logits.argmax(dim=1)     # (B, H, W)

        for i in range(preds.size(0)):
            pred_i = preds[i]
            gt_i   = masks[i]
            pixel_acc = (pred_i == gt_i).float().mean().item()


            pet_dice, pet_iou     = dice_iou_single_class(pred_i, gt_i, cls=0)
            back_dice, back_iou = dice_iou_single_class(pred_i, gt_i, cls=1)
            border_dice, border_iou = dice_iou_single_class(pred_i, gt_i, cls=2)

            rows.append({
                "class_id": class_ids[i],
                "pixel_acc": pixel_acc,
                "pet_dice": pet_dice,
                "pet_iou": pet_iou,
                "border_dice": border_dice,
                "border_iou": border_iou,
                "background_dice": back_dice,
                "background_iou": back_iou
            })

    return pd.DataFrame(rows)

def join_metrics_with_test_df(test_df: pd.DataFrame, metrics_df: pd.DataFrame):
    """
    Joins on class_id. Assumes test_df has class_id, breed_name, species_name.
    """
    df_joined = test_df.merge(metrics_df, on="class_id", how="inner")
    # sanity check
    if len(df_joined) != len(test_df):
        print(f"[Warn] Joined rows: {len(df_joined)} vs test_df rows: {len(test_df)} "
              f"(some class_id mismatches?)")
    return df_joined

def summarize_group(df, group_col, sort_by="pet_dice"):
    cols = ["pixel_acc", "pet_dice", "pet_iou", "border_dice", "border_iou"]
    out = df.groupby(group_col)[cols].mean().sort_values(sort_by)
    return out

def filter_species(df, species_name=None, breed_name=None):
    out = df
    if species_name is not None:
        out = out[out["species_name"] == species_name]
    if breed_name is not None:
        out = out[out["breed_name"] == breed_name]
    return out

## Losses for segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N, C, H, W)
        # targets: (N, H, W)
        num_classes = logits.shape[1]

        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice
        return loss.mean()

ce_loss   = nn.CrossEntropyLoss()
dice_loss = DiceLoss()

def combined_loss(logits, targets):
    return 0.8 * ce_loss(logits, targets) + 0.2 * dice_loss(logits, targets)



def plot_metric_by_species_row(
    df,
    metric_col="pet_dice",
    species_col="species_name",
    species_order=("Cat", "Dog"),
    bins=30,
    figsize=(18, 5),
    suptitle=None,
):

    # Clean arrays per species
    data = {}
    for sp in species_order:
        vals = df.loc[df[species_col] == sp, metric_col].dropna().astype(float).values
        data[sp] = vals

    missing = [sp for sp, v in data.items() if len(v) == 0]
    if missing:
        raise ValueError(f"No valid '{metric_col}' values found for species: {missing}")

    all_vals = np.concatenate([data[sp] for sp in species_order])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Box-plot
    axes[0].boxplot(
        [data[sp] for sp in species_order],
        tick_labels=list(species_order),
        showfliers=True
    )
    axes[0].set_title("Boxplot")
    axes[0].set_xlabel("Species")
    axes[0].set_ylabel(metric_col)

    # Histogram
    bin_edges = (
        np.linspace(vmin, vmax, bins + 1)
        if vmin != vmax
        else np.linspace(vmin - 1e-6, vmax + 1e-6, bins + 1)
    )
    for sp in species_order:
        axes[1].hist(data[sp], bins=bin_edges, alpha=0.5, label=sp)
    axes[1].set_title("Histogram")
    axes[1].set_xlabel(metric_col)
    axes[1].set_ylabel("Count")
    axes[1].legend()

    # Empirical CDF
    def ecdf(x: np.ndarray):
        x_sorted = np.sort(x)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    for sp in species_order:
        x, y = ecdf(data[sp])
        axes[2].plot(x, y, label=sp)
    axes[2].set_title("ECDF")
    axes[2].set_xlabel(metric_col)
    axes[2].set_ylabel("F(x)")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()


