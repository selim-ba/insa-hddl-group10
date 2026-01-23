import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn.functional as F

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


## Data Analsysis 
def show_image_mask_bbox(class_id,df):

    row = df.loc[class_id] if class_id in df.index else df[df["class_id"] == class_id].iloc[0]

    img = Image.open(row["img_path"]).convert("RGB")
    ann = Image.open(row["trimap_path"])

    img_arr = np.array(img)
    ann_arr = np.array(ann)

    if img_arr.shape[:2] == ann_arr.shape[:2]:
        print("The mask and image have the same size")
    else:
        print("Warning: image and mask sizes differ:",
              img_arr.shape[:2], ann_arr.shape[:2])

    contour = img_arr.copy()
    contour[ann_arr == 3] = [0, 255, 0] # we highlight the contour
    has_xml = not pd.isna(row["xml_path"])
    n_panels = 4 if has_xml else 3

    print(f"Showing: {row['class_id']}")

    fig, ax = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Original image
    ax[0].imshow(img)
    ax[0].set_title(row["class_id"])
    ax[0].axis("off")

    # Annotation from trimap
    ax[1].imshow(ann_arr, cmap="viridis")
    ax[1].set_title("Annotation")
    ax[1].axis("off")

    # Contour
    ax[2].imshow(contour)
    ax[2].set_title("Contour")
    ax[2].axis("off")

    # Bounding box (when available)
    if has_xml:
        x_min = int(row["bb_xmin"])
        y_min = int(row["bb_ymin"])
        x_max = int(row["bb_xmax"])
        y_max = int(row["bb_ymax"])
        box_w = x_max - x_min
        box_h = y_max - y_min

        ax[3].imshow(img_arr)
        rect = patches.Rectangle(
            (x_min, y_min),
            box_w,
            box_h,
            linewidth=4,
            edgecolor="lime",
            facecolor="none"
        )
        ax[3].add_patch(rect)
        ax[3].set_title("Bounding box")
        ax[3].axis("off")

    plt.tight_layout()
    plt.show()

def show_9_samples(df,random_state_value):
    sample_nine = df.sample(9,random_state=random_state_value)
    plt.figure(figsize=(10, 10))
    for i, (class_id, row) in enumerate(sample_nine.iterrows(), start=1):

        img = Image.open(row["img_path"]).convert("RGB")
        mask = Image.open(row["trimap_path"])

        img_arr = np.array(img)
        mask_arr = np.array(mask)

        overlay = img_arr.copy()
        overlay[mask_arr == 3] = [0, 255, 0] 

        species = row["species_name"].capitalize()    
        breed = row["breed_name"].replace("_", " ").title()   
        title = f"{species} — {class_id}"

        plt.subplot(3, 3, i)
        plt.imshow(overlay)
        plt.title(title, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

#######################################################################################################
## Classification ##
def denormalize_image(tensor):
    """Undo normalization so we can display the image."""
    img = tensor.cpu().numpy().transpose(1,2,0)
    img = img * IMAGE_NET_STD + IMAGE_NET_MEAN
    img = np.clip(img, 0, 1)
    return img

def show_random_classif_predictions(model, dataset, device, class_names, num_images=9):
    model.eval()

    indices = random.sample(range(len(dataset)), num_images)

    grid_size = int(num_images ** 0.5)
    plt.figure(figsize=(grid_size * 4, grid_size * 4))

    for i, idx in enumerate(indices):
        img, label, filename = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]  # shape: [num_classes]
            pred = torch.argmax(probs).item()

        img_vis = denormalize_image(img)

        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img_vis)
        plt.axis("off")

        prob_str = ", ".join([f"P_{name}={probs[j]:.2f}" for j, name in enumerate(class_names)])

        pred_name = class_names[pred]
        true_name = class_names[label]

        title = (
            f"{filename}\n"
            f"Pred: {pred_name}\n"
            f"{prob_str}\n"
            f"True: {true_name}"
        )

        plt.title(title, fontsize=10)

    plt.tight_layout()
    plt.show()

def show_random_classif_predictions_topk(
    model,
    dataset,
    device,
    class_names,
    num_images=9,
    k=3,
):
    model.eval()

    indices = random.sample(range(len(dataset)), num_images)

    grid_size = int(np.ceil(num_images ** 0.5))
    plt.figure(figsize=(grid_size * 4, grid_size * 4))

    for i, idx in enumerate(indices):
        img, label, filename = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]  # shape: [num_classes]
            pred = torch.argmax(probs).item()

        img_vis = denormalize_image(img)

        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img_vis)
        plt.axis("off")

        # top-k predictions
        topk_vals, topk_idx = torch.topk(probs, k)
        topk_vals = topk_vals.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()

        topk_str_lines = []
        for rank, (cidx, p) in enumerate(zip(topk_idx, topk_vals), start=1):
            cname = class_names[cidx]
            topk_str_lines.append(f"{rank}) {cname}: {p:.2f}")

        pred_name = class_names[pred]
        true_name = class_names[label]

        title = (
            f"{filename}\n"
            f"Pred: {pred_name} | True: {true_name}\n"
            + "\n".join(topk_str_lines)
        )

        plt.title(title, fontsize=9)

    plt.tight_layout()
    plt.show()

######################################################################################################
## Multi-class Classification ##
def predict_image_for_multiclass(model, img_path, transform, device="cpu"):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()

    return pred, img

def study_most_confused_breeds(cm, label_map, top_k=10):

    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    pairs = []
    num_classes = len(cm_normalized)

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                true_label = label_map[i] if not isinstance(label_map, dict) else label_map[i]
                pred_label = label_map[j] if not isinstance(label_map, dict) else label_map[j]
                pairs.append((true_label, pred_label, cm_normalized[i, j]))

    confusion_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

    # most confused pairs
    print(f"\nTop {top_k} most confused class pairs:")
    for true_label, pred_label, rate in confusion_pairs[:top_k]:
        print(f"{true_label} → {pred_label}: {rate:.2f}")

    return confusion_pairs

def plot_misclassified_pair(
    model,
    samples,
    true_label_idx,
    pred_label_idx,
    label_map,
    transform,
    device,
    n_samples=5,
    label_col="breed_idx",  
    path_col="img_path",    
):
    model.eval()
    misclassified = []

    for _, row in samples.iterrows():
        img_path = row[path_col]
        pred_idx, img = predict_image_for_multiclass(model, img_path, transform, device)
        true_idx = row[label_col]

        if true_idx == true_label_idx and pred_idx == pred_label_idx:
            misclassified.append((img, true_idx, pred_idx))

    if len(misclassified) == 0:
        print("No misclassified samples found for this pair.")
        return

    examples = random.sample(misclassified, min(n_samples, len(misclassified)))

    n_cols = len(examples)
    plt.figure(figsize=(4 * n_cols, 4))
    for i, (img, true_idx, pred_idx) in enumerate(examples):
        plt.subplot(1, n_cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True: {label_map[true_idx]}\nPred: {label_map[pred_idx]}")
    plt.show()

########################################################################################################
## Segmentation ##
def decode_segmentation_mask(mask):
    """
    mask: (H, W) tensor or numpy, values in {0,1,2}
    returns: (H, W, 3) uint8 color image
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    palette = {
        0: (255, 182,  193),    # pet  
        1: (0,   0,   0),    # background     
        2: (0,   255, 0),    # border     
    }

    for cls_id, color in palette.items():
        color_mask[mask == cls_id] = color

    return color_mask


def visualize_segmentation_predictions(model, loader, device, num_images=5):
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)               # (B, C, H, W)
            preds   = outputs.argmax(dim=1)     # (B, H, W)

            batch_size = imgs.size(0)

            for i in range(batch_size):
                if images_shown >= num_images:
                    return

                img  = imgs[i]
                gt   = masks[i]
                pred = preds[i]

                img_np  = denormalize_image(img)
                gt_np   = decode_segmentation_mask(gt)
                pred_np = decode_segmentation_mask(pred)

                plt.figure(figsize=(8, 4)) #was 12,4

                plt.subplot(1, 3, 1)
                plt.imshow(img_np)
                plt.axis("off")
                plt.title("Input image")

                plt.subplot(1, 3, 2)
                plt.imshow(gt_np)
                plt.axis("off")
                plt.title("Ground truth")

                plt.subplot(1, 3, 3)
                plt.imshow(pred_np)
                plt.axis("off")
                plt.title("Prediction")

                plt.tight_layout()
                plt.show()

                images_shown += 1

@torch.no_grad()
def visualize_topk_by_metric(
    model,
    dataset,         
    df_joined,      
    device,
    metric="pet_dice",
    k=6,
    worst=True
):
    """
    Shows the worst/best K images globally (not per breed). Use filters below for breed/species.
    Requires dataset[i] returns (img, mask, class_id) for segmentation.
    """
    model.eval()

    # sort rows
    df_sorted = df_joined.sort_values(metric, ascending=worst).head(k)

    for _, row in df_sorted.iterrows():
        class_id = row["class_id"]

        idx = None
        for j in range(len(dataset)):
            _, _, cid = dataset[j]
            if cid == class_id:
                idx = j
                break
        if idx is None:
            print(f"[Skip] class_id {class_id} not found in dataset")
            continue

        img, gt_mask, _ = dataset[idx]
        img_b = img.unsqueeze(0).to(device)

        logits = model(img_b)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu()

        img_np  = denormalize_image(img)
        gt_np   = decode_segmentation_mask(gt_mask)
        pred_np = decode_segmentation_mask(pred_mask)

        title = (f"{row['species_name']} | {row['breed_name']} | {class_id} | "
                f"{metric}={row[metric]:.3f} | "
                f"pet_dice={row['pet_dice']:.3f} | "
                f"bg_dice={row['background_dice']:.3f} | "
                f"border_dice={row['border_dice']:.3f}")


        plt.figure(figsize=(8, 4)) #was 12,4
        plt.suptitle(title, y=1.02)

        plt.subplot(1, 3, 1); plt.imshow(img_np);  plt.axis("off"); plt.title("Input")
        plt.subplot(1, 3, 2); plt.imshow(gt_np);   plt.axis("off"); plt.title("Ground truth")
        plt.subplot(1, 3, 3); plt.imshow(pred_np); plt.axis("off"); plt.title("Prediction")

        plt.tight_layout()
        plt.show()