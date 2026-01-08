# Utilities & Metrics for project 3 : /utils/utils_metrics_project_3.py
import time
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt


def get_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm

@torch.no_grad()
def accuracy_top1(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def sync(device):
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def allocated_bytes(device):
    device = torch.device(device) if isinstance(device, str) else device

    if device.type == "cuda":
        return torch.cuda.memory_allocated()
    if device.type == "mps" and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory()
    return 0

def allocated_mb(device):
    return allocated_bytes(device) / (1024 ** 2)

def reset_peak_memory(device):
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

def peak_allocated_bytes(device):
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    # MPS: no true peak stat, fall back to current allocated
    if device.type == "mps" and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory()
    return 0

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

@torch.no_grad()
def benchmark_latency(model, img_size=224, batch_size=32, n_warmup=10, n_iters=50, device=None):
    model.eval()
    device = device or next(model.parameters()).device

    xb = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # warmup
    for _ in range(n_warmup):
        _ = model(xb)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model(xb)
    sync(device)
    t = time.perf_counter() - t0

    ms_per_batch = (t / n_iters) * 1000.0
    ms_per_image = ms_per_batch / batch_size
    return ms_per_image

def plot_history(history, title=""):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss" if title else "Loss")
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy" if title else "Accuracy")
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix (Normalized)"):
    cm = np.asarray(cm).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)  #
    cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums!=0)

    n = cm.shape[0]
    plt.figure(figsize=(8, 7))
    im = plt.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(range(n), class_names, rotation=45, ha="right")
    plt.yticks(range(n), class_names)

    plt.colorbar(im, fraction=0.046, pad=0.04)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            plt.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val > 0.6 else "black",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()


def show_predictions_grid(dataset, indices, y_true, y_pred, class_names, mean, std, title=""):
    """
    dataset: should be the SAME dataset used for test_loader (same transforms/normalization)
    indices: list of sample indices in that dataset to display
    y_true/y_pred: arrays aligned with dataset order (evaluate() produces in loader order; test loader is sequential)
    mean/std: torch tensors shape [3] used to unnormalize for display
    """
    n = len(indices)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3*cols, 3*rows))

    for k, idx in enumerate(indices):
        x, _ = dataset[idx]  # tensor [3,H,W] already normalized
        x = x * std[:, None, None] + mean[:, None, None]
        x = x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        t = int(y_true[idx])
        p = int(y_pred[idx])

        plt.subplot(rows, cols, k + 1)
        plt.imshow(x)
        plt.axis("off")
        plt.title(f"T: {class_names[t]}\nP: {class_names[p]}", fontsize=9,
                  color=("red" if t != p else "black"))

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def pick_indices(y_true, y_pred, n=18, only_errors=True, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if only_errors:
        pool = np.where(y_true != y_pred)[0]
    else:
        pool = np.arange(len(y_true))

    if len(pool) == 0:
        print("No errors to display.")
        return []

    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return chosen.tolist()

def top_confusions(y_true, y_pred, class_names, top_k=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    pairs = [(int(t), int(p)) for t, p in zip(y_true, y_pred) if t != p]
    counts = Counter(pairs).most_common(top_k)

    print(f"Top {top_k} confusions (true -> pred : count):")
    for (t, p), c in counts:
        print(f"{class_names[t]} -> {class_names[p]} : {c}")

def show_predictions_grid_viz(dataset_viz, indices, y_true, y_pred, class_names, title=""):
    n = len(indices)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3*cols, 3*rows))

    for k, idx in enumerate(indices):
        img, _ = dataset_viz[idx]          # [3,32,32] in [0,1]
        img = img.permute(1, 2, 0).numpy() # HWC

        t = int(y_true[idx])
        p = int(y_pred[idx])

        plt.subplot(rows, cols, k + 1)
        plt.imshow(img, interpolation="nearest")
        plt.axis("off")
        plt.title(f"T: {class_names[t]}\nP: {class_names[p]}",
                  fontsize=9, color=("red" if t != p else "black"))

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
