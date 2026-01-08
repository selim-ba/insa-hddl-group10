# Utilities & Metrics for project 3 : /utils/utils_metrics_project_3.py
import time
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt


def get_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm

def sync(device):
    # Needed for accurate timing (MPS/CUDA are async)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def accuracy_top1(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def mps_allocated():
    # returns bytes (0 if not MPS (Apple))
    if torch.backends.mps.is_available():
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

