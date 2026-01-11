import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from utils.utils_metrics_project_3 import (
    macro_f1, sync, accuracy_top1,
    reset_peak_memory, peak_allocated_bytes
)


def save_best_checkpoint(save_path, model, history, metrics, config=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "metrics": metrics,
            "config": config or {},
        },
        save_path
    )
    return str(save_path)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    reset_peak_memory(device)
    sync(device)
    t0 = time.perf_counter()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_top1(logits.detach(), yb)
        n_batches += 1

    sync(device)
    epoch_time = time.perf_counter() - t0
    peak_mem = peak_allocated_bytes(device)

    return running_loss / n_batches, running_acc / n_batches, epoch_time, peak_mem

@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()

    reset_peak_memory(device)
    sync(device)

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    all_y, all_pred = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item()
        running_acc += accuracy_top1(logits, yb)
        n_batches += 1

        if return_preds:
            all_y.append(yb.cpu())
            all_pred.append(logits.argmax(dim=1).cpu())

    sync(device)
    peak_mem = peak_allocated_bytes(device)

    if return_preds:
        all_y = torch.cat(all_y).numpy()
        all_pred = torch.cat(all_pred).numpy()
        return running_loss / n_batches, running_acc / n_batches, peak_mem, all_y, all_pred

    return running_loss / n_batches, running_acc / n_batches, peak_mem


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs=30,
    lr=3e-4,
    device=None,
    save_folder_name="model_name",
):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # For ViT on Food-101
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)



    # History (only what you actually use / plot)
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_time_sec": [],
    }

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    train_peak_mem_bytes = 0
    val_peak_mem_bytes = 0

    t_total0 = time.perf_counter()

    pbar = tqdm(range(epochs), desc="epochs")
    for epoch in pbar:
        tr_loss, tr_acc, ep_time, tr_mem = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        va_loss, va_acc, va_mem = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step() # for ViT on Food-101
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["epoch_time_sec"].append(ep_time)

        train_peak_mem_bytes = max(train_peak_mem_bytes, int(tr_mem))
        val_peak_mem_bytes = max(val_peak_mem_bytes, int(va_mem))

        # Clean tqdm update instead of prints
        pbar.set_postfix(
            tr_loss=f"{tr_loss:.3f}",
            tr_acc=f"{tr_acc:.3f}",
            va_loss=f"{va_loss:.3f}",
            va_acc=f"{va_acc:.3f}",
            sec=f"{ep_time:.0f}",
            lr=f"{current_lr:.2e}",
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.perf_counter() - t_total0

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics (macro-F1) on val and test using BEST weights
    val_loss, val_acc, val_mem, y_true_val, y_pred_val = evaluate(
        model, val_loader, criterion, device, return_preds=True
    )
    test_loss, test_acc, test_mem, y_true_test, y_pred_test = evaluate(
        model, test_loader, criterion, device, return_preds=True
    )

    metrics = {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),

        "final_val_acc": float(val_acc),
        "final_val_macro_f1": float(macro_f1(y_true_val, y_pred_val)),

        "final_test_acc": float(test_acc),
        "final_test_macro_f1": float(macro_f1(y_true_test, y_pred_test)),

        "total_training_time_sec": float(total_time),

        "train_peak_mem_bytes": int(train_peak_mem_bytes),
        "val_peak_mem_bytes": int(max(val_peak_mem_bytes, int(val_mem))),
        "test_peak_mem_bytes": int(test_mem),

        "device": device.type,
        "lr": float(lr),
        "epochs": int(epochs),
    }

    save_dir = Path(f"weights/project_3/{save_folder_name}_{epochs}ep")
    filename = f"{save_folder_name}_{epochs}ep_model_weights.pth"
    saved_to = save_best_checkpoint(
        save_path=save_dir / filename,
        model=model,
        history=history,
        metrics=metrics,
        #config={"epochs": epochs, "lr": lr, "device": device.type},
        config={"epochs": epochs, "lr": lr, "wd": 0.05, "scheduler": "cosine", "device": device.type}, # for ViT on Food-101
    )

    print("\nSaved best checkpoint to:", saved_to)
    print("Metrics:", metrics)
    return model, history, metrics, saved_to
