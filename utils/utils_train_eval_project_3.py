import time
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.nn as nn

import importlib
import utils.utils_metrics_project_3 as ump3
importlib.reload(ump3)
from utils.utils_metrics_project_3 import macro_f1, sync, accuracy_top1, mps_allocated



def save_best_checkpoint(save_path, model, history, metrics, config=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": metrics,
        "config": config or {},
    }
    torch.save(ckpt, save_path)
    return str(save_path)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches = 0

    max_mem = 0
    sync(device)
    t0 = time.perf_counter()

    pbar = tqdm(loader, desc="train", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc  += accuracy_top1(logits.detach(), yb)
        n_batches += 1

        max_mem = max(max_mem, mps_allocated())

        # show running averages
        pbar.set_postfix({
            "loss": f"{running_loss / n_batches:.4f}",
            "acc":  f"{running_acc / n_batches:.3f}"
        })

    sync(device)
    epoch_time = time.perf_counter() - t0

    return (running_loss / n_batches), (running_acc / n_batches), epoch_time, max_mem



@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches = 0

    max_mem = 0
    all_y = []
    all_pred = []

    pbar = tqdm(loader, desc="eval", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item()
        running_acc  += accuracy_top1(logits, yb)
        n_batches += 1

        max_mem = max(max_mem, mps_allocated())

        pbar.set_postfix({
            "loss": f"{running_loss / n_batches:.4f}",
            "acc":  f"{running_acc / n_batches:.3f}"
        })

        if return_preds:
            all_y.append(yb.cpu())
            all_pred.append(logits.argmax(dim=1).cpu())

    if return_preds:
        all_y = torch.cat(all_y).numpy()
        all_pred = torch.cat(all_pred).numpy()
        return (running_loss / n_batches), (running_acc / n_batches), max_mem, all_y, all_pred

    return (running_loss / n_batches), (running_acc / n_batches), max_mem



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
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_time_sec": [],
        "train_peak_mem": [],
        "val_peak_mem": [],
    }

    best_val_acc = -1.0
    best_state = None

    t_total0 = time.perf_counter()
    for epoch in tqdm(range(epochs),desc="epochs"):
        tr_loss, tr_acc, ep_time, tr_mem = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_mem = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["epoch_time_sec"].append(ep_time)
        history["train_peak_mem"].append(tr_mem)
        history["val_peak_mem"].append(va_mem)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
              f"time {ep_time:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.perf_counter() - t_total0

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics (macro-F1) on val and test
    val_loss, val_acc, val_mem, y_true_val, y_pred_val = evaluate(model, val_loader, criterion, device, return_preds=True)
    test_loss, test_acc, test_mem, y_true_test, y_pred_test = evaluate(model, test_loader, criterion, device, return_preds=True)

    metrics = {
        "best_val_acc": float(best_val_acc),
        "final_val_acc": float(val_acc),
        "final_val_macro_f1": float(macro_f1(y_true_val, y_pred_val)),
        "final_test_acc": float(test_acc),
        "final_test_macro_f1": float(macro_f1(y_true_test, y_pred_test)),
        "total_training_time_sec": float(total_time),
        "val_peak_mem_bytes": int(val_mem),
        "test_peak_mem_bytes": int(test_mem),
        "train_peak_mem_bytes": int(max(history["train_peak_mem"]) if history["train_peak_mem"] else 0),
    }

    # Build save paths inside the function (safe and simple)
    save_dir = Path(f"weights/project_3/{save_folder_name}_{epochs}ep")
    filename = f"{save_folder_name}_{epochs}ep_model_weights.pth"
    save_path = save_dir / filename

    saved_to = save_best_checkpoint(
        save_path=save_path,
        model=model,
        history=history,
        metrics=metrics,
        config={"epochs": epochs, "lr": lr},
    )

    print("\nSaved best checkpoint to:", saved_to)
    print("Metrics:", metrics)
    return model, history, metrics, saved_to
