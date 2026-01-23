import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, classification_report, roc_curve, auc, average_precision_score, precision_recall_curve




def classification_metrics(y_true, y_prob, class_names=None, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if y_prob.ndim == 1:
        y_prob_pos = y_prob
        y_prob = np.column_stack([1 - y_prob_pos, y_prob_pos])  # (N, 2)

    elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
        # (N,1) -> (N,2)
        y_prob_pos = y_prob[:, 0]
        y_prob = np.column_stack([1 - y_prob_pos, y_prob_pos])  # (N, 2)

    n_samples, n_classes = y_prob.shape

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # Predictions and ROC AUC
    if n_classes == 2:
        # Binary: thresholding
        y_scores_pos = y_prob[:, 1]
        y_pred = (y_scores_pos >= threshold).astype(int)
        roc_auc = roc_auc_score(y_true, y_scores_pos)
    else:
        # Multiclass: argmax
        y_pred = np.argmax(y_prob, axis=1)
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")


    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true,y_pred,target_names=class_names,digits=3)

    # Per-class PR curves & AP
    precision = {}
    recall = {}
    pr_thresholds = {}
    ap = {}

    y_true_bin = np.eye(n_classes)[y_true]  # shape (N, n_classes)

    for i, cname in enumerate(class_names):
        precision[cname], recall[cname], pr_thresholds[cname] = precision_recall_curve(
            y_true_bin[:, i], y_prob[:, i]
        )
        ap[cname] = average_precision_score(y_true_bin[:, i], y_prob[:, i])

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
        "precision": precision,           
        "recall": recall,                 
        "pr_thresholds": pr_thresholds,   
        "average_precision": ap,          
    }


def classification_metrics_multiclass(y_true, y_prob, class_names=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if y_prob.ndim != 2:
        raise ValueError(
            f"Expected y_prob to have shape (N, C) for multiclass, got shape {y_prob.shape}"
        )

    n_samples, n_classes = y_prob.shape

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    else:
        assert len(class_names) == n_classes, \
            f"len(class_names)={len(class_names)} does not match n_classes={n_classes}"
        
    y_pred = np.argmax(y_prob, axis=1)

    # conf matrix
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(n_classes),   
        target_names=class_names,
        digits=3,
        zero_division=0,
    )

    # Macro ROC AUC OVR
    try:
        roc_auc_macro_ovr = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=np.arange(n_classes),
        )
    except ValueError:
        roc_auc_macro_ovr = None

    # PR curves, AP per class
    precision = {}
    recall = {}
    pr_thresholds = {}
    ap_per_class = {}

    # y_true one-hot
    y_true_bin = np.eye(n_classes)[y_true]  # shape (N, n_classes)

    for i, cname in enumerate(class_names):
        precision[cname], recall[cname], pr_thresholds[cname] = precision_recall_curve(
            y_true_bin[:, i], y_prob[:, i]
        )
        ap_per_class[cname] = average_precision_score(
            y_true_bin[:, i], y_prob[:, i]
        )

    # Macro-average AP
    average_precision_macro = float(np.mean(list(ap_per_class.values())))

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc_macro_ovr": roc_auc_macro_ovr,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "average_precision_per_class": ap_per_class,
        "average_precision_macro": average_precision_macro,
        "roc_auc": roc_auc_macro_ovr,
        "average_precision": ap_per_class,
    }


### Plotting metrics and losses
def plot_accuracy(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(6, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(12,12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_true, y_prob, title="Precision–Recall Curve"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_train_val_loss(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(6, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
