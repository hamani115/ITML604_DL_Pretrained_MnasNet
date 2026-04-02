"""
filename:
./mnasnet_imagenette_simple_split.py


Code expect dataset in the following layout:
  $DATA_ROOT/
    train/<class folders...>
    val/<class folders...>

"""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mnasnet0_5
from torchvision.models.mnasnet import MNASNet0_5_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

@torch.no_grad()
def topk_accuracy(logits, targets, k = 1):
    k = min(k, logits.size(1))
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred)).any(dim=1)
    return correct.float().mean().item()

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_targets = []
    all_preds = []
    all_probs = []

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item()
        total_top1 += topk_accuracy(logits, targets, k=1)
        total_top5 += topk_accuracy(logits, targets, k=5)
        n_batches += 1

        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().numpy())

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    y_prob = np.asarray(all_probs)

    f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    auc_macro_ovr = np.nan
    try:
        y_true_1hot = label_binarize(y_true, classes=list(range(num_classes)))
        auc_macro_ovr = float(
            roc_auc_score(y_true_1hot, y_prob, average="macro", multi_class="ovr")
        )
    except Exception:
        pass

    return {
        "loss": total_loss / max(1, n_batches),
        "top1": total_top1 / max(1, n_batches),
        "top5": total_top5 / max(1, n_batches),
        "f1_macro": f1_macro,
        "auc_macro_ovr": auc_macro_ovr,
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }

def main():
    # ==================================================
    # 1. Parse command-line arguments
    # ==================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Directory path containing train/ and val/")
    parser.add_argument("--out_dir", type=str, default="runs_mnasnet_test")
    parser.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    args = parser.parse_args()

    # ####################################################
    # 2. Reproducibility, device, and output directory
    # ##################################################
    # -----------------------Seed-----------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "out_dir": args.out_dir,
                "variant": "0_5",
                "weights": args.weights,
                "img_size": args.img_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "amp": not args.no_amp,
                "label_smoothing": args.label_smoothing,
            },
            indent=2,
        )
    )

    # ==================================================
    # 3. Dataset and data loaders
    # ==================================================
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected {train_dir} to exist")
    if not val_dir.exists():
        raise FileNotFoundError(f"Expected {val_dir} to exist")
    # -----------------Build Transforms-----------------
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(args.img_size * 1.15)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # --------------------------------------------------
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    if train_ds.classes != val_ds.classes:
        raise ValueError("train/val class order mismatch")

    class_names = train_ds.classes
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    print(f"Loaded dataset with {num_classes} classes")

    # ==================================================
    # 4. Model, loss, optimizer, and scheduler
    # ==================================================
    # -------------------Build Model--------------------
    modelHandler = mnasnet0_5(weights=MNASNet0_5_Weights.DEFAULT if args.weights == "imagenet" else None)
    in_features = modelHandler.classifier[1].in_features
    modelHandler.classifier[1] = nn.Linear(in_features, num_classes)
    model = modelHandler.to(device)
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ==================================================
    # 5. Training loop
    # ==================================================
    best_val_top1 = -1.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_top1": [],
        "val_top1": [],
        "train_f1": [],
        "val_f1": [],
    }

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_top1 = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_top1 += topk_accuracy(logits.detach(), targets, k=1)
            n_batches += 1

        scheduler.step()

        train_eval_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
        train_eval = evaluate(model, train_eval_loader, device, num_classes)
        val_eval = evaluate(model, val_loader, device, num_classes)

        history["train_loss"].append(epoch_loss / max(1, n_batches))
        history["train_top1"].append(epoch_top1 / max(1, n_batches))
        history["train_f1"].append(float(train_eval["f1_macro"]))
        history["val_loss"].append(float(val_eval["loss"]))
        history["val_top1"].append(float(val_eval["top1"]))
        history["val_f1"].append(float(val_eval["f1_macro"]))

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train loss {history['train_loss'][-1]:.4f} top1 {history['train_top1'][-1]:.4f} | "
            f"val loss {history['val_loss'][-1]:.4f} top1 {history['val_top1'][-1]:.4f} "
            f"f1 {history['val_f1'][-1]:.4f} auc {val_eval['auc_macro_ovr']}"
        )

        if float(val_eval["top1"]) > best_val_top1:
            best_val_top1 = float(val_eval["top1"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "data_root": args.data_root,
                        "variant": "0_5",
                        "weights": args.weights,
                        "img_size": args.img_size,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "seed": args.seed,
                        "num_workers": args.num_workers,
                        "amp": not args.no_amp,
                        "label_smoothing": args.label_smoothing,
                    },
                },
                run_dir / "best.pt",
            )

            (run_dir / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "val": {k: v for k, v in val_eval.items() if k != "classification_report"},
                        "classification_report": str(val_eval["classification_report"]),
                    },
                    indent=2,
                )
            )

    # ==================================================
    # 6. Curves, confusion matrix, ROC curve, and summary
    # ==================================================
    # ------------------Training Curves-----------------
    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "curve_loss.png", dpi=160)
    plt.close()
    # Top1 Accuracy
    plt.figure()
    plt.plot(history["train_top1"], label="train")
    plt.plot(history["val_top1"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("top-1 acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "curve_top1.png", dpi=160)
    plt.close()
    # F1-score
    plt.figure()
    plt.plot(history["train_f1"], label="train")
    plt.plot(history["val_f1"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "curve_f1.png", dpi=160)
    plt.close()
    # -----------------Confusion Matrix-----------------
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []

        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            y_true.extend(targets.tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=class_names,
            xticks_rotation=45,
            cmap=None,
            normalize=None,
            values_format="d",
        )
        disp.figure_.set_size_inches(9, 7)
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_matrix_val.png", dpi=160)
        plt.close()
    # ---------------Multiclass ROC Curve---------------
    model.eval()
    y_true = []
    y_prob = []
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        y_true.extend(targets.numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy())

    n_classes = len(class_names)
    y_true_1hot = label_binarize(np.asarray(y_true), classes=list(range(n_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_1hot[:, i], np.asarray(y_prob)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro (AUC={roc_auc['macro']:.3f})")
    for i in range(min(n_classes, 6)):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})", alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "roc_val.png", dpi=160)
    plt.close()
    # --------------------------------------------------
    
    best_json = json.loads((run_dir / "best_metrics.json").read_text())
    summary = {
        "best_val_top1": float(best_json["val"]["top1"]),
        "best_val_top5": float(best_json["val"]["top5"]),
        "best_val_f1": float(best_json["val"]["f1_macro"]),
        "best_val_auc": float(best_json["val"]["auc_macro_ovr"])
        if best_json["val"]["auc_macro_ovr"] is not None
        else float("nan"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done. Best:", summary)


if __name__ == "__main__":
    main()
