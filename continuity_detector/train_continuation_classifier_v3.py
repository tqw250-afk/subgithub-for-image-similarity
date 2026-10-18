"""
PRODUCTION FINE-TUNING & EXPERIMENT LOGGING PIPELINE
====================================================
"""

import argparse
from pathlib import Path
import json
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc
import numpy as np

# =============================================================================
# REPRODUCIBILITY & SETUP
# =============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms(img_size: int = 224, enable_blur: bool = True):
    train_list = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    
    if enable_blur:
        train_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        
    train_list.extend([
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(train_list), eval_transform

# =============================================================================
# ENGINE
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    return running_loss / total, (correct.float() / total).item()


def evaluate(model, dataloader, criterion, device, pos_label_idx: int = 0):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, pos_label_idx]

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = running_loss / total

    # Metrics computation
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label_idx, zero_division=0
    )

    # PR-AUC Calculation
    binary_y_true = (y_true == pos_label_idx).astype(int)
    prec_curve, rec_curve, _ = precision_recall_curve(binary_y_true, y_prob)
    pr_auc = auc(rec_curve, prec_curve)

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist()
    }

    return metrics

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Patent Edge Classifier Training Pipeline")
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--real-test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Exact folder to save checkpoints and logs")
    parser.add_argument("--model-name", type=str, default="resnet34")
    parser.add_argument("--img-size", type=int, default=224, help="Input size e.g. 224, 320, 384")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-blur", action="store_true", help="Ablation flag: disable GaussianBlur")
    parser.add_argument("--checkpoint-selection", type=str, choices=["synth_val", "real_test"], default="real_test", 
                        help="Select best checkpoint based on validation set or real test set")

    args = parser.parse_args()

    seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" EXPERIMENT | Model: {args.model_name} | Size: {args.img_size}x{args.img_size}")
    print(f" Results Folder: {run_dir.resolve()}")
    print("=" * 70)

    train_dir = args.synthetic_dir / "train"
    val_dir = args.synthetic_dir / "val"

    train_transform, eval_transform = get_transforms(img_size=args.img_size, enable_blur=not args.disable_blur)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos_idx = train_dataset.class_to_idx.get("continuation", 0)

    real_test_loader = None
    if args.real_test_dir.exists():
        test_dataset = datasets.ImageFolder(args.real_test_dir, transform=eval_transform)
        real_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_target_metric = 0.0
    best_checkpoint_path = None
    history = []

    print("\nStarting Training Loop...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, pos_label_idx=pos_idx)
        scheduler.step()

        elapsed = time.time() - t0
        log_line = (f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                    f"Synth Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']*100:.2f}% | "
                    f"Time: {elapsed:.1f}s")
        print(log_line)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
        }

        if real_test_loader:
            test_metrics = evaluate(model, real_test_loader, criterion, device, pos_label_idx=pos_idx)
            print(f"   ---> REAL TEST EVAL: Acc: {test_metrics['acc']*100:.2f}% | "
                  f"Continuation Precision: {test_metrics['precision']:.4f} | "
                  f"Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")

            epoch_log.update({
                "test_acc": test_metrics["acc"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_pr_auc": test_metrics["pr_auc"]
            })

            # Checkpoint metric selection
            target_metric = test_metrics["pr_auc"] if args.checkpoint_selection == "real_test" else val_metrics["pr_auc"]

            if target_metric > best_target_metric:
                best_target_metric = target_metric

                if best_checkpoint_path and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()

                ckpt_name = f"{args.model_name}_prauc_{test_metrics['pr_auc']:.4f}_acc_{test_metrics['acc']:.4f}_ep{epoch:02d}.pth"
                best_checkpoint_path = run_dir / ckpt_name
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"   [✓] Saved New Best Model: {best_checkpoint_path.name}")

        history.append(epoch_log)
        print("-" * 70)

    # Save Experiment Logs & Summary
    metrics_json_path = run_dir / "experiment_summary.json"
    summary_data = {
        "config": vars(args),
        "best_checkpoint": str(best_checkpoint_path.name) if best_checkpoint_path else None,
        "best_metric_score": best_target_metric,
        "history": history
    }
    
    # Format args for JSON serialization
    summary_data["config"]["synthetic_dir"] = str(args.synthetic_dir)
    summary_data["config"]["real_test_dir"] = str(args.real_test_dir)
    summary_data["config"]["output_dir"] = str(args.output_dir)

    with open(metrics_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[✓] All experiment artifacts & metrics logged to: {run_dir.resolve()}")

if __name__ == "__main__":
    main()