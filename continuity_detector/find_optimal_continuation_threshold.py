import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-test-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="resnet34")
    parser.add_argument("--img-size", type=int, default=320)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    # Load Model
    model = timm.create_model(args.model_name, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Data Setup
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(args.real_test_dir, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    pos_idx = dataset.class_to_idx.get("continuation", 0)

    y_true = []
    y_prob = []

    print(f"Evaluating {args.checkpoint.name} to find optimal threshold...")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, pos_idx]
            
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Convert multiclass labels to binary (1 for continuation, 0 for separate)
    binary_y_true = (y_true == pos_idx).astype(int)

    # Calculate metrics for all thresholds
    precisions, recalls, thresholds = precision_recall_curve(binary_y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    
    # Calculate F1 for each threshold
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]
    
    print("\n" + "=" * 60)
    print(f" THRESHOLD OPTIMIZATION RESULTS (PR-AUC: {pr_auc:.4f})")
    print("=" * 60)
    print(f" [Default 0.50 Threshold]")
    default_preds = (y_prob >= 0.50).astype(int)
    def_f1 = f1_score(binary_y_true, default_preds)
    print(f"   F1-Score:  {def_f1:.4f}")
    
    print(f"\n [OPTIMIZED Threshold: {best_threshold:.4f}]")
    print(f"   F1-Score:  {best_f1:.4f}  <--- True Capability!")
    print(f"   Precision: {precisions[best_idx]:.4f}")
    print(f"   Recall:    {recalls[best_idx]:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()