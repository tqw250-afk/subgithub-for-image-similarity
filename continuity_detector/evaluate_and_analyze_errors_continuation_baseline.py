"""
REAL TEST SET ERROR ANALYSIS & CONTACT SHEET GENERATOR
=====================================================
Analyzes misclassified samples from a trained model checkpoint, generates a visual
contact sheet of failures, and logs diagnostic JSON/CSV reports into ../results/.
"""

import argparse
from pathlib import Path
import json
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont
import timm
import numpy as np

def build_model(model_name: str, checkpoint_path: Path, device: torch.device):
    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def create_contact_sheet(error_records, output_image_path: Path, grid_cols: int = 3, img_dim: tuple = (400, 200)):
    """Generates a visual contact grid of misclassified image pairs with metadata overlays."""
    if not error_records:
        print("No errors found to render in contact sheet!")
        return

    n_errors = len(error_records)
    grid_rows = (n_errors + grid_cols - 1) // grid_cols
    
    cell_w, cell_h = img_dim
    header_h = 60
    card_h = cell_h + header_h

    sheet_w = grid_cols * cell_w
    sheet_h = grid_rows * card_h

    canvas = Image.new("RGB", (sheet_w, sheet_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    for idx, rec in enumerate(error_records):
        row = idx // grid_cols
        col = idx % grid_cols
        x_offset = col * cell_w
        y_offset = row * card_h

        # Load original image
        raw_img = Image.open(rec["file_path"]).convert("RGB")
        resized_img = raw_img.resize((cell_w, cell_h))
        canvas.paste(resized_img, (x_offset, y_offset + header_h))

        # Header background color (Red for FP/FN)
        box_color = (200, 50, 50) if rec["error_type"] == "False Positive" else (200, 100, 0)
        draw.rectangle([x_offset, y_offset, x_offset + cell_w, y_offset + header_h], fill=box_color)

        text_lines = [
            f"File: {Path(rec['file_path']).name[:25]}",
            f"True: {rec['true_label']} | Pred: {rec['pred_label']}",
            f"Prob(Continuation): {rec['prob_continuation']:.4f}"
        ]
        
        for line_i, line_text in enumerate(text_lines):
            draw.text((x_offset + 8, y_offset + 4 + (line_i * 18)), line_text, fill=(255, 255, 255))

    canvas.save(output_image_path)
    print(f"[✓] Visual Contact Sheet saved to: {output_image_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Error Analysis for Patent Continuation Model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best model .pth file")
    parser.add_argument("--real-test-dir", type=Path, required=True, help="Path to 152 test dataset")
    parser.add_argument("--model-name", type=str, default="resnet34")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--results-dir", type=Path, default=Path("../results"))
    parser.add_argument("--threshold", type=float, default=0.1929, help="Custom decision threshold")
    args = parser.parse_args()

    # Hardware device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    out_dir = args.results_dir / f"{args.checkpoint.stem}_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Running Error Analysis on Checkpoint: {args.checkpoint.name}")
    print(f"[+] Output Directory: {out_dir.resolve()}")

    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(args.real_test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = build_model(args.model_name, args.checkpoint, device)
    
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    cont_idx = test_dataset.class_to_idx.get("continuation", 0)

    error_records = []
    all_predictions = []

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            pred = torch.argmax(outputs, dim=1).item()
            true_lbl = label.item()

            img_path, _ = test_dataset.samples[i]
            prob_cont = float(probs[cont_idx])
            
            if prob_cont >= args.threshold:
                pred = cont_idx
            else:
                pred = 1 - cont_idx

            record = {
                "sample_index": i,
                "file_path": str(img_path),
                "file_name": Path(img_path).name,
                "true_label": idx_to_class[true_lbl],
                "pred_label": idx_to_class[pred],
                "prob_continuation": prob_cont,
                "is_error": true_lbl != pred
            }

            if true_lbl != pred:
                if pred == cont_idx:
                    record["error_type"] = "False Positive"  # Predicted Continuation when it was Separate
                else:
                    record["error_type"] = "False Negative"  # Predicted Separate when it was Continuation
                error_records.append(record)

            all_predictions.append(record)

    print(f"\n[!] Total Test Samples: {len(test_dataset)}")
    print(f"[!] Total Misclassifications: {len(error_records)} / {len(test_dataset)}")

    # 1. Save Errors JSON
    json_path = out_dir / "error_summary.json"
    with open(json_path, "w") as f:
        json.dump({"total_errors": len(error_records), "errors": error_records}, f, indent=2)
    print(f"[✓] Error summary JSON saved to: {json_path.resolve()}")

    # 2. Save Errors CSV
    csv_path = out_dir / "error_summary.csv"
    if error_records:
        keys = error_records[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(error_records)
        print(f"[✓] Error summary CSV saved to: {csv_path.resolve()}")

    # 3. Generate Contact Sheet Image
    contact_sheet_path = out_dir / "error_contact_sheet.png"
    create_contact_sheet(error_records, contact_sheet_path)

if __name__ == "__main__":
    main()