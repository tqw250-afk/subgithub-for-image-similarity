import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

def get_optimal_device():
    """Detects the best available hardware accelerator (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Hardware Accelerator: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Hardware Accelerator: Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("Hardware Accelerator: CPU")
    return device

def load_production_model(checkpoint_path, device):
    """Loads the 93% F1 ResNet34 champion model."""
    model = timm.create_model("resnet34", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform

def create_zero_gap_crop(image_a_path, image_b_path, crop_ratio=0.20, max_height=900):
    """
    Dynamically loads two full pages, crops 20% of the facing edges, 
    normalizes their heights to 900px (to match synthetic training distribution), 
    and returns the stitched canvas.
    """
    img_a = Image.open(image_a_path).convert("RGB")
    img_b = Image.open(image_b_path).convert("RGB")
    
    w_a, h_a = img_a.size
    w_b, h_b = img_b.size
    
    # 1. Crop exact fractions from facing edges
    crop_w_a = int(w_a * crop_ratio)
    crop_w_b = int(w_b * crop_ratio)
    
    right_edge_a = img_a.crop((w_a - crop_w_a, 0, w_a, h_a))
    left_edge_b = img_b.crop((0, 0, crop_w_b, h_b))
    
    # 2. Normalize heights to match synthetic parameters while maintaining aspect ratio
    scale_a = max_height / float(h_a)
    scale_b = max_height / float(h_b)
    
    norm_w_a = int(crop_w_a * scale_a)
    norm_w_b = int(crop_w_b * scale_b)
    
    right_edge_a = right_edge_a.resize((norm_w_a, max_height), Image.Resampling.BILINEAR)
    left_edge_b = left_edge_b.resize((norm_w_b, max_height), Image.Resampling.BILINEAR)
    
    # 3. Stitch with zero gap
    stitch_w = norm_w_a + norm_w_b
    stitched = Image.new('RGB', (stitch_w, max_height), (255, 255, 255))
    
    stitched.paste(right_edge_a, (0, 0))
    stitched.paste(left_edge_b, (norm_w_a, 0))
    
    return stitched

def stitch_continuation_run(run_page_paths, output_path):
    """
    Physically stitches full pages side-by-side for ML-accepted continuations.
    Since there is no physical overlap, this is a pure geometric concatenation.
    """
    images = [Image.open(p).convert("RGB") for p in run_page_paths]
    
    total_width = sum(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    
    stitched_sheet = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    x_offset = 0
    for img in images:
        stitched_sheet.paste(img, (x_offset, 0))
        x_offset += img.size[0]
        
    stitched_sheet.save(output_path)
    return total_width, max_height

def process_single_patent(csv_path, images_dir, stitch_dir, output_csv, model, transform, device, threshold, crop_ratio, max_height):
    """Processes rejected Stage A pairs for a single patent, runs inference, and stitches."""
    results = []
    ml_decisions = {}
    
    # STEP 1: Evaluate all rejected pairs from the CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['accepted'] == 'True':
                continue # Skip physical overlaps already accepted by Stage A
                
            img_a_path = Path(images_dir) / row['file_a']
            img_b_path = Path(images_dir) / row['file_b']
            
            if not img_a_path.exists() or not img_b_path.exists():
                print(f"  [WARN] Missing images for {row['file_a']} or {row['file_b']}")
                continue

            # 1. Check Normal Orientation (A Right + B Left)
            crop_normal = create_zero_gap_crop(img_a_path, img_b_path, crop_ratio, max_height)
            tensor_norm = transform(crop_normal).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prob_normal = F.softmax(model(tensor_norm), dim=1)[0, 0].item()
            
            accepted = False
            orientation = "normal"
            prob = prob_normal
            
            # 2. Check Reverse Orientation if normal is rejected (B Right + A Left)
            if prob_normal >= threshold:
                accepted = True
            else:
                crop_reverse = create_zero_gap_crop(img_b_path, img_a_path, crop_ratio, max_height)
                tensor_rev = transform(crop_reverse).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    prob_reverse = F.softmax(model(tensor_rev), dim=1)[0, 0].item()
                
                if prob_reverse >= threshold:
                    accepted = True
                    orientation = "reverse"
                    prob = prob_reverse
            
            page_a_num = int(row['page_a'])
            page_b_num = int(row['page_b'])
            
            results.append({
                "patent_id": row['patent_id'],
                "file_a": row['file_a'],
                "file_b": row['file_b'],
                "is_continuation": accepted,
                "orientation": orientation if accepted else "none",
                "confidence": round(prob, 4)
            })
            
            status = f"CONT ({orientation})" if accepted else "REJECT"
            print(f"  Evaluated {row['file_a']} -> {row['file_b']} | {status} (Conf: {prob:.4f})")
            
            if accepted:
                ml_decisions[(page_a_num, page_b_num)] = {
                    "path_a": img_a_path, 
                    "path_b": img_b_path,
                    "orientation": orientation,
                    "patent_id": row['patent_id']
                }
                
    # STEP 2: Chain and Stitch accepted runs
    stitched_count = 0
    if ml_decisions:
        stitch_dir.mkdir(parents=True, exist_ok=True)
        
        all_pages = sorted(list(set([k[0] for k in ml_decisions.keys()] + [k[1] for k in ml_decisions.keys()])))
        
        start = 0
        while start < len(all_pages) - 1:
            pa = all_pages[start]
            first_pair = ml_decisions.get((pa, pa + 1))
            
            if first_pair is None:
                start += 1
                continue
                
            patent_id = first_pair["patent_id"]
            
            run_paths = [first_pair["path_a"], first_pair["path_b"]]
            if first_pair["orientation"] == "reverse":
                run_paths = [first_pair["path_b"], first_pair["path_a"]]
                
            j = start + 1
            while j < len(all_pages) - 1:
                curr_p = all_pages[j]
                next_p = all_pages[j + 1]
                pair = ml_decisions.get((curr_p, next_p))
                
                if pair is None:
                    break
                    
                if pair["orientation"] == "reverse":
                    run_paths.insert(0, pair["path_b"])
                else:
                    run_paths.append(pair["path_b"])
                    
                j += 1
                
            out_name = f"{patent_id}_ml_sheet_pages_{all_pages[start]:04d}_to_{all_pages[j]:04d}.png"
            stitch_continuation_run(run_paths, stitch_dir / out_name)
            print(f"  -> Stitched massive ML sheet: {out_name} ({len(run_paths)} pages)")
            stitched_count += 1
            
            start = j

    # STEP 3: Export CSV
    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    return len(results), stitched_count


def main():
    parser = argparse.ArgumentParser(description="Stage B: ResNet Continuation Inference (Batch Mode)")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to the Stage A output directory containing all patent folders")
    parser.add_argument("--images-dir", type=str, required=True, help="Base directory containing the source pages")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the ResNet34 .pth model")
    parser.add_argument("--threshold", type=float, default=0.1929, help="Calibrated decision threshold")
    parser.add_argument("--crop-ratio", type=float, default=0.20, help="Fraction of page width to crop")
    parser.add_argument("--max-height", type=int, default=900, help="Height normalization prior to stacking")
    
    args = parser.parse_args()

    # Setup Hardware & Load Model Once
    device = get_optimal_device()
    model, transform = load_production_model(args.checkpoint, device)
    
    # Recursively find all overlap_results.csv files from Stage A
    csv_paths = sorted(list(Path(args.results_dir).rglob("overlap_results.csv")))
    
    if not csv_paths:
        print(f"No overlap_results.csv files found in {args.results_dir}")
        return

    print(f"\nFound {len(csv_paths)} patent directories to evaluate.")
    
    total_pairs_evaluated = 0
    total_ml_sheets_stitched = 0
    
    for csv_path in csv_paths:
        patent_dir = csv_path.parent.parent
        patent_id = patent_dir.name
        output_csv = patent_dir / "ml_continuation_results.csv"
        stitch_dir = patent_dir / "ml_stitched"
        
        print(f"\n[{patent_id}] Processing...")
        
        evaluated, stitched = process_single_patent(
            csv_path=csv_path,
            images_dir=args.images_dir,
            stitch_dir=stitch_dir,
            output_csv=output_csv,
            model=model,
            transform=transform,
            device=device,
            threshold=args.threshold,
            crop_ratio=args.crop_ratio,
            max_height=args.max_height
        )
        
        total_pairs_evaluated += evaluated
        total_ml_sheets_stitched += stitched

    print("\n========== STAGE B COMPLETE ==========")
    print(f"Total Rejected Pairs Evaluated by ML: {total_pairs_evaluated}")
    print(f"Total ML Continuation Sheets Stitched: {total_ml_sheets_stitched}")

if __name__ == "__main__":
    main()