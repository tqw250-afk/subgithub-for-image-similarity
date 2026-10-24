import torch
from torchvision import models, transforms
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# --- CONFIG ---
MANIFEST_FILE = "../results/master_candidate_manifest_v1.csv"
IMAGE_DIR = "../data/sketches_from_7000_pdfs"
MODEL_WEIGHTS = "/Users/garveetjain/Code/Patent-figure-segmentor/final_data_processing_pipeline_swedish_1000/results/v4_resnet34_robust_data_resnet34_sz320/v4_resnet34_robust_data_resnet34_prauc_0.9273_acc_0.7566_ep05.pth"  # UPDATE THIS PATH
CROP_PERCENT = 0.20
BATCH_SIZE = 150

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_stitch(img_path_a, img_path_b):
    img_a = cv2.imread(str(img_path_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(img_path_b), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None: return None

    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape
    crop_w_a, crop_w_b = int(w_a * CROP_PERCENT), int(w_b * CROP_PERCENT)
    
    crop_a = img_a[:, -crop_w_a:]
    crop_b = img_b[:, :crop_w_b]
    
    if h_a != h_b and h_b > 0:
        crop_b = cv2.resize(crop_b, (crop_w_b, h_a), interpolation=cv2.INTER_AREA)
        
    stitched = np.hstack((crop_a, crop_b))
    return cv2.cvtColor(stitched, cv2.COLOR_GRAY2RGB)

def run_active_learning_miner():
    df = pd.read_csv(MANIFEST_FILE)
    
    # Grab unlabeled training data
    unlabeled_train = df[(df['split'] == 'train') & (df['human_label'] == 'unlabeled')].copy()
    
    if unlabeled_train.empty:
        print("No unlabeled training data left!")
        return

    print(f"Scanning {len(unlabeled_train)} unlabeled train pairs to find hard examples...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    uncertainty_scores = []
    
    with torch.no_grad():
        # Limit to 1000 random samples for speed if dataset is huge
        sample_df = unlabeled_train.sample(min(1000, len(unlabeled_train)))
        
        for idx, row in sample_df.iterrows():
            path_a = Path(IMAGE_DIR) / row['file_a']
            path_b = Path(IMAGE_DIR) / row['file_b']
            
            stitched_rgb = load_and_stitch(path_a, path_b)
            if stitched_rgb is None:
                uncertainty_scores.append(-1)
                continue
                
            input_tensor = transform(stitched_rgb).unsqueeze(0).to(device)
            logits = model(input_tensor)
            prob_yes = torch.nn.functional.softmax(logits, dim=1)[0, 1].item()
            
            # ACTIVE LEARNING LOGIC:
            # We want cases where Stage A said False, but Model says True (High disagreement)
            # OR cases where the model is hovering around 50% (High uncertainty)
            
            stage_a_accepted = str(row['stage_a_accepted']).lower() == 'true'
            
            score = 0
            if not stage_a_accepted and prob_yes > 0.80:
                score = 100 # Massive disagreement (Likely a False Positive bug)
            else:
                # Score based on how close probability is to 0.5 (perfectly confused)
                score = 1.0 - abs(prob_yes - 0.5) * 2 
                
            uncertainty_scores.append(score)

    sample_df['al_score'] = uncertainty_scores
    
    # Sort by our Active Learning score (highest first)
    hard_cases = sample_df.sort_values(by='al_score', ascending=False).head(BATCH_SIZE)
    
    # Mark them in the manifest so the UI can find them
    df.loc[hard_cases.index, 'selection_reason'] = 'al_batch_1'
    df.to_csv(MANIFEST_FILE, index=False)
    
    print(f"\nActive Learning complete! Selected {len(hard_cases)} difficult pairs.")
    print("These have been marked as 'al_batch_1' in your manifest.")

if __name__ == "__main__":
    run_active_learning_miner()