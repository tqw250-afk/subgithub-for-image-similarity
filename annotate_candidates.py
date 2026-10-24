import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Paths
MANIFEST_FILE = "../results/master_candidate_manifest_v1.csv"
IMAGE_DIR = "../data/sketches_from_7000_pdfs"
CROP_PERCENT = 0.20
DISPLAY_HEIGHT = 900

def load_and_preprocess(img_path_a, img_path_b):
    img_a = cv2.imread(str(img_path_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(img_path_b), cv2.IMREAD_GRAYSCALE)
    
    if img_a is None or img_b is None:
        return None, None, None

    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape
    
    # Stage B logic: Crop facing edges (Right of A, Left of B)
    crop_w_a = int(w_a * CROP_PERCENT)
    crop_w_b = int(w_b * CROP_PERCENT)
    
    crop_a = img_a[:, -crop_w_a:]
    crop_b = img_b[:, :crop_w_b]
    
    # Height normalization (Stage B expectation)
    crop_b_resized = cv2.resize(crop_b, (crop_w_b, h_a), interpolation=cv2.INTER_AREA)
    
    # Zero-gap stitch (what the ResNet actually sees)
    stitched = np.hstack((crop_a, crop_b_resized))
    
    # Resize all to uniform height for UI display
    def resize_for_ui(img):
        aspect = img.shape[1] / img.shape[0]
        return cv2.resize(img, (int(DISPLAY_HEIGHT * aspect), DISPLAY_HEIGHT))
    
    disp_a = cv2.cvtColor(resize_for_ui(img_a), cv2.COLOR_GRAY2BGR)
    disp_b = cv2.cvtColor(resize_for_ui(img_b), cv2.COLOR_GRAY2BGR)
    disp_stitch = cv2.cvtColor(resize_for_ui(stitched), cv2.COLOR_GRAY2BGR)
    
    # Draw red boundary line on the stitch
    midpoint = int((crop_a.shape[1] / stitched.shape[1]) * disp_stitch.shape[1])
    cv2.line(disp_stitch, (midpoint, 0), (midpoint, DISPLAY_HEIGHT), (0, 0, 255), 2)
    
    return disp_a, disp_stitch, disp_b

def run_annotation_loop(target_split='test', batch_size=200):
    print(f"Loading manifest: {MANIFEST_FILE}")
    df = pd.read_csv(MANIFEST_FILE)
    
    # Filter for unlabeled candidates in the target split
    mask = (df['split'] == target_split) & (df['human_label'] == 'unlabeled')
    unlabeled = df[mask]
    
    if unlabeled.empty:
        print(f"No unlabeled candidates found in the '{target_split}' split!")
        return

    # Shuffle to ensure random sampling, then grab the batch
    unlabeled = unlabeled.sample(frac=1, random_state=42).head(batch_size)
    
    print(f"Starting review session for {len(unlabeled)} pairs...")
    print("--- Annotation Key ---")
    print("[1] Confirmed Continuation (Yes)")
    print("[0] Confirmed Non-continuation (No)")
    print("[u] Uncertain (damaged/ambiguous)")
    print("[q] Quit and Save Progress")
    
    cv2.namedWindow("Patent Annotator", cv2.WINDOW_NORMAL)
    
    count = 1
    for idx, row in unlabeled.iterrows():
        path_a = Path(IMAGE_DIR) / row['file_a']
        path_b = Path(IMAGE_DIR) / row['file_b']
        
        disp_a, disp_stitch, disp_b = load_and_preprocess(path_a, path_b)
        
        if disp_a is None:
            df.at[idx, 'human_label'] = 'error'
            print(f"Error loading images for {row['pair_id']}. Skipped.")
            continue
            
        border = np.zeros((DISPLAY_HEIGHT, 10, 3), dtype=np.uint8)
        canvas = np.hstack((disp_a, border, disp_stitch, border, disp_b))
        
        # Overlay context
        cv2.putText(canvas, f"Pair {count}/{len(unlabeled)} | ID: {row['pair_id']} | Stage A Accepted: {row['stage_a_accepted']}", 
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(canvas, f"Pair {count}/{len(unlabeled)} | ID: {row['pair_id']} | Stage A Accepted: {row['stage_a_accepted']}", 
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Patent Annotator", canvas)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('1'):
                df.at[idx, 'human_label'] = '1'
                print(f"[{count}/{len(unlabeled)}] {row['pair_id']} -> Pos (1)")
                break
            elif key == ord('0'):
                df.at[idx, 'human_label'] = '0'
                print(f"[{count}/{len(unlabeled)}] {row['pair_id']} -> Neg (0)")
                break
            elif key == ord('u'):
                df.at[idx, 'human_label'] = 'uncertain'
                print(f"[{count}/{len(unlabeled)}] {row['pair_id']} -> Uncertain (u)")
                break
            elif key == ord('q'):
                df.to_csv(MANIFEST_FILE, index=False)
                print("\nProgress saved. Exiting early.")
                cv2.destroyAllWindows()
                return
        count += 1

    df.to_csv(MANIFEST_FILE, index=False)
    print("\nBatch complete! Progress saved to manifest.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_annotation_loop(target_split='test', batch_size=200)