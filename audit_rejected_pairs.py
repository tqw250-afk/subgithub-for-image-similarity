import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
MANIFEST_FILE = "../results/master_candidate_manifest_v1.csv"
IMAGE_DIR = "../data/sketches_from_7000_pdfs"
CROP_PERCENT = 0.20
SAMPLE_SIZE = 2042  # Change to len(rejected_df) if you want all 2,042

CATEGORY_MAP = {
    ord('0'): 'SEPARATE',
    ord('1'): 'MISSED_PHYSICAL_OVERLAP',
    ord('2'): 'WHITESPACE_CONTINUATION',
    ord('3'): 'VERTICAL_OR_ROTATED',
    ord('4'): 'DUPLICATE_OR_VARIANT',
    ord('5'): 'UNCERTAIN'
}

def load_audit_view(img_path_a, img_path_b, target_h=900):
    img_a = cv2.imread(str(img_path_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(img_path_b), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None: return None

    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape
    
    # 1. Resize Full Pages to target_h height
    scale_a = target_h / h_a if h_a > 0 else 1.0
    scale_b = target_h / h_b if h_b > 0 else 1.0
    
    full_a = cv2.resize(img_a, (int(w_a * scale_a), target_h))
    full_b = cv2.resize(img_b, (int(w_b * scale_b), target_h))
    
    # 2. Create Zoomed Edge Crops (20% width)
    crop_w_a = int(w_a * CROP_PERCENT)
    crop_w_b = int(w_b * CROP_PERCENT)
    crop_a = img_a[:, -crop_w_a:]
    crop_b = img_b[:, :crop_w_b]
    
    if crop_a.shape[0] != target_h:
        crop_a = cv2.resize(crop_a, (crop_a.shape[1], target_h), interpolation=cv2.INTER_AREA)
    if crop_b.shape[0] != target_h:
        crop_b = cv2.resize(crop_b, (crop_b.shape[1], target_h), interpolation=cv2.INTER_AREA)
        
    # 3. Create center seam preview with a red line down the middle
    seam = np.hstack((crop_a, crop_b))
    seam_bgr = cv2.cvtColor(seam, cv2.COLOR_GRAY2BGR)
    red_line_x = crop_a.shape[1]
    cv2.line(seam_bgr, (red_line_x, 0), (red_line_x, target_h), (0, 0, 255), 2)
    
    # Convert grayscale full pages to BGR for uniform stacking
    full_a_bgr = cv2.cvtColor(full_a, cv2.COLOR_GRAY2BGR)
    full_b_bgr = cv2.cvtColor(full_b, cv2.COLOR_GRAY2BGR)
    
    # 4. Thick black bars to separate sections (Width = 15px)
    bar_width = 15
    black_bar = np.zeros((target_h, bar_width, 3), dtype=np.uint8)
    
    # 5. Assemble exact original layout:
    # [ Full Page A ] | [ Black Bar ] | [ Seam Preview + Red Line ] | [ Black Bar ] | [ Full Page B ]
    combined_canvas = np.hstack((full_a_bgr, black_bar, seam_bgr, black_bar, full_b_bgr))
    
    return combined_canvas

def run_audit():
    df = pd.read_csv(MANIFEST_FILE)
    
    if 'audit_label' not in df.columns:
        df['audit_label'] = np.nan
        
    rejected_df = df[df['stage_a_accepted'] == False]
    unaudited_rejected = rejected_df[rejected_df['audit_label'].isna()]
    
    if len(unaudited_rejected) == 0:
        print("All rejected pairs have already been audited!")
        print_summary(df)
        return

    sample_size = min(SAMPLE_SIZE, len(unaudited_rejected))
    sample_df = unaudited_rejected.sample(n=sample_size, random_state=42)
    
    print(f"Starting audit of {len(sample_df)} rejected pairs...")
    print("\nControls:")
    print(" [0] SEPARATE (Unrelated drawings)")
    print(" [1] MISSED_PHYSICAL_OVERLAP (Scans overlap, OpenCV failed)")
    print(" [2] WHITESPACE_CONTINUATION (Drawing strokes continue across gap)")
    print(" [3] VERTICAL_OR_ROTATED (Vertical, reversed or rotated)")
    print(" [4] DUPLICATE_OR_VARIANT (Same drawing, different presentation)")
    print(" [5] UNCERTAIN (Not enough evidence)")
    print(" [q] Quit and save progress\n")

    cv2.namedWindow("Patent Audit UI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Patent Audit UI", 1600, 900)
    
    count = 0
    for idx, row in sample_df.iterrows():
        count += 1
        path_a = Path(IMAGE_DIR) / row['file_a']
        path_b = Path(IMAGE_DIR) / row['file_b']
        
        canvas = load_audit_view(path_a, path_b, target_h=900)
        if canvas is None:
            continue
            
        # Add text banner at the top
        display_img = canvas.copy()
        cv2.putText(display_img, f"Audit {count}/{len(sample_df)} | ID: {row['file_a']} vs {row['file_b']}", 
                    (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Patent Audit UI", display_img)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting early. Progress saved to manifest.")
            break
            
        if key in CATEGORY_MAP:
            label = CATEGORY_MAP[key]
            df.loc[idx, 'audit_label'] = label
            df.to_csv(MANIFEST_FILE, index=False)
            print(f"[{count}/{len(sample_df)}] {row['file_a']} vs {row['file_b']} -> {label}")
        else:
            print("Invalid key pressed. Skipping.")

    cv2.destroyAllWindows()
    print_summary(df)

def print_summary(df):
    audited = df[df['audit_label'].notna()]
    if len(audited) == 0:
        print("No audit data recorded yet.")
        return
        
    print("\n" + "="*40)
    print("      AUDIT RESULTS SUMMARY")
    print("="*40)
    print(f"Total Audited Rejections: {len(audited)}")
    counts = audited['audit_label'].value_counts()
    for cat, count in counts.items():
        pct = (count / len(audited)) * 100
        print(f" - {cat}: {count} ({pct:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    run_audit()