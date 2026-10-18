"""
ROBUST SYNTHETIC PAIR DATASET GENERATOR FOR CONTINUATION CLASSIFIER
===================================================================
Generates synthetic Continuation (+) and Separate (-) edge crops 
from multi-page and single-page patent images.

Includes Targeted Augmentations:
1. Hard Negatives: Margin text headers/footers ("Till Patentet", horizontal rules) 
   on Separate pairs to eliminate false positives on text banners spanning across pages.
2. Positive Noise: Faint stroke fading and micro-blur on Continuation pairs to prevent 
   fine drawing trajectories from getting lost during downsampling.

Dataset Output Structure:
    synthetic_dataset/
    ├── train/
    │   ├── continuation/
    │   └── separate/
    └── val/
        ├── continuation/
        └── separate/
"""

import argparse
from collections import defaultdict
import random
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# =============================================================================
# CONFIGURATION & PARAMETERS
# =============================================================================

EDGE_CROP_FRAC = 0.20  # 20% width crop from facing edges
EDGE_MAX_HEIGHT = 900  # Normalized crop height before stacking
RANDOM_SEED = 42

# =============================================================================
# TARGETED AUGMENTATION HELPERS
# =============================================================================

def add_faint_stroke_noise(image: np.ndarray, apply_prob: float = 0.6) -> np.ndarray:
    """
    Simulates faint or low-density drawing lines getting washed out 
    during scan downsampling (addresses False Negatives).
    """
    if random.random() > apply_prob:
        return image

    # Micro-blur to simulate ink bleeding / scanner distortion
    sigma = random.uniform(0.5, 1.5)
    blurred = cv.GaussianBlur(image, (3, 3), sigma)

    # Fade contrast towards white background to simulate faint/thin strokes
    fade_factor = random.uniform(0.55, 0.85)
    white_bg = np.full_like(blurred, 255, dtype=np.uint8)
    faded = cv.addWeighted(blurred, fade_factor, white_bg, 1.0 - fade_factor, 0)

    return faded


def add_margin_text_header(image: np.ndarray, apply_prob: float = 0.6) -> np.ndarray:
    """
    Adds tricky horizontal text banners and margin lines to negative pairs 
    to create hard negatives (addresses False Positives on text margins).
    """
    if random.random() > apply_prob:
        return image

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    text_options = [
        "Till Patentet",
        "Generalsstabens Litografiska Anstalt",
        "Fig. 1", "Fig. 2a", "Fig. 3",
        "PATENT SPECIFICATION",
        "SHEET 1 OF 3", "Pl. I", "Pl. II"
    ]

    text = random.choice(text_options)

    try:
        font = ImageFont.truetype("arial.ttf", size=random.randint(12, 22))
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position text near upper or lower page boundary / gap margin
    x = random.randint(10, max(11, w - text_w - 10))
    center_y = h // 2

    # Place near top margin or gap margin
    if random.choice([True, False]):
        y = max(5, center_y - text_h - random.randint(10, 40))
    else:
        y = min(h - text_h - 5, center_y + random.randint(10, 40))

    draw.text((x, y), text, fill=0, font=font)

    # 30% chance to draw a horizontal rule below text (false horizontal continuity signal)
    if random.random() < 0.30:
        line_y = y + text_h + 3 if y < center_y else y - 3
        draw.line([(x, line_y), (x + text_w + random.randint(20, 60), line_y)], fill=0, width=random.randint(1, 2))

    return np.array(pil_img)

# =============================================================================
# IMAGE & CROP HELPERS
# =============================================================================

def crop_facing_edge(image: np.ndarray, side: str, crop_frac: float = EDGE_CROP_FRAC) -> np.ndarray:
    """Crops the left or right edge of an image and normalizes height."""
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    h, w = image.shape
    crop_w = max(40, int(w * crop_frac))

    if side == "right":
        crop = image[:, max(0, w - crop_w):]
    elif side == "left":
        crop = image[:, :min(w, crop_w)]
    else:
        raise ValueError("Side must be 'left' or 'right'")

    # Scale height if larger than max
    if crop.shape[0] > EDGE_MAX_HEIGHT:
        scale = EDGE_MAX_HEIGHT / crop.shape[0]
        new_w = max(1, int(crop.shape[1] * scale))
        crop = cv.resize(crop, (new_w, EDGE_MAX_HEIGHT), interpolation=cv.INTER_AREA)

    return crop


def make_zero_gap_pair(crop_left: np.ndarray, crop_right: np.ndarray) -> np.ndarray:
    """Stacks two edge crops horizontally with zero gap."""
    h1, w1 = crop_left.shape
    h2, w2 = crop_right.shape

    target_h = min(h1, h2)
    
    if h1 != target_h:
        w1_new = max(1, int(w1 * (target_h / h1)))
        crop_left = cv.resize(crop_left, (w1_new, target_h), interpolation=cv.INTER_AREA)
    if h2 != target_h:
        w2_new = max(1, int(w2 * (target_h / h2)))
        crop_right = cv.resize(crop_right, (w2_new, target_h), interpolation=cv.INTER_AREA)

    return np.hstack([crop_left, crop_right])


def make_realistic_pair(crop_left: np.ndarray, crop_right: np.ndarray, is_positive: bool = True) -> np.ndarray:
    """Stacks two edge crops horizontally, applying gaps and jitter indiscriminately to prevent shortcut learning."""
    h1, w1 = crop_left.shape
    h2, w2 = crop_right.shape

    target_h = min(h1, h2)
    
    if h1 != target_h:
        crop_left = cv.resize(crop_left, (max(1, int(w1 * (target_h / h1))), target_h))
    if h2 != target_h:
        crop_right = cv.resize(crop_right, (max(1, int(w2 * (target_h / h2))), target_h))

    # Apply realistic white gaps randomly to BOTH classes
    if random.choice([True, False]):
        gap_width = random.randint(10, 35)  
        white_gap = np.ones((target_h, gap_width), dtype=np.uint8) * 255
        stacked = np.hstack([crop_left, white_gap, crop_right])
    else:
        stacked = np.hstack([crop_left, crop_right])
        
    # Apply MICRO-JITTER to BOTH classes
    jitter = random.randint(-5, 5)
    if jitter != 0:
        M = np.float32([[1, 0, 0], [0, 1, jitter]])
        stacked = cv.warpAffine(stacked, M, (stacked.shape[1], target_h), borderValue=255)

    # Apply class-specific noise (this is safe as it forces robustness rather than giving a free seam clue)
    if is_positive:
        return add_faint_stroke_noise(stacked, apply_prob=0.6)
    else:
        return add_margin_text_header(stacked, apply_prob=0.6)


def generate_synthetic_positive(image: np.ndarray, jitter_pct: float = 0.05) -> np.ndarray:
    """
    Creates a guaranteed CONTINUATION sample by slicing perfectly through the INK 
    (avoiding blank margins), and rejoining the cut edges.
    """
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 1. Binarize to find ink (white pixels on black background)
    _, binary = cv.threshold(image, 200, 255, cv.THRESH_BINARY_INV)
    
    # 2. Find columns that actually contain ink
    col_sums = np.sum(binary, axis=0)
    ink_cols = np.where(col_sums > 0)[0]
    
    if len(ink_cols) == 0:
        raise ValueError("Blank image, cannot create continuation")
        
    # 3. Find the exact bounding box of the drawing
    min_x, max_x = ink_cols[0], ink_cols[-1]
    drawing_width = max_x - min_x
    
    if drawing_width < 100:
        raise ValueError("Drawing is too narrow to split safely")

    # 4. Cut exactly through the middle of the DRAWING (not the whole page)
    center_of_drawing = min_x + int(drawing_width * (0.50 + random.uniform(-jitter_pct, jitter_pct)))
    
    left_half = image[:, :center_of_drawing]
    right_half = image[:, center_of_drawing:]

    # 5. Crop facing edges and stack them with realistic gap, jitter, and faint line noise
    left_crop = crop_facing_edge(left_half, side="right")
    right_crop = crop_facing_edge(right_half, side="left")

    return make_realistic_pair(left_crop, right_crop, is_positive=True)


def generate_synthetic_negative(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    Creates a 100% Guaranteed SEPARATE sample by pairing facing edges 
    from two completely unrelated pages or non-adjacent drawings.
    """
    crop_a = crop_facing_edge(img_a, side="right")
    crop_b = crop_facing_edge(img_b, side="left")

    return make_realistic_pair(crop_a, crop_b, is_positive=False)

# =============================================================================
# DATASET GROUPING & GENERATION PIPELINE
# =============================================================================

def group_dataset(input_dir: Path):
    """Groups image files into multi-page and single-page patent dictionaries."""
    extensions = ("*.png", "*.jpg", "*.jpeg")
    all_paths = []
    for ext in extensions:
        all_paths.extend(list(input_dir.glob(ext)))

    patents = defaultdict(list)
    for p in all_paths:
        stem = p.stem
        if "_page_" in stem:
            patent_id = stem.split("_page_", 1)[0]
        else:
            patent_id = stem
        patents[patent_id].append(p)

    multi_page = {k: v for k, v in patents.items() if len(v) > 1}
    single_page = {k: v for k, v in patents.items() if len(v) == 1}

    return multi_page, single_page


def build_synthetic_dataset(
    input_dir: Path,
    output_dir: Path,
    num_positives: int = 2000,
    num_negatives: int = 2000,
    val_split: float = 0.20,
    include_single_page: bool = True,
):
    random.seed(RANDOM_SEED)

    multi_page, single_page = group_dataset(input_dir)
    print(f"Dataset Scan: Found {len(multi_page)} Multi-Page Patents and {len(single_page)} Single-Page Patents.")

    all_pages = []
    for p_list in multi_page.values():
        all_pages.extend(p_list)

    if include_single_page:
        for p_list in single_page.values():
            all_pages.extend(p_list)
        print(f"Total candidate pages available for generation: {len(all_pages)}")
    else:
        print(f"Single-page patents excluded. Candidates available: {len(all_pages)}")

    if not all_pages:
        raise FileNotFoundError(f"No valid image files found in {input_dir}")

    # Prepare directories
    train_pos = output_dir / "train" / "continuation"
    train_neg = output_dir / "train" / "separate"
    val_pos = output_dir / "val" / "continuation"
    val_neg = output_dir / "val" / "separate"

    for d in [train_pos, train_neg, val_pos, val_neg]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. GENERATE POSITIVES (CONTINUATION)
    # -------------------------------------------------------------------------
    print("\n[1/2] Generating Synthetic Positive Samples (Continuation + Faint Noise)...")
    pos_count = 0
    pbar = tqdm(total=num_positives, desc="Positives")

    while pos_count < num_positives:
        img_path = random.choice(all_pages)
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

        if img is None or img.shape[1] < 200:
            continue

        try:
            synth_pos = generate_synthetic_positive(img)
            
            # Train / Val Split
            is_val = random.random() < val_split
            target_dir = val_pos if is_val else train_pos
            out_file = target_dir / f"synth_pos_{pos_count:05d}.png"

            cv.imwrite(str(out_file), synth_pos)
            pos_count += 1
            pbar.update(1)
        except Exception:
            continue

    pbar.close()

    # -------------------------------------------------------------------------
    # 2. GENERATE NEGATIVES (SEPARATE)
    # -------------------------------------------------------------------------
    print("\n[2/2] Generating Synthetic Negative Samples (Separate + Margin Text Traps)...")
    neg_count = 0
    pbar = tqdm(total=num_negatives, desc="Negatives")

    while neg_count < num_negatives:
        path_a, path_b = random.sample(all_pages, 2)

        # Ensure we don't accidentally pick consecutive pages of the same patent
        stem_a, stem_b = path_a.stem, path_b.stem
        if "_page_" in stem_a and "_page_" in stem_b:
            p_id_a, p_num_a = stem_a.split("_page_", 1)
            p_id_b, p_num_b = stem_b.split("_page_", 1)

            if p_id_a == p_id_b and abs(int(p_num_a) - int(p_num_b)) <= 1:
                continue  # Skip consecutive or same page

        img_a = cv.imread(str(path_a), cv.IMREAD_GRAYSCALE)
        img_b = cv.imread(str(path_b), cv.IMREAD_GRAYSCALE)

        if img_a is None or img_b is None:
            continue

        try:
            synth_neg = generate_synthetic_negative(img_a, img_b)

            # Train / Val Split
            is_val = random.random() < val_split
            target_dir = val_neg if is_val else train_neg
            out_file = target_dir / f"synth_neg_{neg_count:05d}.png"

            cv.imwrite(str(out_file), synth_neg)
            neg_count += 1
            pbar.update(1)
        except Exception:
            continue

    pbar.close()

    print("\n" + "=" * 60)
    print(" SYNTHETIC DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f" Train Positives (Continuation) : {len(list(train_pos.glob('*.png')))}")
    print(f" Train Negatives (Separate)     : {len(list(train_neg.glob('*.png')))}")
    print(f" Val Positives   (Continuation) : {len(list(val_pos.glob('*.png')))}")
    print(f" Val Negatives   (Separate)     : {len(list(val_neg.glob('*.png')))}")
    print(f" Saved location                 : {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Patent Pair Generator")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing patent images")
    parser.add_argument("--output-dir", type=Path, default=Path("../data/synthetic_dataset_continuation_ViT"), help="Output dataset directory")
    parser.add_argument("--num-positives", type=int, default=2000, help="Target positive pairs to generate")
    parser.add_argument("--num-negatives", type=int, default=2000, help="Target negative pairs to generate")
    parser.add_argument("--exclude-single-page", action="store_true", help="Set flag to exclude single-page patents")

    args = parser.parse_args()

    build_synthetic_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_positives=args.num_positives,
        num_negatives=args.num_negatives,
        include_single_page=not args.exclude_single_page,
    )