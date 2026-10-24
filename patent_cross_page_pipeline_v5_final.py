"""
PATENT-LEVEL CROSS-PAGE PIPELINE (V5 - Final Waterfall & Unified Canvas)

Input filenames:
    2382_2387_page_2.png
    2382_2387_page_3.png

WATERFALL PIPELINE:
  1. Edge Overlap (Normal & Reverse)
  2. Full-Page Deduplication (Drops redundant crops/variants >90% contained)
  3. Full-Page Vertical/Rotated Stitching (Stitches new protruding territory)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv
import json
import re
import time
import shutil

import cv2 as cv
import numpy as np


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
INPUT_DIR = Path("sketches_from_7000_pdfs")
OUTPUT_DIR = Path("patent_cross_page_results")

# Stage A Edge
MATCH_STRIP_FRAC = 0.40
ORB_FEATURES = 8000
ORB_FAST_THRESHOLD = 7
RATIO_TEST = 0.78
RANSAC_REPROJ_PX = 4.0
MIN_GOOD_MATCHES = 40
MIN_INLIERS = 20
MIN_INLIER_RATIO = 0.25
VERTICAL_BINS = 8
MIN_Y_BIN_COVERAGE = 0.50
MIN_OVERLAP_RATIO = 0.05
MAX_OVERLAP_RATIO = 0.55
MAX_RELATIVE_SCALE_ERROR = 0.15

# Stage A Full-Page (Waterfall)
FULL_PAGE_MIN_MATCH_COUNT = 250  # Raised to ignore boilerplate text
INSIDE_RATIO_THRESHOLD = 0.90  # >90% area contained = Duplicate

FILENAME_RE = re.compile(
    r"^(?P<patent>.+)_page_(?P<page>\d+)\.(?P<ext>png|jpg|jpeg)$",
    re.IGNORECASE,
)


@dataclass
class OverlapResult:
    patent_id: str
    page_a: int
    page_b: int
    file_a: str
    file_b: str
    accepted: bool
    reason: str
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    y_coverage: float = 0.0
    overlap_ratio: float = 0.0
    scale: float = 0.0
    scale_error: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    transform_a_to_b: Optional[np.ndarray] = None
    inlier_points_a: Optional[np.ndarray] = None
    inlier_points_b: Optional[np.ndarray] = None
    orientation: str = "normal"
    
    # Waterfall Full-Page Extensions
    angle: int = 0
    H_b_to_a: Optional[np.ndarray] = None
    projected_corners: Optional[np.ndarray] = None
    ratio_b_inside_a: float = 0.0
    ratio_a_inside_b: float = 0.0
    img_b_rotated: Optional[np.ndarray] = None


def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("patent"), int(m.group("page"))


def affine_to_homogeneous(M):
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H


def vertical_bin_coverage(y_values, h):
    if len(y_values) == 0:
        return 0.0
    bins = np.floor(
        np.clip(y_values / max(1, h), 0, 0.999999) * VERTICAL_BINS
    ).astype(int)
    return float(len(np.unique(bins))) / VERTICAL_BINS


# =============================================================================
# STAGE A — PHYSICAL SCAN OVERLAP
# =============================================================================

def detect_scan_overlap(gray_a, gray_b, patent_id, page_a, page_b, file_a, file_b):
    h_a, w_a = gray_a.shape[:2]
    h_b, w_b = gray_b.shape[:2]

    if h_a != h_b and h_b > 0:
        scale_b = h_a / h_b
        new_w_b = max(1, int(round(w_b * scale_b)))
        gray_b = cv.resize(gray_b, (new_w_b, h_a), interpolation=cv.INTER_AREA)

    h, w = gray_a.shape[:2]
    strip_w = max(64, int(MATCH_STRIP_FRAC * w))
    a_x0 = w - strip_w

    crop_a = gray_a[:, a_x0:w]
    crop_b = gray_b[:, :strip_w]

    orb = cv.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=ORB_FAST_THRESHOLD)
    kp_a, des_a = orb.detectAndCompute(crop_a, None)
    kp_b, des_b = orb.detectAndCompute(crop_b, None)

    if des_a is None or des_b is None:
        return OverlapResult(patent_id, page_a, page_b, file_a, file_b, False, "missing ORB descriptors")

    matches = cv.BFMatcher(cv.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]

    if len(good) < MIN_GOOD_MATCHES:
        return OverlapResult(patent_id, page_a, page_b, file_a, file_b, False, "too few good matches", good_matches=len(good))

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    M, mask = cv.estimateAffine2D(
        pts_a, pts_b, method=cv.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_PX,
        maxIters=5000, confidence=0.99, refineIters=20
    )

    if M is None or mask is None:
        return OverlapResult(patent_id, page_a, page_b, file_a, file_b, False, "RANSAC affine fit failed", good_matches=len(good))

    inliers = mask.ravel().astype(bool)
    ia = pts_a[inliers]
    ib = pts_b[inliers]
    n_inliers = int(inliers.sum())
    inlier_ratio = n_inliers / max(1, len(good))
    y_cov = vertical_bin_coverage(ia[:, 1], h)

    ia_full = ia.copy()
    ia_full[:, 0] += a_x0

    dx = float(np.median(ia_full[:, 0] - ib[:, 0]))
    overlap_ratio = 1.0 - (dx / max(1.0, w))
    dy = float(np.median(ia[:, 1] - ib[:, 1]))

    singular_values = np.linalg.svd(M[:, :2], compute_uv=False)
    scale = float(np.mean(singular_values))
    scale_error = abs(scale - 1.0)

    accepted = (
        n_inliers >= MIN_INLIERS
        and inlier_ratio >= MIN_INLIER_RATIO
        and y_cov >= MIN_Y_BIN_COVERAGE
        and MIN_OVERLAP_RATIO <= overlap_ratio <= MAX_OVERLAP_RATIO
        and scale_error <= MAX_RELATIVE_SCALE_ERROR
    )

    reason = "physical scan overlap accepted" if accepted else "no reliable physical scan overlap"
    full_to_strip = np.array([[1.0, 0.0, -a_x0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T_full = affine_to_homogeneous(M) @ full_to_strip

    return OverlapResult(
        patent_id, page_a, page_b, file_a, file_b, accepted, reason,
        good_matches=len(good), inliers=n_inliers, inlier_ratio=float(inlier_ratio),
        y_coverage=float(y_cov), overlap_ratio=float(overlap_ratio),
        scale=scale, scale_error=scale_error, dx=dx, dy=dy,
        transform_a_to_b=T_full, inlier_points_a=ia_full, inlier_points_b=ib,
    )


# =============================================================================
# STAGE A.1/A.2 — WATERFALL: FULL PAGE ROTATION & DEDUPE
# =============================================================================

def get_rotated_versions(img):
    return {
        0: img,
        90: cv.rotate(img, cv.ROTATE_90_CLOCKWISE),
        180: cv.rotate(img, cv.ROTATE_180),
        270: cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    }

def detect_full_page_overlap(gray_a, gray_b, patent_id, page_a, page_b, file_a, file_b):
    h_a, w_a = gray_a.shape[:2]
    h_b, w_b = gray_b.shape[:2]

    if h_a != h_b and h_b > 0:
        scale_b = h_a / h_b
        new_w_b = max(1, int(round(w_b * scale_b)))
        gray_b = cv.resize(gray_b, (new_w_b, h_a), interpolation=cv.INTER_AREA)

    orb = cv.ORB_create(nfeatures=10000)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)

    if des_a is None or len(kp_a) < FULL_PAGE_MIN_MATCH_COUNT:
        return OverlapResult(patent_id, page_a, page_b, file_a, file_b, False, "Not enough ORB features A")

    best_result = None
    max_inliers = 0
    rotated_bs = get_rotated_versions(gray_b)

    for angle, img_b in rotated_bs.items():
        kp_b, des_b = orb.detectAndCompute(img_b, None)
        if des_b is None or len(kp_b) < FULL_PAGE_MIN_MATCH_COUNT:
            continue

        bf = cv.BFMatcher(cv.NORM_HAMMING)
        matches = bf.knnMatch(des_a, des_b, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > FULL_PAGE_MIN_MATCH_COUNT:
            src_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

            # Replaced findHomography with Partial Affine to prevent 3D "bowtie" collapses
            M, mask = cv.estimateAffinePartial2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)
            
            H = None
            if M is not None:
                scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
                if 0.5 < scale < 2.0:  # Enforce realistic scale bounds
                    H = np.eye(3, dtype=np.float64)
                    H[:2, :] = M

            if H is not None:
                inliers = np.sum(mask)
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_result = (angle, img_b, H, mask, inliers, good, kp_a, kp_b)

    if best_result is None or max_inliers < FULL_PAGE_MIN_MATCH_COUNT:
        return OverlapResult(patent_id, page_a, page_b, file_a, file_b, False, "no reliable full-page overlap")

    angle, img_b, H, mask, inliers, good, kp_a, kp_b = best_result
    h_b_rot, w_b_rot = img_b.shape

    corners_b = np.float32([[0, 0], [w_b_rot, 0], [w_b_rot, h_b_rot], [0, h_b_rot]]).reshape(-1, 1, 2)
    projected_corners = cv.perspectiveTransform(corners_b, H)

    min_x, max_x = int(np.min(projected_corners[:, 0, 0])), int(np.max(projected_corners[:, 0, 0]))
    min_y, max_y = int(np.min(projected_corners[:, 0, 1])), int(np.max(projected_corners[:, 0, 1]))

    projected_area = max(1, (max_x - min_x) * (max_y - min_y))
    area_a = w_a * h_a

    intersect_min_x, intersect_max_x = max(0, min_x), min(w_a, max_x)
    intersect_min_y, intersect_max_y = max(0, min_y), min(h_a, max_y)
    intersect_area = max(0, intersect_max_x - intersect_min_x) * max(0, intersect_max_y - intersect_min_y)

    ratio_b_inside_a = intersect_area / projected_area
    ratio_a_inside_b = intersect_area / max(1, area_a)

    if ratio_b_inside_a > INSIDE_RATIO_THRESHOLD or ratio_a_inside_b > INSIDE_RATIO_THRESHOLD:
        accepted = False
        reason = "duplicate_or_variant"
    else:
        accepted = True
        reason = "vertical_or_rotated_continuation"

    try:
        T_full = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return OverlapResult(patent_id, page_a, page_b, file_a, file_b, False, "Homography inversion failed")

    result = OverlapResult(
        patent_id, page_a, page_b, file_a, file_b, accepted, reason,
        good_matches=len(good), inliers=inliers, transform_a_to_b=T_full, angle=angle
    )
    
    inliers_bool = mask.ravel().astype(bool)
    result.inlier_points_a = np.float32([kp_a[m.queryIdx].pt for m in good])[inliers_bool]
    result.inlier_points_b = np.float32([kp_b[m.trainIdx].pt for m in good])[inliers_bool]
    result.H_b_to_a = H
    result.projected_corners = projected_corners
    result.ratio_b_inside_a = ratio_b_inside_a
    result.ratio_a_inside_b = ratio_a_inside_b
    result.img_b_rotated = img_b

    return result


# =============================================================================
# DEBUG & STITCHING
# =============================================================================

def save_overlap_debug(gray_a, gray_b, result, out_path):
    h_a, w_a = gray_a.shape[:2]
    h_b, w_b = gray_b.shape[:2]

    if h_a != h_b and h_b > 0:
        scale_b = h_a / h_b
        gray_b = cv.resize(gray_b, (max(1, int(round(w_b * scale_b))), h_a), interpolation=cv.INTER_AREA)

    h, w = gray_a.shape[:2]
    strip_w = max(64, int(MATCH_STRIP_FRAC * w))
    left = cv.cvtColor(gray_a, cv.COLOR_GRAY2BGR)
    right = cv.cvtColor(gray_b, cv.COLOR_GRAY2BGR)

    cv.rectangle(left, (w - strip_w, 0), (w - 1, h - 1), (180, 0, 180), 3)
    cv.rectangle(right, (0, 0), (strip_w - 1, h - 1), (180, 0, 180), 3)
    canvas = np.hstack([left, right])

    if result.inlier_points_a is not None and result.inlier_points_b is not None:
        for idx in np.linspace(0, len(result.inlier_points_a) - 1, min(250, len(result.inlier_points_a)), dtype=int):
            a = result.inlier_points_a[idx]
            b = result.inlier_points_b[idx]
            cv.line(canvas, (int(a[0]), int(a[1])), (int(b[0] + w), int(b[1])), (0, 220, 0), 1, cv.LINE_AA)

    title = f"p{result.page_a}->{result.page_b} | {result.orientation.upper()} | inliers={result.inliers}"
    cv.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 42), (255, 255, 255), -1)
    cv.putText(canvas, title, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv.LINE_AA)
    cv.imwrite(str(out_path), canvas)


def save_full_page_debug(gray_a, gray_b_rot, result, out_path):
    if gray_b_rot is None or result.H_b_to_a is None: return

    color = (0, 0, 255) if not result.accepted else (0, 255, 0)
    img_a_bgr = cv.cvtColor(gray_a, cv.COLOR_GRAY2BGR)
    img_b_bgr = cv.cvtColor(gray_b_rot, cv.COLOR_GRAY2BGR)

    cv.polylines(img_a_bgr, [np.int32(result.projected_corners)], True, color, 8, cv.LINE_AA)

    h_a, w_a = img_a_bgr.shape[:2]
    h_b, w_b = img_b_bgr.shape[:2]
    canvas_h, canvas_w = max(h_a, h_b), w_a + w_b
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    canvas[:h_a, :w_a] = img_a_bgr
    canvas[:h_b, w_a:w_a+w_b] = img_b_bgr

    if result.inlier_points_a is not None and result.inlier_points_b is not None:
        for a, b in zip(result.inlier_points_a, result.inlier_points_b):
            cv.line(canvas, (int(a[0]), int(a[1])), (int(b[0] + w_a), int(b[1])), color, 1, cv.LINE_AA)

    banner = np.zeros((80, canvas_w, 3), dtype=np.uint8)
    text = f"{result.reason.upper()} | Rot: {result.angle} | Inliers: {result.inliers} | B inside A: {result.ratio_b_inside_a:.2f}"
    cv.putText(banner, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv.LINE_AA)

    cv.imwrite(str(out_path), np.vstack((banner, canvas)))


def apply_homography(H, points):
    if len(points) == 0: return np.empty((0, 2), dtype=np.float64)
    p = np.hstack([points.astype(np.float64), np.ones((len(points), 1))])
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def stitch_run(page_items, overlap_results, out_path):
    if len(page_items) < 2 or len(overlap_results) != len(page_items) - 1: return None
    if any(r.transform_a_to_b is None or not r.accepted for r in overlap_results): return None

    images = []
    for _, path in page_items:
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None: return None
        images.append(img)

    target_h = images[0].shape[0]
    
    for idx in range(1, len(images)):
        r = overlap_results[idx - 1]
        
        # 1. Height normalization (matching Stage A logic)
        ih, iw = images[idx].shape[:2]
        if ih != target_h and ih > 0:
            scale = target_h / ih
            images[idx] = cv.resize(images[idx], (max(1, int(round(iw * scale))), target_h), interpolation=cv.INTER_AREA)

        # 2. Apply Full-Page Rotation if required
        if getattr(r, 'angle', 0) != 0:
            if r.angle == 90: images[idx] = cv.rotate(images[idx], cv.ROTATE_90_CLOCKWISE)
            elif r.angle == 180: images[idx] = cv.rotate(images[idx], cv.ROTATE_180)
            elif r.angle == 270: images[idx] = cv.rotate(images[idx], cv.ROTATE_90_COUNTERCLOCKWISE)

    G = [np.eye(3, dtype=np.float64)]
    for r in overlap_results:
        G.append(G[-1] @ np.linalg.inv(r.transform_a_to_b))

    all_corners = []
    for img, Gi in zip(images, G):
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        all_corners.append(apply_homography(Gi, corners))

    pts = np.vstack(all_corners)
    min_x, min_y = np.floor(pts.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(pts.max(axis=0)).astype(int)
    cw, ch = int(max_x - min_x), int(max_y - min_y)

    if cw <= 0 or ch <= 0 or cw > 40000 or ch > 40000: return None

    shift = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    canvas = np.full((ch, cw), 255, dtype=np.uint8)

    for img, Gi in zip(images, G):
        H = shift @ Gi
        warped = cv.warpPerspective(img, H, (cw, ch), borderValue=255)
        mask = warped < 250
        canvas[mask] = warped[mask]

    cv.imwrite(str(out_path), canvas)
    return {"pages": [p for p, _ in page_items], "output": str(out_path), "width": cw, "height": ch}


def group_pages_by_patent(input_dir):
    grouped = defaultdict(list)
    for path in list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")):
        parsed = parse_filename(path)
        if parsed is None: continue
        grouped[parsed[0]].append((parsed[1], path))

    for patent_id in grouped: grouped[patent_id].sort(key=lambda x: x[0])
    return grouped


# =============================================================================
# MAIN PROCESS PIPELINE
# =============================================================================

def process_patent(patent_id, pages, output_dir, save_debug=False):
    patent_dir = output_dir / patent_id
    meta_path = patent_dir / "metadata.json"

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception: pass

    if save_debug:
        overlap_dir = patent_dir / "overlap"
        overlap_dir.mkdir(parents=True, exist_ok=True)

    overlap_results = []
    
    if len(pages) > 1: print(f"----- {patent_id} ({len(pages)} pages) -----")

    for i in range(len(pages) - 1):
        page_a, path_a = pages[i]
        page_b, path_b = pages[i + 1]
        if page_b != page_a + 1: continue

        gray_a = cv.imread(str(path_a), cv.IMREAD_GRAYSCALE)
        gray_b = cv.imread(str(path_b), cv.IMREAD_GRAYSCALE)
        if gray_a is None or gray_b is None: continue

        # WATERFALL STEP 1: Normal Edge Match
        overlap = detect_scan_overlap(gray_a, gray_b, patent_id, page_a, page_b, path_a.name, path_b.name)
        overlap.orientation = "normal"

        if not overlap.accepted:
            # WATERFALL STEP 1.5: Reverse Edge Match
            rev_overlap = detect_scan_overlap(gray_b, gray_a, patent_id, page_b, page_a, path_b.name, path_a.name)
            if rev_overlap.accepted:
                overlap = rev_overlap
                overlap.orientation = "reverse"
                overlap.page_a, overlap.page_b = page_a, page_b
                overlap.file_a, overlap.file_b = path_a.name, path_b.name
                if overlap.transform_a_to_b is not None:
                    overlap.transform_a_to_b = np.linalg.inv(overlap.transform_a_to_b)
            else:
                # WATERFALL STEP 2 & 3: Full-Page Deduplication & Vertical Check
                full_overlap = detect_full_page_overlap(gray_a, gray_b, patent_id, page_a, page_b, path_a.name, path_b.name)
                if full_overlap.accepted:
                    overlap = full_overlap
                    overlap.orientation = "vertical_or_rotated_continuation"
                elif full_overlap.reason == "duplicate_or_variant":
                    overlap = full_overlap
                    overlap.orientation = "duplicate_or_variant"

        overlap_results.append(overlap)

        if save_debug:
            if overlap.accepted and overlap.orientation == "normal":
                save_overlap_debug(gray_a, gray_b, overlap, overlap_dir / f"pair_{page_a:04d}_{page_b:04d}_NORMAL.png")
            elif overlap.accepted and overlap.orientation == "reverse":
                save_overlap_debug(gray_b, gray_a, overlap, overlap_dir / f"pair_{page_a:04d}_{page_b:04d}_REVERSE.png")
            elif getattr(overlap, 'img_b_rotated', None) is not None:
                save_full_page_debug(gray_a, overlap.img_b_rotated, overlap, overlap_dir / f"pair_{page_a:04d}_{page_b:04d}_{overlap.reason.upper()}.png")

    # Stitch Accepted Runs directly into final_ready_canvases
    final_canvases_dir = patent_dir / "final_ready_canvases"
    final_canvases_dir.mkdir(parents=True, exist_ok=True)
    
    by_pair = {(r.page_a, r.page_b): r for r in overlap_results if r.accepted}
    stitched_runs = []
    start = 0
    while start < len(pages) - 1:
        page_a, _ = pages[start]
        first = by_pair.get((page_a, page_a + 1))
        if first is None:
            start += 1
            continue

        run_pages = [pages[start], pages[start + 1]]
        run_results = [first]
        j = start + 1
        while j < len(pages) - 1:
            pa, _ = pages[j]
            pb, _ = pages[j + 1]
            r = by_pair.get((pa, pb))
            if r is None: break
            run_pages.append(pages[j + 1])
            run_results.append(r)
            j += 1

        out_path = final_canvases_dir / f"sheet_pages_{run_pages[0][0]:04d}_{run_pages[-1][0]:04d}.png"
        info = stitch_run(run_pages, run_results, out_path)
        if info: stitched_runs.append(info)
        start = j

    # --- UNIFIED CANVAS LOGIC ---
    # 1. Figure out which pages are locked inside a stitch
    stitched_page_nums = set()
    for run in stitched_runs:
        stitched_page_nums.update(run["pages"])

    # 2. Figure out which pages were explicitly dropped as duplicates
    dropped_page_nums = set()
    for r in overlap_results:
        if r.reason == "duplicate_or_variant":
            dropped_page_nums.add(r.page_b) # B is dropped

    # 3. Find the survivors (Standalone pages)
    standalone_pages = []
    for page_num, path in pages:
        if page_num not in stitched_page_nums and page_num not in dropped_page_nums:
            standalone_pages.append(page_num)
            # Copy surviving standalone page to final folder
            shutil.copy(str(path), final_canvases_dir / path.name)

    metadata = {
        "patent_id": patent_id,
        "source_pages": [{"page_number": p, "filename": path.name} for p, path in pages],
        "num_pages": len(pages),
        "summary": {
            "adjacent_pairs_tested": len(overlap_results),
            "physical_overlap_accepted": sum(r.accepted and r.orientation in ["normal", "reverse"] for r in overlap_results),
            "reversed_overlap_accepted": sum(r.accepted and r.orientation == "reverse" for r in overlap_results),
            "vertical_rotated_continuations": sum(r.accepted and r.reason == "vertical_or_rotated_continuation" for r in overlap_results),
            "duplicates_dropped": sum(r.reason == "duplicate_or_variant" for r in overlap_results),
            "stitched_runs": len(stitched_runs),
            "standalone_pages_preserved": len(standalone_pages)
        },
        "stitched_runs": stitched_runs,
        "standalone_pages": standalone_pages
    }

    with open(meta_path, "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)
    return metadata


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="sketches_from_7000_pdfs")
    parser.add_argument("--output-dir", type=str, default="patent_cross_page_results")
    parser.add_argument("--save-debug", action="store_true", help="Flag to save intermediate overlap/stitch debug images")
    args = parser.parse_args()

    start_time = time.time()
    
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    patents = group_pages_by_patent(input_dir)
    if not patents: return
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_meta = [process_patent(pid, patents[pid], output_dir, args.save_debug) for pid in sorted(patents)]

    end_time = time.time()
    execution_time_seconds = round(end_time - start_time, 2)

    overall = {
        "execution_metrics": {
            "execution_time_seconds": execution_time_seconds,
            "execution_time_minutes": round(execution_time_seconds / 60, 2)
        },
        "dataset_composition": {
            "total_patents": len(patents),
            "total_extracted_pages": sum(m["num_pages"] for m in all_meta),
        },
        "overlap_results": {
            "adjacent_pairs_tested": sum(m["summary"]["adjacent_pairs_tested"] for m in all_meta),
            "edge_overlap_accepted": sum(m["summary"]["physical_overlap_accepted"] for m in all_meta),
            "vertical_rotated_continuations": sum(m["summary"]["vertical_rotated_continuations"] for m in all_meta),
            "duplicates_dropped": sum(m["summary"]["duplicates_dropped"] for m in all_meta),
            "stitched_runs_created": sum(m["summary"]["stitched_runs"] for m in all_meta),
            "standalone_pages_preserved": sum(m["summary"]["standalone_pages_preserved"] for m in all_meta)
        }
    }

    with open(output_dir / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"Pipeline Execution Complete in {execution_time_seconds}s. Metadata saved.")

if __name__ == "__main__":
    main()
