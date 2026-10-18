"""
PATENT-LEVEL CROSS-PAGE PIPELINE

Input filenames:
    2382_2387_page_2.png
    2382_2387_page_3.png
    2382_2387_page_4.png

The prefix before '_page_' is the patent ID. Pages are NEVER compared
across different patent IDs.

For every patent the output is:

output_dir/
    2382_2387/
        metadata.json
        overlap/
            pair_0002_0003.png
            pair_0003_0004.png
            overlap_results.csv
        stitched/
            sheet_pages_0002_0004.png
            ...

STAGE A: physical scan overlap
    RIGHT side of page N <-> LEFT side of page N+1
    ORB + ratio test + RANSAC affine fit + global coverage checks.

STAGE B: stitch consecutive accepted overlap pairs into larger sheets.

No OCR is used here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import csv
import json
import re

import cv2 as cv
import numpy as np


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
INPUT_DIR = Path("sketches_from_1000_pdfs")
OUTPUT_DIR = Path("patent_cross_page_results")

# Stage A
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

# Example: 2382_2387_page_2.png
FILENAME_RE = re.compile(
    r"^(?P<patent>\d+_\d+)_page_(?P<page>\d+)\.(?P<ext>png|jpg|jpeg)$",
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


def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("patent"), int(m.group("page"))


def binarize(gray):
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv.threshold(
        blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )
    return ink


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


def safe_corr(a, b):
    if a.size == 0 or b.size == 0 or a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a.astype(np.float32), b.astype(np.float32))[0, 1])


# =============================================================================
# STAGE A — PHYSICAL SCAN OVERLAP
# =============================================================================

def detect_scan_overlap(gray_a, gray_b, patent_id, page_a, page_b, file_a, file_b):
    h, w = gray_a.shape[:2]
    strip_w = max(64, int(MATCH_STRIP_FRAC * w))
    a_x0 = w - strip_w

    crop_a = gray_a[:, a_x0:w]
    crop_b = gray_b[:, :strip_w]

    orb = cv.ORB_create(
        nfeatures=ORB_FEATURES,
        fastThreshold=ORB_FAST_THRESHOLD,
    )
    kp_a, des_a = orb.detectAndCompute(crop_a, None)
    kp_b, des_b = orb.detectAndCompute(crop_b, None)

    if des_a is None or des_b is None:
        return OverlapResult(
            patent_id, page_a, page_b, file_a, file_b,
            False, "missing ORB descriptors"
        )

    matches = cv.BFMatcher(cv.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]

    if len(good) < MIN_GOOD_MATCHES:
        return OverlapResult(
            patent_id, page_a, page_b, file_a, file_b,
            False, "too few good matches", good_matches=len(good)
        )

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    M, mask = cv.estimateAffine2D(
        pts_a,
        pts_b,
        method=cv.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_PX,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )

    if M is None or mask is None:
        return OverlapResult(
            patent_id, page_a, page_b, file_a, file_b,
            False, "RANSAC affine fit failed", good_matches=len(good)
        )

    inliers = mask.ravel().astype(bool)
    ia = pts_a[inliers]
    ib = pts_b[inliers]
    n_inliers = int(inliers.sum())
    inlier_ratio = n_inliers / max(1, len(good))
    y_cov = vertical_bin_coverage(ia[:, 1], h)

    # Convert A-strip coordinates back to full A-page coordinates.
    ia_full = ia.copy()
    ia_full[:, 0] += a_x0

    # If A right edge overlaps B left edge:
    #     x_b ~= x_a - overlap_width
    # so overlap_width ~= x_a - x_b.
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

    if accepted:
        reason = "physical scan overlap accepted"
    elif n_inliers >= MIN_INLIERS and y_cov < MIN_Y_BIN_COVERAGE:
        reason = "localized match; not accepted as whole-sheet overlap"
    else:
        reason = "no reliable physical scan overlap"

    # Full-page A -> page-B transform.
    full_to_strip = np.array(
        [[1.0, 0.0, -a_x0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T_full = affine_to_homogeneous(M) @ full_to_strip

    return OverlapResult(
        patent_id, page_a, page_b, file_a, file_b,
        accepted, reason,
        good_matches=len(good),
        inliers=n_inliers,
        inlier_ratio=float(inlier_ratio),
        y_coverage=float(y_cov),
        overlap_ratio=float(overlap_ratio),
        scale=scale,
        scale_error=scale_error,
        dx=dx,
        dy=dy,
        transform_a_to_b=T_full,
        inlier_points_a=ia_full,
        inlier_points_b=ib,
    )


# =============================================================================
# DEBUG OUTPUT
# =============================================================================

def save_overlap_debug(gray_a, gray_b, result, out_path):
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

    title = (
        f"p{result.page_a}->{result.page_b} OVERLAP={result.accepted} "
        f"good={result.good_matches} inliers={result.inliers} "
        f"ratio={result.inlier_ratio:.2f} ycov={result.y_coverage:.2f} "
        f"overlap={result.overlap_ratio:.2f}"
    )
    cv.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 42), (255, 255, 255), -1)
    cv.putText(canvas, title, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv.LINE_AA)
    cv.imwrite(str(out_path), canvas)

# =============================================================================
# STAGE C — STITCH ACCEPTED OVERLAP RUNS
# =============================================================================

def apply_homography(H, points):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    p = np.hstack([points.astype(np.float64), np.ones((len(points), 1))])
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def stitch_run(page_items, overlap_results, out_path):
    """page_items = [(page_num, Path), ...] for one accepted overlap run."""
    if len(page_items) < 2 or len(overlap_results) != len(page_items) - 1:
        return None
    if any(r.transform_a_to_b is None or not r.accepted for r in overlap_results):
        return None

    images = []
    for _, path in page_items:
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            return None
        images.append(img)

    # G[i] maps page i coordinates into the global coordinate frame of page 0.
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

    if cw <= 0 or ch <= 0 or cw > 30000 or ch > 30000:
        return None

    shift = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    canvas = np.full((ch, cw), 255, dtype=np.uint8)

    for img, Gi in zip(images, G):
        H = shift @ Gi
        warped = cv.warpPerspective(img, H, (cw, ch), borderValue=255)
        mask = warped < 250
        canvas[mask] = warped[mask]

    cv.imwrite(str(out_path), canvas)
    return {
        "pages": [p for p, _ in page_items],
        "output": str(out_path),
        "width": cw,
        "height": ch,
    }


# =============================================================================
# PATENT GROUPING + PROCESSING
# =============================================================================

def group_pages_by_patent(input_dir):
    grouped = defaultdict(list)
    for path in list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")):
        parsed = parse_filename(path)
        if parsed is None:
            print(f"[SKIP] unexpected filename: {path.name}")
            continue
        patent_id, page_num = parsed
        grouped[patent_id].append((page_num, path))

    for patent_id in grouped:
        grouped[patent_id].sort(key=lambda x: x[0])
    return grouped


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def process_patent(patent_id, pages, output_dir):
    patent_dir = output_dir / patent_id
    overlap_dir = patent_dir / "overlap"
    stitched_dir = patent_dir / "stitched"

    overlap_dir.mkdir(parents=True, exist_ok=True)
    stitched_dir.mkdir(parents=True, exist_ok=True)

    overlap_results = []

    print(f"\n===== {patent_id}: {len(pages)} pages =====")

    for i in range(len(pages) - 1):
        page_a, path_a = pages[i]
        page_b, path_b = pages[i + 1]

        # IMPORTANT: do not compare page 2 directly to page 4 if page 3 is absent.
        if page_b != page_a + 1:
            print(f"  {page_a}->{page_b}: skipped (not consecutive page numbers)")
            continue

        gray_a = cv.imread(str(path_a), cv.IMREAD_GRAYSCALE)
        gray_b = cv.imread(str(path_b), cv.IMREAD_GRAYSCALE)
        if gray_a is None or gray_b is None:
            print(f"  [WARN] could not read {path_a.name} or {path_b.name}")
            continue

        overlap = detect_scan_overlap(
            gray_a, gray_b, patent_id,
            page_a, page_b,
            path_a.name, path_b.name,
        )
        
        overlap.orientation = "normal" # Add a small tracking attribute

        if not overlap.accepted:
            rev_overlap = detect_scan_overlap(
                gray_b, gray_a, patent_id,
                page_b, page_a, path_b.name, path_a.name,
            )
            if rev_overlap.accepted:
                overlap = rev_overlap
                overlap.orientation = "reverse"
                
                # Restore original sequential IDs so the stitch_run dictionary can find it
                overlap.page_a = page_a
                overlap.page_b = page_b
                overlap.file_a = path_a.name
                overlap.file_b = path_b.name
                
                # Invert the transform matrix so it correctly maps A -> B instead of B -> A
                if overlap.transform_a_to_b is not None:
                    overlap.transform_a_to_b = np.linalg.inv(overlap.transform_a_to_b)

        overlap_results.append(overlap)

        if overlap.orientation == "reverse":
            # Pass the swapped images so the RANSAC lines match the visual
            save_overlap_debug(
                gray_b, gray_a, overlap,
                overlap_dir / f"pair_{page_a:04d}_{page_b:04d}_REVERSE.png",
            )
        else:
            save_overlap_debug(
                gray_a, gray_b, overlap,
                overlap_dir / f"pair_{page_a:04d}_{page_b:04d}_NORMAL.png",
            )

        print(
            f"  {page_a}->{page_b} | "
            f"OVERLAP={overlap.accepted} ({overlap.orientation}) |"
            f"inliers={overlap.inliers} "
            f"ycov={overlap.y_coverage:.2f} "
            f"ov={overlap.overlap_ratio:.2f}"
        )

    # ----- CSVs -----
    overlap_rows = []
    for r in overlap_results:
        overlap_rows.append({
            "patent_id": r.patent_id,
            "page_a": r.page_a,
            "page_b": r.page_b,
            "file_a": r.file_a,
            "file_b": r.file_b,
            "accepted": r.accepted,
            "reason": r.reason,
            "good_matches": r.good_matches,
            "inliers": r.inliers,
            "inlier_ratio": round(r.inlier_ratio, 4),
            "y_coverage": round(r.y_coverage, 4),
            "overlap_ratio": round(r.overlap_ratio, 4),
            "scale": round(r.scale, 4),
            "scale_error": round(r.scale_error, 4),
            "dx": round(r.dx, 2),
            "dy": round(r.dy, 2),
        })
    write_csv(overlap_dir / "overlap_results.csv", overlap_rows)

    # ----- Stitch consecutive accepted OVERLAP pairs -----
    by_pair = {(r.page_a, r.page_b): r for r in overlap_results if r.accepted}
    stitched_runs = []

    start = 0
    while start < len(pages) - 1:
        # Find an accepted pair starting at this page.
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
            if r is None:
                break
            run_pages.append(pages[j + 1])
            run_results.append(r)
            j += 1

        out_path = stitched_dir / f"sheet_pages_{run_pages[0][0]:04d}_{run_pages[-1][0]:04d}.png"
        info = stitch_run(run_pages, run_results, out_path)
        if info:
            stitched_runs.append(info)

        start = j

    # ----- Metadata -----
    metadata = {
        "patent_id": patent_id,
        "source_pages": [
            {"page_number": p, "filename": path.name}
            for p, path in pages
        ],
        "num_pages": len(pages),
        "config": {
            "match_strip_frac": MATCH_STRIP_FRAC,
            "orb_features": ORB_FEATURES,
            "orb_fast_threshold": ORB_FAST_THRESHOLD,
            "ratio_test": RATIO_TEST,
            "ransac_reproj_px": RANSAC_REPROJ_PX,
            "min_good_matches": MIN_GOOD_MATCHES,
            "min_inliers": MIN_INLIERS,
            "min_inlier_ratio": MIN_INLIER_RATIO,
            "vertical_bins": VERTICAL_BINS,
            "min_y_bin_coverage": MIN_Y_BIN_COVERAGE,
            "min_overlap_ratio": MIN_OVERLAP_RATIO,
            "max_overlap_ratio": MAX_OVERLAP_RATIO,
            "max_relative_scale_error": MAX_RELATIVE_SCALE_ERROR,
        },
        "summary": {
            "adjacent_pairs_tested": len(overlap_results),
            "physical_overlap_accepted": sum(r.accepted for r in overlap_results),
            "physical_overlap_rejected": sum(not r.accepted for r in overlap_results),
            "stitched_runs": len(stitched_runs),
        },
        "stitched_runs": stitched_runs,
    }

    with open(patent_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage A: Physical Scan Overlap (OpenCV)")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing source pages")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save Stage A results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    patents = group_pages_by_patent(input_dir)
    if not patents:
        raise RuntimeError(f"No matching pages found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(patents)} patents in {input_dir}")

    all_meta = []
    for patent_id in sorted(patents):
        all_meta.append(process_patent(patent_id, patents[patent_id], output_dir))

    total_pages = sum(m["num_pages"] for m in all_meta)
    total_pairs = sum(m["summary"]["adjacent_pairs_tested"] for m in all_meta)
    total_overlap = sum(m["summary"]["physical_overlap_accepted"] for m in all_meta)
    total_stitched = sum(m["summary"]["stitched_runs"] for m in all_meta)

    overall = {
        "total_patents": len(all_meta),
        "total_pages": total_pages,
        "adjacent_pairs_tested": total_pairs,
        "physical_overlap_accepted": total_overlap,
        "stitched_runs": total_stitched,
    }

    with open(output_dir / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n========== OVERALL ==========")
    print(json.dumps(overall, indent=2))
    print(f"Output directory: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
