"""Page-first extraction and cross-page assembly for patent drawings.

The extractor consumes PDFs plus the page-level CSV produced by the sibling
``page_classifier`` package.  It renders complete drawing pages, joins pages
that are confidently parts of the same scanned plate, preserves unrelated
plates as separate images, and marks uncertain boundaries as HOLD.

No figure-level crops, PDF-specific coordinates, or manual masks are used.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi


ALGORITHM_VERSION = "page-extractor-v1-v5f"
DEFAULT_RENDER_DPI = 150
ProgressCallback = Callable[[str], None]


def print_progress(message: str) -> None:
    """Print progress immediately, including in non-interactive batch jobs."""
    print(message, flush=True)

PAIR_COLUMNS = [
    "pdf", "left_page", "right_page", "base_decision",
    "overlap_good_matches", "overlap_inliers", "overlap_inlier_ratio",
    "overlap_median_error_px", "overlap_source_coverage",
    "overlap_area_ratio", "boundary_order", "boundary_correlation",
    "boundary_margin", "overlap_error", "supported_overlap_seed",
    "supported_component_bridge",
]

BOUNDARY_COLUMNS = [
    "pass", "pdf", "left_page", "right_page", "left_component_size",
    "right_component_size", "boundary_correlation", "boundary_margin",
    "boundary_order", "supported_component_bridge", "eligible", "decision",
    "direction", "local_inliers", "translation_residual_px",
    "matched_ink_ratio", "matched_ink_pixels", "validation_tier", "error",
]


def find_pdf_inputs(input_path: Path, recursive: bool = False) -> list[Path]:
    """Return PDF inputs in deterministic order and reject duplicate names."""
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*.pdf") if recursive else input_path.glob("*.pdf")
    pdf_paths = sorted(path.resolve() for path in iterator if path.is_file())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under: {input_path}")
    names = [path.name for path in pdf_paths]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "PDF filenames must be unique because predictions use filenames: "
            + ", ".join(duplicates)
        )
    return pdf_paths


def load_prediction_table(path: Path) -> pd.DataFrame:
    """Load and validate page-classifier output."""
    frame = pd.read_csv(path.resolve())
    required = {"pdf", "page", "predicted_page_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["pdf"] = frame["pdf"].astype(str)
    frame["page"] = pd.to_numeric(frame["page"], errors="raise").astype(int)
    frame["predicted_page_type"] = frame["predicted_page_type"].astype(str).str.lower()
    invalid = sorted(set(frame["predicted_page_type"]) - {"text", "drawing"})
    if invalid:
        raise ValueError(f"Unexpected predicted_page_type values: {invalid}")
    if frame.duplicated(["pdf", "page"]).any():
        duplicates = frame.loc[frame.duplicated(["pdf", "page"], keep=False), ["pdf", "page"]]
        raise ValueError(f"Duplicate PDF/page predictions found:\n{duplicates.to_string(index=False)}")
    return frame.sort_values(["pdf", "page"]).reset_index(drop=True)


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def significant_content_bounds(
    image: np.ndarray,
    threshold: int = 242,
    minimum_component_pixels: int = 12,
    padding: int = 18,
) -> tuple[int, int, int, int]:
    ink = (image < threshold).astype(np.uint8)
    height, width = ink.shape
    ink[:max(2, int(height * 0.008)), :] = 0
    ink[-max(2, int(height * 0.008)):, :] = 0
    ink[:, :max(2, int(width * 0.008))] = 0
    ink[:, -max(2, int(width * 0.008)):] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count > 1:
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_component_pixels
        cleaned = keep[labels]
    else:
        cleaned = ink.astype(bool)
    ys, xs = np.where(cleaned)
    if not len(xs):
        return 0, 0, width, height
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + padding + 1),
        min(height, int(ys.max()) + padding + 1),
    )


def trim_outer_blank(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = significant_content_bounds(image)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def register_overlap(
    left: np.ndarray,
    right: np.ndarray,
    ratio: float = 0.76,
    ransac_threshold: float = 4.0,
) -> tuple[np.ndarray, dict]:
    detector = cv2.SIFT_create(
        nfeatures=10000, contrastThreshold=0.02, edgeThreshold=12
    )
    left_keypoints, left_descriptors = detector.detectAndCompute(left, None)
    right_keypoints, right_descriptors = detector.detectAndCompute(right, None)
    if left_descriptors is None or right_descriptors is None:
        raise RuntimeError("SIFT descriptors were not found")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(right_descriptors, left_descriptors, k=2)
    good = [first for first, second in forward if first.distance < ratio * second.distance]
    if len(good) < 8:
        raise RuntimeError(f"Insufficient overlap matches: {len(good)}")

    source_points = np.float32([right_keypoints[item.queryIdx].pt for item in good])
    destination_points = np.float32([left_keypoints[item.trainIdx].pt for item in good])
    homography, mask = cv2.findHomography(
        source_points, destination_points, cv2.RANSAC, ransac_threshold
    )
    if homography is None or mask is None:
        raise RuntimeError("Overlap homography failed")
    inliers = mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    if inlier_count < 8:
        raise RuntimeError(f"Insufficient RANSAC inliers: {inlier_count}")
    projected = cv2.perspectiveTransform(
        source_points[inliers].reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    median_error = float(
        np.median(np.linalg.norm(projected - destination_points[inliers], axis=1))
    )

    right_corners = np.float32(
        [[0, 0], [right.shape[1], 0], [right.shape[1], right.shape[0]], [0, right.shape[0]]]
    )
    transformed_right = cv2.perspectiveTransform(
        right_corners.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    left_corners = np.float32(
        [[0, 0], [left.shape[1], 0], [left.shape[1], left.shape[0]], [0, left.shape[0]]]
    )
    all_corners = np.vstack([left_corners, transformed_right])
    minimum = np.floor(all_corners.min(axis=0)).astype(int)
    maximum = np.ceil(all_corners.max(axis=0)).astype(int)
    translation = np.array(
        [[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=float
    )
    canvas_width, canvas_height = (maximum - minimum).astype(int)
    if canvas_width <= 0 or canvas_height <= 0:
        raise RuntimeError("Invalid overlap canvas size")
    warped_right = cv2.warpPerspective(
        right, translation @ homography, (canvas_width, canvas_height), borderValue=255
    )
    warped_left = cv2.warpPerspective(
        left, translation, (canvas_width, canvas_height), borderValue=255
    )
    combined, crop_box = trim_outer_blank(np.minimum(warped_left, warped_right))
    source_hull_area = abs(
        float(cv2.contourArea(cv2.convexHull(source_points[inliers])))
    )
    return combined, {
        "good_matches": len(good),
        "inliers": inlier_count,
        "inlier_ratio": inlier_count / len(good),
        "median_error_px": median_error,
        "source_coverage": source_hull_area / max(float(right.size), 1.0),
        "homography": homography,
        "crop_box": crop_box,
    }


def boundary_profile(
    image: np.ndarray,
    side: str,
    strip_width: int,
    threshold: int = 205,
    smooth_sigma: int = 4,
) -> tuple[np.ndarray, int]:
    ink = image < threshold
    height, width = ink.shape
    ink[:int(height * 0.05), :] = False
    ink[int(height * 0.95):, :] = False
    width_used = min(strip_width, max(1, width // 3))
    strip = ink[:, -width_used:] if side == "right" else ink[:, :width_used]
    return ndi.gaussian_filter1d(strip.sum(axis=1).astype(float), smooth_sigma), width_used


def trim_seam_margin(
    image: np.ndarray,
    side: str,
    threshold: int = 238,
    minimum_component_pixels: int = 15,
) -> tuple[np.ndarray, int]:
    ink = image < threshold
    height, width = ink.shape
    ink[:int(height * 0.04), :] = False
    ink[int(height * 0.96):, :] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    if count > 1:
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_component_pixels
        ink = keep[labels]
    _, xs = np.where(ink)
    if not len(xs):
        return image, 0
    if side == "right":
        stop = min(width, int(xs.max()) + 1)
        return image[:, :stop], int(width - stop)
    start = max(0, int(xs.min()))
    return image[:, start:], int(start)


def estimate_boundary_shift(
    left: np.ndarray,
    right: np.ndarray,
    maximum_shift: int = 420,
    strip_widths: tuple[int, ...] = (80, 120, 160, 220),
) -> dict:
    scores = []
    for requested_width in strip_widths:
        left_profile, left_width = boundary_profile(left, "right", requested_width)
        right_profile, right_width = boundary_profile(right, "left", requested_width)
        for profile_shift in range(-maximum_shift, maximum_shift + 1):
            left_start = max(int(left.shape[0] * 0.05), int(right.shape[0] * 0.05) - profile_shift)
            left_stop = min(int(left.shape[0] * 0.95), int(right.shape[0] * 0.95) - profile_shift)
            if left_stop - left_start < min(350, int(min(left.shape[0], right.shape[0]) * 0.35)):
                continue
            left_values = left_profile[left_start:left_stop]
            right_values = right_profile[left_start + profile_shift:left_stop + profile_shift]
            if left_values.std() < 1e-6 or right_values.std() < 1e-6:
                continue
            left_values = (left_values - left_values.mean()) / (left_values.std() + 1e-6)
            right_values = (right_values - right_values.mean()) / (right_values.std() + 1e-6)
            scores.append((float(np.mean(left_values * right_values)), profile_shift, int(min(left_width, right_width))))
    if not scores:
        raise RuntimeError("Boundary shift search produced no valid comparison")
    scores.sort(reverse=True)
    best_score, best_profile_shift, best_width = scores[0]
    separated = [
        item for item in scores
        if item[2] == best_width and abs(item[1] - best_profile_shift) >= 25
    ]
    runner_up = separated[0][0] if separated else scores[-1][0]
    return {
        "profile_shift": int(best_profile_shift),
        "right_y_offset": int(-best_profile_shift),
        "correlation": best_score,
        "margin": float(best_score - runner_up),
        "selected_strip_width": int(best_width),
    }


def join_boundary(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict]:
    left_piece, left_removed = trim_seam_margin(left, "right")
    right_piece, right_removed = trim_seam_margin(right, "left")
    shift = estimate_boundary_shift(left_piece, right_piece)
    right_y = int(shift["right_y_offset"])
    minimum_y = min(0, right_y)
    maximum_y = max(left_piece.shape[0], right_y + right_piece.shape[0])
    canvas = np.full(
        (maximum_y - minimum_y, left_piece.shape[1] + right_piece.shape[1]),
        255,
        np.uint8,
    )
    left_y = -minimum_y
    right_canvas_y = right_y - minimum_y
    canvas[left_y:left_y + left_piece.shape[0], :left_piece.shape[1]] = left_piece
    canvas[
        right_canvas_y:right_canvas_y + right_piece.shape[0], left_piece.shape[1]:
    ] = right_piece
    combined, crop_box = trim_outer_blank(canvas)
    shift.update({
        "left_margin_removed_px": left_removed,
        "right_margin_removed_px": right_removed,
        "crop_box": crop_box,
    })
    return combined, shift


def local_translation_consensus(
    left: np.ndarray,
    right: np.ndarray,
    coarse_y: int,
    maximum_band: int = 1100,
    bin_size: int = 20,
    radius: int = 30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    left_piece, left_removed = trim_seam_margin(left, "right")
    right_piece, right_removed = trim_seam_margin(right, "left")
    band_width = int(min(maximum_band, left_piece.shape[1] * 0.72, right_piece.shape[1] * 0.72))
    left_start = left_piece.shape[1] - band_width
    left_band = left_piece[:, left_start:]
    right_band = right_piece[:, :band_width]
    sift = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.012, edgeThreshold=14)
    key_left, desc_left = sift.detectAndCompute(left_band, None)
    key_right, desc_right = sift.detectAndCompute(right_band, None)
    if desc_left is None or desc_right is None:
        raise RuntimeError("No local descriptors at seam")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = []
    for pair in matcher.knnMatch(desc_left, desc_right, k=2):
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance >= 0.86 * second.distance:
            continue
        point_left = np.array(key_left[first.queryIdx].pt, dtype=np.float64) + np.array([left_start, 0.0])
        point_right = np.array(key_right[first.trainIdx].pt, dtype=np.float64)
        size_ratio = key_left[first.queryIdx].size / max(key_right[first.trainIdx].size, 1e-6)
        if not 0.52 <= size_ratio <= 1.92:
            continue
        displacement = point_left - point_right
        overlap = left_piece.shape[1] - displacement[0]
        if not 12 <= overlap <= band_width:
            continue
        if abs(displacement[1] - coarse_y) > 440:
            continue
        candidates.append((displacement[0], displacement[1], first.distance))
    if len(candidates) < 6:
        raise RuntimeError(f"Insufficient local translation votes: {len(candidates)}")
    votes = np.asarray(candidates, dtype=np.float64)
    bins = np.round(votes[:, :2] / bin_size).astype(int)
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    best = None
    for index in np.argsort(counts)[::-1][:12]:
        center = unique_bins[index].astype(np.float64) * bin_size
        distance = np.linalg.norm(votes[:, :2] - center, axis=1)
        inliers = votes[distance <= radius]
        if len(inliers) < 3:
            continue
        location = np.median(inliers[:, :2], axis=0)
        residual = np.linalg.norm(inliers[:, :2] - location, axis=1)
        proposal = {
            "right_x": int(round(location[0])),
            "right_y": int(round(location[1])),
            "inliers": int(len(inliers)),
            "median_residual_px": float(np.median(residual)),
            "vote_count": int(len(votes)),
            "left_margin_removed_px": int(left_removed),
            "right_margin_removed_px": int(right_removed),
        }
        proposal["overlap_px"] = int(left_piece.shape[1] - proposal["right_x"])
        if best is None or (proposal["inliers"], -proposal["median_residual_px"]) > (best["inliers"], -best["median_residual_px"]):
            best = proposal
    if best is None:
        raise RuntimeError("No stable local translation cluster")
    return left_piece, right_piece, best


def compose_with_seam_cut(
    left: np.ndarray, right: np.ndarray, right_x: int, right_y: int
) -> tuple[np.ndarray, dict]:
    minimum_x, minimum_y = min(0, right_x), min(0, right_y)
    maximum_x = max(left.shape[1], right_x + right.shape[1])
    maximum_y = max(left.shape[0], right_y + right.shape[0])
    canvas = np.full((maximum_y - minimum_y, maximum_x - minimum_x), 255, np.uint8)
    left_x, left_y = -minimum_x, -minimum_y
    right_canvas_x, right_canvas_y = right_x - minimum_x, right_y - minimum_y
    canvas[left_y:left_y + left.shape[0], left_x:left_x + left.shape[1]] = left
    overlap_start = max(left_x, right_canvas_x)
    overlap_stop = min(left_x + left.shape[1], right_canvas_x + right.shape[1])
    cut_x = (overlap_start + overlap_stop) // 2 if overlap_stop > overlap_start else overlap_start
    right_target = canvas[
        right_canvas_y:right_canvas_y + right.shape[0],
        right_canvas_x:right_canvas_x + right.shape[1],
    ]
    columns = np.arange(right.shape[1]) + right_canvas_x
    use_right = columns >= cut_x
    global_rows = np.arange(right.shape[0]) + right_canvas_y
    outside_left_y = (global_rows < left_y) | (global_rows >= left_y + left.shape[0])
    right_target[:, use_right] = right[:, use_right]
    if np.any(outside_left_y):
        right_target[outside_left_y, :] = right[outside_left_y, :]
    canvas[
        right_canvas_y:right_canvas_y + right.shape[0],
        right_canvas_x:right_canvas_x + right.shape[1],
    ] = right_target
    combined, crop_box = trim_outer_blank(canvas)
    return combined, {"cut_x_canvas": int(cut_x), "crop_box": crop_box}


def seam_duplicate_score(
    left: np.ndarray,
    right: np.ndarray,
    right_x: int,
    right_y: int,
    tolerance: float = 3.0,
) -> dict:
    x0, x1 = max(0, right_x), min(left.shape[1], right_x + right.shape[1])
    y0, y1 = max(0, right_y), min(left.shape[0], right_y + right.shape[0])
    if x1 <= x0 or y1 <= y0:
        return {"matched_ink_ratio": 0.0, "matched_ink_pixels": 0}
    left_patch = left[y0:y1, x0:x1]
    right_patch = right[y0 - right_y:y1 - right_y, x0 - right_x:x1 - right_x]
    left_ink, right_ink = left_patch < 205, right_patch < 205
    distance_to_right = cv2.distanceTransform((~right_ink).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_left = cv2.distanceTransform((~left_ink).astype(np.uint8), cv2.DIST_L2, 3)
    matched_left = int(np.count_nonzero(left_ink & (distance_to_right <= tolerance)))
    matched_right = int(np.count_nonzero(right_ink & (distance_to_left <= tolerance)))
    denominator = max(int(left_ink.sum() + right_ink.sum()), 1)
    return {
        "matched_ink_ratio": float((matched_left + matched_right) / denominator),
        "matched_ink_pixels": int(min(matched_left, matched_right)),
    }


class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _load_font(size: int = 16, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _write_review_sheets(
    output_manifest: pd.DataFrame,
    boundary_metrics: pd.DataFrame,
    review_root: Path,
) -> int:
    hold_pdfs = set(
        output_manifest.loc[
            output_manifest["output_type"].astype(str).str.startswith("HOLD"), "pdf"
        ]
    )
    optimized_pdfs = set(
        boundary_metrics.loc[
            boundary_metrics["decision"].isin(["AUTO_LOCAL_SEAM", "AUTO_ITERATIVE_BRIDGE"]),
            "pdf",
        ]
    ) if len(boundary_metrics) else set()
    review_pdfs = sorted(hold_pdfs | optimized_pdfs)
    title_font, body_font = _load_font(22, True), _load_font(14, False)
    written = 0
    for pdf_name in review_pdfs:
        rows = output_manifest[output_manifest["pdf"] == pdf_name]
        if rows.empty:
            continue
        columns = min(3, len(rows))
        rows_count = math.ceil(len(rows) / columns)
        sheet = Image.new("RGB", (columns * 600, 100 + rows_count * 700), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 18), f"{pdf_name} | delivered outputs", fill=(20, 45, 75), font=title_font)
        for index, record in enumerate(rows.itertuples(index=False)):
            with Image.open(record.output_path) as opened:
                thumb = ImageOps.contain(opened.convert("RGB"), (550, 610), Image.Resampling.LANCZOS)
            column, row_index = index % columns, index // columns
            x0, y0 = column * 600, 85 + row_index * 700
            left = x0 + 25 + (550 - thumb.width) // 2
            top = y0 + 55 + (610 - thumb.height) // 2
            sheet.paste(thumb, (left, top))
            color = (180, 70, 35) if record.output_type.startswith("HOLD") else (20, 125, 75)
            draw.rectangle((x0 + 10, y0 + 35, x0 + 590, y0 + 680), outline=color, width=4)
            draw.text((x0 + 20, y0 + 8), f"{record.output_type} | pages {record.pages}", fill=color, font=body_font)
        sheet.save(review_root / f"{Path(pdf_name).stem}__review.png", optimize=True)
        written += 1
    return written


def extract_drawings(
    pdf_paths: list[Path],
    predictions: pd.DataFrame,
    output_root: Path,
    render_dpi: int = DEFAULT_RENDER_DPI,
    write_review_sheets: bool = True,
    progress: ProgressCallback = print_progress,
) -> dict:
    """Run the full page-first delivery pipeline and return its summary."""
    started = time.time()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output_root}"
        )
    pdf_paths = [Path(path).resolve() for path in pdf_paths]
    if not pdf_paths:
        raise ValueError("No PDFs were selected")
    pdf_names = [path.name for path in pdf_paths]
    if len(pdf_names) != len(set(pdf_names)):
        raise ValueError("Selected PDF filenames must be unique")
    missing_predictions = sorted(set(pdf_names) - set(predictions["pdf"]))
    if missing_predictions:
        raise ValueError(
            "Prediction CSV has no rows for: " + ", ".join(missing_predictions)
        )

    raw_root = output_root / "audit" / "raw_pages"
    review_root = output_root / "audit" / "reviews"
    extracted_root = output_root / "Extracted image"
    metrics_root = output_root / "metrics"
    for directory in (raw_root, review_root, extracted_root, metrics_root):
        directory.mkdir(parents=True, exist_ok=True)
    for pdf_name in pdf_names:
        (extracted_root / Path(pdf_name).stem).mkdir(parents=True, exist_ok=True)

    selected_predictions = predictions[predictions["pdf"].isin(pdf_names)].copy()
    selected_predictions = selected_predictions.sort_values(["pdf", "page"]).reset_index(drop=True)
    drawing_pages = selected_predictions[
        selected_predictions["predicted_page_type"] == "drawing"
    ].copy()
    source_paths: dict[tuple[str, int], Path] = {}
    page_counts: dict[str, int] = {}
    scale = render_dpi / 72.0
    for pdf_index, pdf_path in enumerate(pdf_paths, start=1):
        pdf_name = pdf_path.name
        pdf_predictions = selected_predictions[selected_predictions["pdf"] == pdf_name]
        with pymupdf.open(pdf_path) as document:
            page_counts[pdf_name] = int(document.page_count)
            expected_pages = set(range(1, document.page_count + 1))
            supplied_pages = set(pdf_predictions["page"].astype(int))
            if supplied_pages != expected_pages:
                missing = sorted(expected_pages - supplied_pages)
                extra = sorted(supplied_pages - expected_pages)
                raise ValueError(
                    f"Predictions do not cover {pdf_name} exactly; missing={missing}, extra={extra}"
                )
            selected_pages = pdf_predictions.loc[
                pdf_predictions["predicted_page_type"] == "drawing", "page"
            ].astype(int).tolist()
            if selected_pages:
                pdf_output = raw_root / pdf_path.stem
                pdf_output.mkdir(parents=True, exist_ok=True)
                for page_number in selected_pages:
                    pixmap = document.load_page(page_number - 1).get_pixmap(
                        matrix=pymupdf.Matrix(scale, scale),
                        colorspace=pymupdf.csGRAY,
                        alpha=False,
                    )
                    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
                    output_path = pdf_output / f"page-{page_number:04d}.jpg"
                    image.save(output_path, quality=92, optimize=True)
                    source_paths[(pdf_name, page_number)] = output_path
        if pdf_index % 50 == 0 or pdf_index == len(pdf_paths):
            progress(f"Rendered {pdf_index}/{len(pdf_paths)} PDFs")

    drawing_pages["source_path"] = [
        str(source_paths[(row.pdf, int(row.page))].resolve())
        for row in drawing_pages.itertuples(index=False)
    ]

    def source_page_path(pdf_name: str, page_number: int) -> Path:
        path = source_paths.get((pdf_name, int(page_number)))
        if path is None or not path.is_file():
            raise FileNotFoundError((pdf_name, page_number))
        return path

    def prepared_page(pdf_name: str, page_number: int) -> np.ndarray:
        return trim_outer_blank(load_gray(source_page_path(pdf_name, page_number)))[0]

    def safe_boundary_metrics(left: np.ndarray, right: np.ndarray) -> dict:
        try:
            _, metrics = join_boundary(left, right)
            return metrics
        except Exception as error:
            return {
                "correlation": np.nan,
                "margin": np.nan,
                "right_y_offset": np.nan,
                "error": str(error),
            }

    pair_records = []
    grouped_drawings = list(drawing_pages.groupby("pdf", sort=False))
    for pdf_index, (pdf_name, frame) in enumerate(grouped_drawings, start=1):
        page_numbers = sorted(frame["page"].astype(int).tolist())
        for left_page, right_page in zip(page_numbers, page_numbers[1:]):
            if right_page != left_page + 1:
                continue
            left, right = prepared_page(pdf_name, left_page), prepared_page(pdf_name, right_page)
            overlap = {
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "median_error_px": np.nan,
                "source_coverage": 0.0,
                "area_ratio": np.nan,
            }
            overlap_error = ""
            try:
                _, overlap_raw = register_overlap(left, right)
                homography = overlap_raw.pop("homography")
                overlap_raw.pop("crop_box", None)
                corners = np.float32(
                    [[[0, 0]], [[right.shape[1], 0]], [[right.shape[1], right.shape[0]]], [[0, right.shape[0]]]]
                )
                transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
                transformed_area = abs(float(cv2.contourArea(transformed.astype(np.float32))))
                overlap = {
                    **overlap,
                    **overlap_raw,
                    "area_ratio": transformed_area / max(float(right.size), 1.0),
                }
            except Exception as error:
                overlap_error = str(error)
            forward, reverse = safe_boundary_metrics(left, right), safe_boundary_metrics(right, left)
            forward_score = np.nan_to_num(float(forward.get("correlation", np.nan)), nan=-999.0)
            reverse_score = np.nan_to_num(float(reverse.get("correlation", np.nan)), nan=-999.0)
            if forward_score >= reverse_score:
                boundary, boundary_order = forward, "pdf_order"
            else:
                boundary, boundary_order = reverse, "reverse_order"
            strong_overlap = (
                overlap["inliers"] >= 24
                and overlap["inlier_ratio"] >= 0.38
                and overlap["median_error_px"] <= 2.5
                and overlap["source_coverage"] >= 0.012
                and 0.78 <= overlap["area_ratio"] <= 1.25
            )
            moderate_overlap = (
                overlap["inliers"] >= 12
                and overlap["inlier_ratio"] >= 0.12
                and overlap["median_error_px"] <= 5.0
                and overlap["source_coverage"] >= 0.004
                and 0.55 <= overlap["area_ratio"] <= 1.55
            )
            boundary_candidate = (
                np.isfinite(boundary.get("correlation", np.nan))
                and boundary["correlation"] >= 0.64
                and boundary.get("margin", 0.0) >= 0.08
            )
            base_decision = (
                "AUTO_OVERLAP" if strong_overlap
                else "BOUNDARY_CANDIDATE" if moderate_overlap or boundary_candidate
                else "BREAK_NO_RELATION"
            )
            pair_records.append({
                "pdf": pdf_name,
                "left_page": int(left_page),
                "right_page": int(right_page),
                "base_decision": base_decision,
                "overlap_good_matches": int(overlap["good_matches"]),
                "overlap_inliers": int(overlap["inliers"]),
                "overlap_inlier_ratio": float(overlap["inlier_ratio"]),
                "overlap_median_error_px": float(overlap["median_error_px"]),
                "overlap_source_coverage": float(overlap["source_coverage"]),
                "overlap_area_ratio": float(overlap["area_ratio"]),
                "boundary_order": boundary_order,
                "boundary_correlation": float(boundary.get("correlation", np.nan)),
                "boundary_margin": float(boundary.get("margin", np.nan)),
                "overlap_error": overlap_error,
            })
        if pdf_index % 50 == 0 or pdf_index == len(grouped_drawings):
            progress(f"Scored {pdf_index}/{len(grouped_drawings)} drawing PDFs")

    pair_metrics = pd.DataFrame(pair_records)
    if pair_metrics.empty:
        pair_metrics = _empty_frame(PAIR_COLUMNS[:-2])
    pair_metrics["supported_overlap_seed"] = False
    for pair_index, pair in pair_metrics.iterrows():
        alternate_overlap = (
            pair["base_decision"] != "AUTO_OVERLAP"
            and pair["overlap_inliers"] >= 60
            and pair["overlap_inlier_ratio"] >= 0.18
            and pair["overlap_median_error_px"] <= 1.0
            and pair["overlap_source_coverage"] >= 0.02
            and 0.80 <= pair["overlap_area_ratio"] <= 1.20
        )
        previous = pair_metrics[
            (pair_metrics["pdf"] == pair["pdf"])
            & (pair_metrics["right_page"] == pair["left_page"])
        ]
        preceding_support = False
        if len(previous) == 1:
            previous = previous.iloc[0]
            preceding_support = (
                previous["boundary_correlation"] >= 0.95
                and previous["boundary_margin"] < 0.06
                and previous["overlap_inliers"] >= 20
                and previous["overlap_median_error_px"] <= 2.0
                and previous["overlap_source_coverage"] >= 0.10
            )
        if alternate_overlap and preceding_support:
            pair_metrics.loc[pair_index, "supported_overlap_seed"] = True
            pair_metrics.loc[pair_index, "base_decision"] = "AUTO_OVERLAP_SUPPORTED_SEED"
    pair_metrics["supported_component_bridge"] = (
        (pair_metrics["boundary_correlation"] >= 0.95)
        & (pair_metrics["boundary_margin"] < 0.06)
        & (pair_metrics["overlap_inliers"] >= 20)
        & (pair_metrics["overlap_median_error_px"] <= 2.0)
        & (pair_metrics["overlap_source_coverage"] >= 0.10)
    )
    pair_metrics = pair_metrics.reindex(columns=PAIR_COLUMNS)
    pair_metrics.to_csv(metrics_root / "adjacent_pair_metrics.csv", index=False)

    def assemble_overlap_segment(pdf_name: str, pages: list[int]) -> tuple[np.ndarray, list[dict]]:
        canvas, diagnostics = prepared_page(pdf_name, pages[0]), []
        for page_number in pages[1:]:
            canvas, metrics = register_overlap(canvas, prepared_page(pdf_name, page_number))
            diagnostics.append({
                "page": int(page_number),
                "inliers": int(metrics["inliers"]),
                "inlier_ratio": float(metrics["inlier_ratio"]),
                "median_error_px": float(metrics["median_error_px"]),
            })
        return canvas, diagnostics

    states: dict[int, dict] = {}
    page_to_state: dict[tuple[str, int], int] = {}
    state_counter = 0
    overlap_failures = []
    for pdf_name, frame in drawing_pages.groupby("pdf", sort=False):
        pages = sorted(frame["page"].astype(int).tolist())
        union = UnionFind(pages)
        pdf_pairs = pair_metrics[pair_metrics["pdf"] == pdf_name]
        for pair in pdf_pairs.itertuples(index=False):
            if pair.base_decision in {"AUTO_OVERLAP", "AUTO_OVERLAP_SUPPORTED_SEED"}:
                union.union(int(pair.left_page), int(pair.right_page))
        components: dict[int, list[int]] = {}
        for page_number in pages:
            components.setdefault(union.find(page_number), []).append(page_number)
        for component_pages in components.values():
            component_pages = sorted(component_pages)
            try:
                if len(component_pages) > 1:
                    canvas, diagnostics = assemble_overlap_segment(pdf_name, component_pages)
                    method = "OVERLAP_COMPONENT"
                else:
                    canvas, diagnostics = prepared_page(pdf_name, component_pages[0]), []
                    method = "SINGLE_PAGE"
                state_counter += 1
                states[state_counter] = {
                    "pdf": pdf_name,
                    "pages": component_pages,
                    "canvas": canvas,
                    "methods": [method],
                    "diagnostics": diagnostics,
                }
                for page_number in component_pages:
                    page_to_state[(pdf_name, page_number)] = state_counter
            except Exception as error:
                overlap_failures.append({
                    "pdf": pdf_name,
                    "pages": "|".join(map(str, component_pages)),
                    "error": str(error),
                })
                for page_number in component_pages:
                    state_counter += 1
                    states[state_counter] = {
                        "pdf": pdf_name,
                        "pages": [page_number],
                        "canvas": prepared_page(pdf_name, page_number),
                        "methods": ["OVERLAP_ASSEMBLY_FALLBACK"],
                        "diagnostics": [{"error": str(error)}],
                    }
                    page_to_state[(pdf_name, page_number)] = state_counter

    boundary_actions, unresolved_edges = [], []
    candidate_pairs = pair_metrics[
        (pair_metrics["base_decision"] == "BOUNDARY_CANDIDATE")
        | pair_metrics["supported_component_bridge"].astype(bool)
    ].sort_values(["pdf", "left_page", "right_page"])

    for pair in candidate_pairs.itertuples(index=False):
        left_key, right_key = (pair.pdf, int(pair.left_page)), (pair.pdf, int(pair.right_page))
        left_id, right_id = page_to_state[left_key], page_to_state[right_key]
        if left_id == right_id:
            continue
        left_state, right_state = states[left_id], states[right_id]
        left_size, right_size = len(left_state["pages"]), len(right_state["pages"])
        singleton_component = min(left_size, right_size) == 1 and max(left_size, right_size) > 1
        two_overlap_components = left_size > 1 and right_size > 1
        high_boundary = pair.boundary_correlation >= 0.75 and pair.boundary_margin >= 0.18
        low_singleton_bridge = singleton_component and pair.boundary_correlation >= 0.64 and pair.boundary_margin >= 0.08
        supported_ambiguous_bridge = singleton_component and bool(pair.supported_component_bridge)
        eligible = high_boundary or low_singleton_bridge or supported_ambiguous_bridge
        if two_overlap_components and not high_boundary:
            eligible = False
        if left_size == 1 and right_size == 1:
            eligible = False
        action = {
            "pass": np.nan,
            "pdf": pair.pdf,
            "left_page": int(pair.left_page),
            "right_page": int(pair.right_page),
            "left_component_size": left_size,
            "right_component_size": right_size,
            "boundary_correlation": float(pair.boundary_correlation),
            "boundary_margin": float(pair.boundary_margin),
            "boundary_order": pair.boundary_order,
            "supported_component_bridge": bool(pair.supported_component_bridge),
            "eligible": bool(eligible),
            "decision": "HOLD_TOPOLOGY",
            "direction": "",
            "local_inliers": 0,
            "translation_residual_px": np.nan,
            "matched_ink_ratio": 0.0,
            "matched_ink_pixels": 0,
            "validation_tier": "NONE",
            "error": "",
        }
        if not eligible:
            unresolved_edges.append((left_key, right_key))
            boundary_actions.append(action)
            continue
        use_pdf_order = supported_ambiguous_bridge or pair.boundary_order == "pdf_order"
        first_id, second_id = (left_id, right_id) if use_pdf_order else (right_id, left_id)
        first_state, second_state = states[first_id], states[second_id]
        try:
            _, coarse_metrics = join_boundary(first_state["canvas"], second_state["canvas"])
            first_piece, second_piece, local = local_translation_consensus(
                first_state["canvas"], second_state["canvas"],
                coarse_y=int(coarse_metrics["right_y_offset"]),
            )
            duplicate = seam_duplicate_score(first_piece, second_piece, local["right_x"], local["right_y"])
            optimized, _ = compose_with_seam_cut(first_piece, second_piece, local["right_x"], local["right_y"])
            strict_precise = local["inliers"] >= 8 and local["median_residual_px"] <= 2.0 and duplicate["matched_ink_pixels"] >= 500 and duplicate["matched_ink_ratio"] >= 0.12
            guarded_exact_sparse = local["inliers"] >= 3 and local["median_residual_px"] <= 0.5 and duplicate["matched_ink_pixels"] >= 2000 and duplicate["matched_ink_ratio"] >= 0.20 and pair.boundary_correlation >= 0.70 and pair.boundary_margin >= 0.30
            guarded_sparse = local["inliers"] >= 6 and local["median_residual_px"] <= 6.0 and duplicate["matched_ink_pixels"] >= 120 and duplicate["matched_ink_ratio"] >= 0.08 and pair.boundary_correlation >= 0.75 and pair.boundary_margin >= 0.15
            guarded_dense = local["inliers"] >= 20 and local["median_residual_px"] <= 12.0 and duplicate["matched_ink_pixels"] >= 1000 and duplicate["matched_ink_ratio"] >= 0.50 and pair.boundary_correlation >= 0.85 and pair.boundary_margin >= 0.25
            supported_component_dense = bool(pair.supported_component_bridge) and local["inliers"] >= 40 and local["median_residual_px"] <= 8.0 and duplicate["matched_ink_pixels"] >= 5000 and duplicate["matched_ink_ratio"] >= 0.40
            accepted = strict_precise or guarded_exact_sparse or guarded_sparse or guarded_dense or supported_component_dense
            validation_tier = (
                "STRICT_PRECISE" if strict_precise
                else "GUARDED_EXACT_SPARSE" if guarded_exact_sparse
                else "GUARDED_SPARSE" if guarded_sparse
                else "GUARDED_DENSE" if guarded_dense
                else "SUPPORTED_COMPONENT_DENSE" if supported_component_dense
                else "NONE"
            )
            action.update({
                "local_inliers": int(local["inliers"]),
                "translation_residual_px": float(local["median_residual_px"]),
                "matched_ink_ratio": float(duplicate["matched_ink_ratio"]),
                "matched_ink_pixels": int(duplicate["matched_ink_pixels"]),
                "validation_tier": validation_tier,
            })
            if not accepted:
                action["decision"] = "HOLD_LOCAL_VALIDATION"
                unresolved_edges.append((left_key, right_key))
                boundary_actions.append(action)
                continue
            merged_pages = sorted(set(first_state["pages"] + second_state["pages"]))
            state_counter += 1
            states[state_counter] = {
                "pdf": pair.pdf,
                "pages": merged_pages,
                "canvas": optimized,
                "methods": first_state["methods"] + ["OPTIMIZED_LOCAL_SEAM"] + second_state["methods"],
                "diagnostics": first_state["diagnostics"] + [action.copy()] + second_state["diagnostics"],
            }
            del states[first_id]
            del states[second_id]
            for page_number in merged_pages:
                page_to_state[(pair.pdf, page_number)] = state_counter
            action["decision"] = "AUTO_LOCAL_SEAM"
        except Exception as error:
            action["decision"] = "HOLD_OPTIMIZER_ERROR"
            action["error"] = str(error)
            unresolved_edges.append((left_key, right_key))
        boundary_actions.append(action)

    def bridge_tier(pair, left_size: int, right_size: int, local: dict, duplicate: dict) -> str:
        singleton_to_group = min(left_size, right_size) == 1 and max(left_size, right_size) > 1
        if left_size > 1 and right_size > 1:
            return "NONE"
        inliers = int(local["inliers"])
        residual = float(local["median_residual_px"])
        duplicate_pixels = int(duplicate["matched_ink_pixels"])
        duplicate_ratio = float(duplicate["matched_ink_ratio"])
        overlap_px = int(local["overlap_px"])
        robust_dense = inliers >= 35 and residual <= 8.0 and duplicate_pixels >= 3000 and duplicate_ratio >= 0.50 and overlap_px >= 80
        overlap_supported = pair.overlap_inliers >= 50 and pair.overlap_inlier_ratio >= 0.25 and pair.overlap_median_error_px <= 1.25 and pair.overlap_source_coverage >= 0.04 and 0.78 <= pair.overlap_area_ratio <= 1.25 and inliers >= 12 and residual <= 22.0 and duplicate_pixels >= 2000 and duplicate_ratio >= 0.20
        high_boundary_tolerant = pair.boundary_correlation >= 0.85 and pair.boundary_margin >= 0.30 and inliers >= 8 and residual <= 20.0 and duplicate_pixels >= 2000 and duplicate_ratio >= 0.30
        precise_narrow = pair.boundary_correlation >= 0.70 and pair.boundary_margin >= 0.35 and inliers >= 8 and residual <= 1.5 and duplicate_pixels >= 200 and duplicate_ratio >= 0.85
        chain_supported = singleton_to_group and pair.boundary_correlation >= 0.65 and inliers >= 4 and residual <= 25.0 and duplicate_pixels >= 4000 and duplicate_ratio >= 0.15
        if robust_dense:
            return "ROBUST_DENSE"
        if overlap_supported:
            return "OVERLAP_SUPPORTED"
        if high_boundary_tolerant:
            return "HIGH_BOUNDARY_TOLERANT"
        if precise_narrow:
            return "PRECISE_NARROW"
        if chain_supported:
            return "CHAIN_SUPPORTED"
        return "NONE"

    def evaluate_bridge_direction(pair, first_state: dict, second_state: dict) -> dict:
        _, coarse_metrics = join_boundary(first_state["canvas"], second_state["canvas"])
        first_piece, second_piece, local = local_translation_consensus(
            first_state["canvas"], second_state["canvas"],
            coarse_y=int(coarse_metrics["right_y_offset"]),
        )
        duplicate = seam_duplicate_score(first_piece, second_piece, local["right_x"], local["right_y"])
        optimized, compose_metrics = compose_with_seam_cut(first_piece, second_piece, local["right_x"], local["right_y"])
        tier = bridge_tier(pair, len(first_state["pages"]), len(second_state["pages"]), local, duplicate)
        quality = int(local["inliers"]) + 120.0 * float(duplicate["matched_ink_ratio"]) + 8.0 * math.log1p(int(duplicate["matched_ink_pixels"])) + 25.0 * float(coarse_metrics["correlation"]) - 2.0 * float(local["median_residual_px"])
        return {
            "canvas": optimized,
            "tier": tier,
            "quality": float(quality),
            "coarse": coarse_metrics,
            "local": local,
            "duplicate": duplicate,
            "compose": compose_metrics,
        }

    v5d_candidate_pairs = pair_metrics[
        ~pair_metrics["base_decision"].isin(["AUTO_OVERLAP", "AUTO_OVERLAP_SUPPORTED_SEED"])
    ].sort_values(["pdf", "left_page", "right_page"])
    tier_priority = {
        "ROBUST_DENSE": 5,
        "OVERLAP_SUPPORTED": 4,
        "HIGH_BOUNDARY_TOLERANT": 3,
        "PRECISE_NARROW": 2,
        "CHAIN_SUPPORTED": 1,
    }
    attempted_signatures = set()
    for pass_index in range(1, 6):
        merge_proposals = []
        for pair in v5d_candidate_pairs.itertuples(index=False):
            left_key, right_key = (pair.pdf, int(pair.left_page)), (pair.pdf, int(pair.right_page))
            left_id, right_id = page_to_state[left_key], page_to_state[right_key]
            if left_id == right_id:
                continue
            left_state, right_state = states[left_id], states[right_id]
            left_size, right_size = len(left_state["pages"]), len(right_state["pages"])
            singleton_to_group = min(left_size, right_size) == 1 and max(left_size, right_size) > 1
            ordinary_candidate = pair.base_decision == "BOUNDARY_CANDIDATE" or bool(pair.supported_component_bridge)
            latent_chain_candidate = singleton_to_group and (pair.boundary_correlation >= 0.40 or pair.overlap_inliers >= 8)
            eligible = (ordinary_candidate or latent_chain_candidate) and not (left_size > 1 and right_size > 1)
            signature = (pair.pdf, int(pair.left_page), int(pair.right_page), left_id, right_id)
            if signature in attempted_signatures:
                continue
            attempted_signatures.add(signature)
            if not eligible:
                continue
            proposals = []
            for direction, first_id, second_id in (
                ("pdf_order", left_id, right_id),
                ("reverse_order", right_id, left_id),
            ):
                try:
                    proposal = evaluate_bridge_direction(pair, states[first_id], states[second_id])
                    proposal.update({"direction": direction, "first_id": first_id, "second_id": second_id})
                    rotated_scaled_duplicate = pair.overlap_inlier_ratio >= 0.60 and pair.overlap_source_coverage >= 0.30 and pair.overlap_area_ratio < 0.65
                    direction_conflict_partial = direction != pair.boundary_order and pair.overlap_inlier_ratio >= 0.20 and pair.overlap_source_coverage < 0.05
                    oversized_homography = np.isfinite(pair.overlap_area_ratio) and pair.overlap_area_ratio > 2.0
                    if rotated_scaled_duplicate or direction_conflict_partial or oversized_homography:
                        proposal["tier"] = "NONE"
                    proposals.append(proposal)
                except Exception:
                    continue
            accepted = [proposal for proposal in proposals if proposal["tier"] != "NONE"]
            if not accepted:
                continue
            best = max(accepted, key=lambda item: (tier_priority[item["tier"]], item["quality"]))
            action = {
                "pass": pass_index,
                "pdf": pair.pdf,
                "left_page": int(pair.left_page),
                "right_page": int(pair.right_page),
                "left_component_size": left_size,
                "right_component_size": right_size,
                "boundary_correlation": float(pair.boundary_correlation),
                "boundary_margin": float(pair.boundary_margin),
                "boundary_order": pair.boundary_order,
                "supported_component_bridge": bool(pair.supported_component_bridge),
                "eligible": True,
                "decision": "AUTO_ITERATIVE_BRIDGE",
                "direction": best["direction"],
                "validation_tier": best["tier"],
                "local_inliers": int(best["local"]["inliers"]),
                "translation_residual_px": float(best["local"]["median_residual_px"]),
                "matched_ink_ratio": float(best["duplicate"]["matched_ink_ratio"]),
                "matched_ink_pixels": int(best["duplicate"]["matched_ink_pixels"]),
                "error": "",
            }
            merge_proposals.append({
                "priority": tier_priority[best["tier"]],
                "quality": best["quality"],
                "pair": pair,
                "left_key": left_key,
                "right_key": right_key,
                "left_id": left_id,
                "right_id": right_id,
                "best": best,
                "action": action,
            })
        changed, used_state_ids = False, set()
        merge_proposals.sort(key=lambda item: (item["priority"], item["quality"]), reverse=True)
        for proposal in merge_proposals:
            left_id, right_id = proposal["left_id"], proposal["right_id"]
            if left_id in used_state_ids or right_id in used_state_ids:
                continue
            if page_to_state[proposal["left_key"]] != left_id or page_to_state[proposal["right_key"]] != right_id:
                continue
            best = proposal["best"]
            first_state, second_state = states[best["first_id"]], states[best["second_id"]]
            merged_pages = sorted(set(first_state["pages"] + second_state["pages"]))
            state_counter += 1
            states[state_counter] = {
                "pdf": proposal["pair"].pdf,
                "pages": merged_pages,
                "canvas": best["canvas"],
                "methods": first_state["methods"] + [f"ITERATIVE_{best['tier']}"] + second_state["methods"],
                "diagnostics": first_state["diagnostics"] + [proposal["action"].copy()] + second_state["diagnostics"],
            }
            del states[best["first_id"]]
            del states[best["second_id"]]
            for page_number in merged_pages:
                page_to_state[(proposal["pair"].pdf, page_number)] = state_counter
            boundary_actions.append(proposal["action"])
            used_state_ids.update([left_id, right_id])
            changed = True
        if not changed:
            break

    unresolved_edges = []
    for pair in pair_metrics.itertuples(index=False):
        if pair.base_decision != "BOUNDARY_CANDIDATE" and not bool(pair.supported_component_bridge):
            continue
        left_key, right_key = (pair.pdf, int(pair.left_page)), (pair.pdf, int(pair.right_page))
        if page_to_state[left_key] != page_to_state[right_key]:
            unresolved_edges.append((left_key, right_key))

    boundary_metrics = pd.DataFrame(boundary_actions)
    if boundary_metrics.empty:
        boundary_metrics = _empty_frame(BOUNDARY_COLUMNS)
    boundary_metrics = boundary_metrics.reindex(columns=BOUNDARY_COLUMNS)
    boundary_metrics.to_csv(metrics_root / "boundary_actions.csv", index=False)
    pd.DataFrame(overlap_failures, columns=["pdf", "pages", "error"]).to_csv(
        metrics_root / "overlap_assembly_failures.csv", index=False
    )

    hold_state_ids = set()
    for left_key, right_key in unresolved_edges:
        left_id, right_id = page_to_state[left_key], page_to_state[right_key]
        if left_id != right_id:
            hold_state_ids.update([left_id, right_id])
    output_records = []
    for state_id, state in sorted(states.items(), key=lambda item: (item[1]["pdf"], min(item[1]["pages"]))):
        pdf_name, pages = state["pdf"], sorted(state["pages"])
        pdf_output = extracted_root / Path(pdf_name).stem
        is_hold = state_id in hold_state_ids
        page_token = "-".join(f"{page:02d}" for page in pages)
        if len(pages) > 1:
            prefix = "HOLD__MERGED" if is_hold else "MERGED"
            output_path = pdf_output / f"{prefix}__pages_{page_token}.png"
            Image.fromarray(state["canvas"]).save(output_path, optimize=True)
            output_type = "HOLD_MERGED" if is_hold else "MERGED"
        else:
            prefix = "HOLD" if is_hold else "SINGLE"
            output_path = pdf_output / f"{prefix}__page_{pages[0]:02d}.jpg"
            shutil.copy2(source_page_path(pdf_name, pages[0]), output_path)
            output_type = "HOLD_SINGLE" if is_hold else "SINGLE"
        output_records.append({
            "pdf": pdf_name,
            "pages": "|".join(map(str, pages)),
            "page_count": len(pages),
            "output_type": output_type,
            "methods": "|".join(state["methods"]),
            "output_path": str(output_path.resolve()),
        })
    output_manifest = pd.DataFrame(
        output_records,
        columns=["pdf", "pages", "page_count", "output_type", "methods", "output_path"],
    )
    output_manifest.to_csv(metrics_root / "output_manifest.csv", index=False)

    pdf_rows = []
    for pdf_name in pdf_names:
        selected = output_manifest[output_manifest["pdf"] == pdf_name]
        pdf_rows.append({
            "pdf": pdf_name,
            "source_page_count": page_counts[pdf_name],
            "drawing_page_count": int((drawing_pages["pdf"] == pdf_name).sum()),
            "delivered_image_count": int(len(selected)),
            "merged_image_count": int(selected["output_type"].isin(["MERGED", "HOLD_MERGED"]).sum()),
            "hold_image_count": int(selected["output_type"].astype(str).str.startswith("HOLD").sum()),
        })
    pdf_summary = pd.DataFrame(pdf_rows)
    pdf_summary.to_csv(metrics_root / "pdf_summary.csv", index=False)
    review_sheet_count = _write_review_sheets(
        output_manifest, boundary_metrics, review_root
    ) if write_review_sheets else 0

    delivered_members = []
    for row in output_manifest.itertuples(index=False):
        for page_number in str(row.pages).split("|"):
            delivered_members.append((row.pdf, int(page_number)))
    expected_members = sorted(
        (row.pdf, int(row.page)) for row in drawing_pages.itertuples(index=False)
    )
    if sorted(delivered_members) != expected_members:
        raise AssertionError("Delivered page membership does not match drawing predictions")
    if len(delivered_members) != len(set(delivered_members)):
        raise AssertionError("A drawing page was delivered more than once")
    if set(path.name for path in extracted_root.iterdir()) != {Path(name).stem for name in pdf_names}:
        raise AssertionError("Delivery folder names do not match input PDFs")
    if list(output_root.rglob("*.txt")):
        raise AssertionError("The pipeline must not generate TXT sidecars")

    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "pdfs": int(len(pdf_paths)),
        "source_pages": int(len(selected_predictions)),
        "predicted_drawing_pages": int(len(drawing_pages)),
        "pdfs_with_drawing_pages": int(drawing_pages["pdf"].nunique()),
        "empty_pdf_folders": int((pdf_summary["delivered_image_count"] == 0).sum()),
        "delivery_images": int(len(output_manifest)),
        "merged_images": int(output_manifest["output_type"].isin(["MERGED", "HOLD_MERGED"]).sum()),
        "hold_images": int(output_manifest["output_type"].astype(str).str.startswith("HOLD").sum()),
        "hold_pdfs": int(output_manifest.loc[output_manifest["output_type"].astype(str).str.startswith("HOLD"), "pdf"].nunique()),
        "automatic_local_seams": int((boundary_metrics["decision"] == "AUTO_LOCAL_SEAM").sum()),
        "automatic_iterative_bridges": int((boundary_metrics["decision"] == "AUTO_ITERATIVE_BRIDGE").sum()),
        "overlap_assembly_failures": int(len(overlap_failures)),
        "review_sheets": int(review_sheet_count),
        "render_dpi": int(render_dpi),
        "elapsed_seconds": round(time.time() - started, 3),
        "all_drawing_pages_delivered_exactly_once": True,
        "figure_level_cropping_used": False,
        "manual_masks_used": False,
        "pdf_specific_geometry_used": False,
        "txt_generated": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def command_extract(args: argparse.Namespace) -> None:
    pdf_paths = find_pdf_inputs(args.input, recursive=args.recursive)
    if args.max_pdfs is not None:
        if args.max_pdfs < 1:
            raise ValueError("--max-pdfs must be positive")
        pdf_paths = pdf_paths[:args.max_pdfs]
    predictions = load_prediction_table(args.predictions)
    extract_drawings(
        pdf_paths=pdf_paths,
        predictions=predictions,
        output_root=args.output,
        render_dpi=args.render_dpi,
        write_review_sheets=not args.no_review_sheets,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract complete patent drawing pages and assemble confident cross-page plates."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract", help="Run the page-first extraction pipeline.")
    extract_parser.add_argument("--input", type=Path, required=True, help="PDF file or PDF directory.")
    extract_parser.add_argument("--predictions", type=Path, required=True, help="CSV produced by page_classifier.py predict.")
    extract_parser.add_argument("--output", type=Path, required=True, help="A new output directory; it must not already exist.")
    extract_parser.add_argument("--recursive", action="store_true", help="Search PDF subdirectories recursively.")
    extract_parser.add_argument("--max-pdfs", type=int, help="Process only the first N PDFs (smoke tests).")
    extract_parser.add_argument("--render-dpi", type=int, default=DEFAULT_RENDER_DPI)
    extract_parser.add_argument("--no-review-sheets", action="store_true", help="Skip audit/review contact sheets.")
    extract_parser.set_defaults(func=command_extract)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
