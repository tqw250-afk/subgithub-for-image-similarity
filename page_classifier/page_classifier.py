"""Frozen V2 patent page classifier.

This module keeps training and inference separate:

1. ``train`` reads the already prepared Stage 10 feature table, fits the
   Extra Trees model once, and saves a versioned joblib bundle.
2. ``predict`` loads that bundle and classifies a PDF or a directory of PDFs
   without fitting the model again.

Required packages:
    numpy pandas pillow opencv-python pymupdf scikit-learn joblib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import pymupdf
import sklearn
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier


MODEL_VERSION = "page-classifier-v2"
RENDER_DPI = 90
WORK_WIDTH = 600
INK_THRESHOLD = 225
DECISION_THRESHOLD = 0.5

VISUAL_FEATURE_COLUMNS = [
    "ink_ratio",
    "component_count_per_mp",
    "small_component_fraction",
    "large_component_ink_fraction",
    "wide_component_fraction",
    "tall_component_fraction",
    "long_line_ink_fraction",
    "row_density_std",
    "column_density_std",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_page_features(
    page_path: Path,
    work_width: int = WORK_WIDTH,
    ink_threshold: int = INK_THRESHOLD,
) -> dict[str, float]:
    """Extract the nine visual features used by the frozen V2 model."""
    with Image.open(page_path) as opened:
        gray = np.asarray(opened.convert("L"))

    scale = work_width / gray.shape[1]
    resized = cv2.resize(
        gray,
        (work_width, max(1, int(round(gray.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    ink = (resized < ink_threshold).astype(np.uint8)
    ink_pixels = int(ink.sum())
    page_area = int(ink.size)

    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
        widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
        heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    else:
        areas = np.array([], dtype=float)
        widths = np.array([], dtype=float)
        heights = np.array([], dtype=float)

    valid = areas >= 2
    areas, widths, heights = areas[valid], widths[valid], heights[valid]

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(12, work_width // 30), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(12, resized.shape[0] // 40))
    )
    horizontal_lines = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, horizontal_kernel
    )
    vertical_lines = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, vertical_kernel
    )
    line_pixels = int(np.logical_or(horizontal_lines, vertical_lines).sum())

    row_density = ink.mean(axis=1)
    column_density = ink.mean(axis=0)
    return {
        "ink_ratio": ink_pixels / page_area,
        "component_count_per_mp": len(areas) / (page_area / 1_000_000),
        "small_component_fraction": (
            float(np.mean(areas <= 80)) if len(areas) else 0.0
        ),
        "large_component_ink_fraction": float(
            areas[areas >= 400].sum() / max(ink_pixels, 1)
        ),
        "wide_component_fraction": (
            float(np.mean(widths >= work_width * 0.08)) if len(widths) else 0.0
        ),
        "tall_component_fraction": (
            float(np.mean(heights >= resized.shape[0] * 0.08))
            if len(heights)
            else 0.0
        ),
        "long_line_ink_fraction": line_pixels / max(ink_pixels, 1),
        "row_density_std": float(row_density.std()),
        "column_density_std": float(column_density.std()),
    }


def render_pdf_pages(
    pdf_path: Path, output_dir: Path, dpi: int = RENDER_DPI
) -> list[Path]:
    """Render all PDF pages to temporary JPEG files in page order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    output_paths: list[Path] = []
    with pymupdf.open(pdf_path) as document:
        for page_index in range(document.page_count):
            output_path = output_dir / f"page-{page_index + 1:04d}.jpg"
            pixmap = document.load_page(page_index).get_pixmap(
                matrix=matrix, alpha=False
            )
            pixmap.save(str(output_path))
            output_paths.append(output_path)
    return output_paths


def build_model_bundle(training_csv: Path) -> dict[str, Any]:
    """Fit V2 once from the saved Stage 10 feature table."""
    training_frame = pd.read_csv(training_csv)
    required = {"manual_label", *VISUAL_FEATURE_COLUMNS}
    missing = required - set(training_frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")

    x_train = training_frame[VISUAL_FEATURE_COLUMNS].to_numpy()
    y_train = (training_frame["manual_label"] == "drawing").astype(int).to_numpy()
    model = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    return {
        "model": model,
        "model_version": MODEL_VERSION,
        "feature_columns": VISUAL_FEATURE_COLUMNS,
        "class_mapping": {0: "text", 1: "drawing"},
        "decision_threshold": DECISION_THRESHOLD,
        "preprocessing": {
            "render_dpi": RENDER_DPI,
            "work_width": WORK_WIDTH,
            "ink_threshold": INK_THRESHOLD,
        },
        "training": {
            "csv_name": training_csv.name,
            "csv_sha256": sha256_file(training_csv),
            "rows": int(len(training_frame)),
            "pdfs": (
                int(training_frame["pdf"].nunique())
                if "pdf" in training_frame.columns
                else None
            ),
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "pymupdf": pymupdf.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def save_model_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)
    metadata = {key: value for key, value in bundle.items() if key != "model"}
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load a trusted, locally produced joblib model bundle."""
    bundle = joblib.load(model_path)
    required = {
        "model",
        "feature_columns",
        "decision_threshold",
        "preprocessing",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    if list(bundle["feature_columns"]) != VISUAL_FEATURE_COLUMNS:
        raise ValueError("Model feature order does not match this runtime.")
    return bundle


def drawing_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features.to_numpy())
    drawing_indices = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(drawing_indices) != 1:
        raise ValueError("The model does not contain the expected drawing class 1.")
    return probabilities[:, int(drawing_indices[0])]


def classify_pdf(pdf_path: Path, bundle: dict[str, Any]) -> pd.DataFrame:
    """Classify every page of one PDF without retraining."""
    preprocessing = bundle["preprocessing"]
    with TemporaryDirectory(prefix="page_classifier_") as temporary_dir:
        page_paths = render_pdf_pages(
            pdf_path,
            Path(temporary_dir),
            dpi=int(preprocessing["render_dpi"]),
        )
        rows = []
        for page_number, page_path in enumerate(page_paths, start=1):
            rows.append(
                {
                    "pdf": pdf_path.name,
                    "pdf_path": str(pdf_path.resolve()),
                    "page": page_number,
                    "page_count": len(page_paths),
                    **extract_page_features(
                        page_path,
                        work_width=int(preprocessing["work_width"]),
                        ink_threshold=int(preprocessing["ink_threshold"]),
                    ),
                }
            )

    result = pd.DataFrame(rows)
    probability = drawing_probability(
        bundle["model"], result[bundle["feature_columns"]]
    )
    threshold = float(bundle["decision_threshold"])
    result["drawing_probability"] = probability
    result["predicted_page_type"] = np.where(
        probability >= threshold, "drawing", "text"
    )
    result["confidence"] = np.maximum(probability, 1.0 - probability)
    return result


def find_pdf_inputs(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*.pdf") if recursive else input_path.glob("*.pdf")
    pdf_paths = sorted(path for path in iterator if path.is_file())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under: {input_path}")
    return pdf_paths


def command_train(args: argparse.Namespace) -> None:
    training_csv = args.training_csv.resolve()
    model_output = args.model_output.resolve()
    bundle = build_model_bundle(training_csv)
    save_model_bundle(bundle, model_output)
    print(f"Saved model: {model_output}")
    print(f"Saved metadata: {model_output.with_suffix('.metadata.json')}")


def command_predict(args: argparse.Namespace) -> None:
    bundle = load_model_bundle(args.model.resolve())
    pdf_paths = find_pdf_inputs(args.input.resolve(), recursive=args.recursive)
    frames = []
    for index, pdf_path in enumerate(pdf_paths, start=1):
        frames.append(classify_pdf(pdf_path, bundle))
        print(f"Classified {index}/{len(pdf_paths)}: {pdf_path.name}")
    predictions = pd.concat(frames, ignore_index=True)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, save, load, and run the frozen V2 page classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Fit V2 once from the saved feature table and save it."
    )
    train_parser.add_argument("--training-csv", type=Path, required=True)
    train_parser.add_argument("--model-output", type=Path, required=True)
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser(
        "predict", help="Load a saved model and classify PDF pages."
    )
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search PDF subdirectories recursively when input is a directory.",
    )
    predict_parser.set_defaults(func=command_predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
