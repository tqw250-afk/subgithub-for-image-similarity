# Swedish Historical Patent Page Classifier

This package contains a lightweight page classifier developed from scanned Swedish patent documents dated **1895–1900**. It is designed to quickly label every page of a patent PDF as one of two types:

- `text` — pages dominated by printed or handwritten textual material;
- `drawing` — pages dominated by patent drawings, diagrams, or technical figures.

The classifier is intended as a preprocessing step for historical-patent digitisation workflows. It can separate drawing pages from descriptive text before downstream OCR, figure extraction, image matching, or stitching. It does **not** perform OCR, identify individual figures, or interpret the technical content of a patent.

## Package contents

| File | Purpose |
|---|---|
| `page_classifier.py` | Command-line training and prediction program. |
| `page_classifier_benchmark.ipynb` | Ready-to-run benchmark using the files in this folder. |
| `page_classifier_benchmark_template.ipynb` | Portable benchmark whose paths are marked with English `# TODO` comments and `xxx` placeholders. |
| `enriched_training_cross_validation.csv` | The labelled feature table used to reconstruct the final V2 classifier. |
| `requirements.txt` | Required Python packages. |
| `input_pdfs/` | Default location for PDFs used by the ready-to-run benchmark. |

No prebuilt `.joblib` model is included. The labelled feature table is included so users can inspect the training material and reconstruct the model locally.

## Training data

The final V2 training table contains **181 manually reviewed pages from 89 PDFs**. Each row records:

- the source PDF and page number;
- the manual label (`text` or `drawing`);
- nine visual page features;
- grouped cross-validation diagnostics retained from model development.

The training table contains numerical visual features and labels, not the original patent page images.

The development process first used a stratified sample of patent documents, then applied an initial classifier across the corpus. Low-confidence pages and abnormal page sequences were manually reviewed and added to the training data. Evaluation grouped pages by PDF so that pages from a document under evaluation were not included in that fold's training set.

The recorded development results were:

- grouped leave-one-PDF-out accuracy: approximately **99.4%**;
- drawing-page recall: **100%** on the development labels;
- untouched audit: **94/94 pages** correctly classified across 24 separately sampled PDFs.

The audit result should not be interpreted as proof of universal 100% accuracy. The 95% Wilson lower bound from the 94-page audit was approximately 96.1%, and performance may be lower on documents from other periods, countries, printing processes, scan qualities, or page-layout conventions.

## Runtime benchmark

On the development computer, the original full benchmark processed **1,000 PDF files containing 3,106 pages in approximately 3 minutes 40 seconds** (about 3–4 minutes). This measurement covered PDF rendering, visual-feature extraction, and page prediction. It corresponds to approximately 0.071 seconds per page, or about 14 pages per second.

Runtime will vary with CPU performance, storage speed, PDF complexity, page count, scan resolution, and background system load. The figure above is an observed benchmark on the development computer, not a guaranteed runtime on other machines.

## Method

### 1. PDF rendering

Each PDF page is rendered at **90 DPI** with PyMuPDF. Rendering at this resolution keeps the classifier fast while retaining the large-scale visual structure needed to distinguish text pages from drawing pages.

### 2. Image normalisation

The rendered page is converted to grayscale and resized to a working width of **600 pixels**, preserving its aspect ratio. Pixels darker than **225** are treated as ink.

### 3. Visual feature extraction

The final model uses nine visual features and deliberately does not use the page's position in the document:

1. ink ratio;
2. connected-component count per megapixel;
3. fraction of small connected components;
4. fraction of ink in large components;
5. fraction of wide components;
6. fraction of tall components;
7. fraction of ink belonging to long horizontal or vertical lines;
8. standard deviation of row ink density;
9. standard deviation of column ink density.

Text pages generally contain many small, regularly arranged components. Drawing pages more often contain long lines, larger connected structures, and different row/column density patterns. Ink density alone is insufficient because dense text pages can contain more ink than sparse drawings.

### 4. Classification

The final V2 classifier is a scikit-learn `ExtraTreesClassifier` configured with:

```python
ExtraTreesClassifier(
    n_estimators=500,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=1,
)
```

The model returns a drawing probability for every page. Probabilities of 0.5 or above are labelled `drawing`; lower probabilities are labelled `text`. The output also includes confidence, defined as the larger of the text and drawing probabilities.

## Installation

Python 3.10 or newer is recommended.

Clone or download the repository, open a terminal in this folder, and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Only load `.joblib` model files that you generated yourself or obtained from a trusted source. Python model deserialisation is not safe for untrusted files.

## Option A: run the benchmark notebook

1. Put the PDF files to be tested in `input_pdfs/`.
2. Open `page_classifier_benchmark.ipynb` in Jupyter.
3. Keep `RUN_MODE = "sample"` to test up to 100 PDFs, or set `RUN_MODE = "full"` to process every PDF in `input_pdfs/`.
4. Run all cells.

The notebook reads `enriched_training_cross_validation.csv`, reconstructs V2, and then benchmarks PDF rendering, feature extraction, and prediction. Reconstructing the model from 181 rows is normally very quick. Results are written to `benchmark_outputs/`:

- `<mode>_predictions.csv`;
- `<mode>_pdf_timings.csv`;
- `<mode>_summary.json`.

Use `page_classifier_benchmark_template.ipynb` when the classifier code, training CSV, input PDFs, and output folder live in different locations. Replace every `Path("xxx")` entry as instructed by the English `# TODO` comments.

## Option B: use the command-line program

The command-line workflow separates one-time training from routine prediction.

### Train and save a local model

Run this once from the package folder:

```bash
python page_classifier.py train \
  --training-csv enriched_training_cross_validation.csv \
  --model-output artifacts/page_classifier_v2.joblib
```

This creates:

```text
artifacts/page_classifier_v2.joblib
artifacts/page_classifier_v2.metadata.json
```

The metadata records preprocessing settings, feature order, dependency versions, training-table size, and the SHA-256 hash of the training CSV.

### Classify one PDF

```bash
python page_classifier.py predict \
  --model artifacts/page_classifier_v2.joblib \
  --input input_pdfs/example.pdf \
  --output predictions/example_pages.csv
```

### Classify a folder of PDFs

```bash
python page_classifier.py predict \
  --model artifacts/page_classifier_v2.joblib \
  --input input_pdfs \
  --output predictions/all_pages.csv
```

Add `--recursive` to include PDF files in subdirectories:

```bash
python page_classifier.py predict \
  --model artifacts/page_classifier_v2.joblib \
  --input input_pdfs \
  --output predictions/all_pages.csv \
  --recursive
```

On Windows PowerShell, either enter a command on one line or replace each Bash continuation character (`\`) with a PowerShell backtick (`` ` ``).

## Prediction output

The output CSV contains one row per page. Important columns are:

| Column | Meaning |
|---|---|
| `pdf` | Source PDF filename. |
| `pdf_path` | Resolved source path. |
| `page` | One-based page number. |
| `page_count` | Number of pages in the PDF. |
| `drawing_probability` | Model probability for the drawing class. |
| `predicted_page_type` | `text` or `drawing`. |
| `confidence` | Confidence of the selected class. |

The nine extracted feature columns are retained to support auditing and error analysis.

## Limitations and appropriate use

- The model was developed for Swedish historical patent scans from 1895–1900. Validate it before applying it to substantially different collections.
- Mixed pages are forced into one of the two classes; there is no third `mixed` class.
- Poor scans, unusual page borders, photographic plates, handwritten annotations, and severe bleed-through can reduce accuracy.
- Low-confidence pages should be reviewed manually in high-stakes digitisation workflows.
- This classifier separates page types only. It does not crop figures, recognise figure numbers, match repeated drawings, or stitch fragments across pages.

