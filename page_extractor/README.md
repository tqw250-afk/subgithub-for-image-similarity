# Patent Drawing Page Extractor

This package is the page-first extraction and cross-page assembly step that follows the sibling `page_classifier` package. It accepts patent PDFs and the classifier's page-level prediction CSV, keeps every predicted drawing page, and joins pages only when they are confidently parts of the same scanned plate.

The package is intended for reproducible batch processing and can be run either from the command line or from the included notebook. It does not depend on the development notebooks, historical test folders, saved intermediate images, manual masks, or PDF-specific coordinates.

## What it does

1. Reads `text` / `drawing` predictions produced by `page_classifier.py predict`.
2. Creates one delivery folder for every input PDF.
3. Renders every predicted drawing page at 150 DPI.
4. Detects confident cross-page relationships using local feature matches, page-boundary continuity, and geometric checks.
5. Assembles confidently related pages into one PNG.
6. Preserves unrelated drawing pages as separate images.
7. Marks unresolved but plausible relationships as `HOLD` without discarding either page.
8. Verifies that every predicted drawing page is delivered exactly once.

This is deliberately a **page/plate extractor**, not a figure-level cropper. A page containing several figures remains a complete page unless it is joined to another page. One PDF may therefore produce one image or several independent images; the pipeline never forces all drawings from one PDF into a single PNG.

## Package contents

| File | Purpose |
|---|---|
| `page_extractor.py` | Standalone command-line batch program and importable Python module. |
| `page_extractor.ipynb` | Ready-to-run notebook using the same module and algorithm. |
| `requirements.txt` | Python dependencies. |
| `input_pdfs/` | Optional default location for input PDFs. |
| `input_predictions/` | Optional default location for the page-classifier CSV. |

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Required input

### PDFs

Provide either one PDF or a folder of PDFs. PDF filenames must be unique when recursive input is used.

### Page-classifier CSV

The CSV must contain at least these columns:

| Column | Meaning |
|---|---|
| `pdf` | PDF filename, including `.pdf`. |
| `page` | One-based page number. |
| `predicted_page_type` | `text` or `drawing`. |

The normal way to create this file is the sibling classifier:

```bash
python ../page_classifier/page_classifier.py predict \
  --model ../page_classifier/artifacts/page_classifier_v2.joblib \
  --input ./input_pdfs \
  --output ./input_predictions/full_predictions.csv
```

## Command-line use

Run the extractor from this folder:

```bash
python page_extractor.py extract \
  --input ./input_pdfs \
  --predictions ./input_predictions/full_predictions.csv \
  --output ./outputs/run_001
```

On Windows PowerShell the same command can be written on one line:

```powershell
python .\page_extractor.py extract --input .\input_pdfs --predictions .\input_predictions\full_predictions.csv --output .\outputs\run_001
```

Useful options:

- `--recursive` searches subfolders for PDFs.
- `--max-pdfs 30` processes only the first 30 PDFs for a smoke test.
- `--render-dpi 150` changes the drawing-page rendering resolution.
- `--no-review-sheets` skips visual review sheets while retaining metrics and delivery images.

The output directory must be new. The program refuses to overwrite an existing run so that earlier results cannot be silently mixed with a later run.

## Notebook use

Open `page_extractor.ipynb`, edit the configuration cell if necessary, and run all cells. By default it uses:

- `input_pdfs/` for PDFs;
- `input_predictions/full_predictions.csv` for page predictions;
- `outputs/notebook_run` for results.

The notebook imports `page_extractor.py`; it does not contain a second copy of the extraction algorithm. This keeps command-line and notebook results consistent.

The configuration can also be supplied through environment variables:

| Variable | Default |
|---|---|
| `PAGE_EXTRACTOR_PDF_INPUT` | `input_pdfs` |
| `PAGE_EXTRACTOR_PREDICTIONS` | `input_predictions/full_predictions.csv` |
| `PAGE_EXTRACTOR_OUTPUT` | `outputs/notebook_run` |
| `PAGE_EXTRACTOR_RUN_MODE` | `full` (`sample` is also supported) |
| `PAGE_EXTRACTOR_SAMPLE_COUNT` | `30` |
| `PAGE_EXTRACTOR_RECURSIVE` | `false` |

## Output structure

```text
run_001/
├── Extracted image/
│   ├── 1234_5678/
│   │   ├── SINGLE__page_02.jpg
│   │   └── MERGED__pages_03-04.png
│   └── another_pdf/
├── audit/
│   ├── raw_pages/
│   └── reviews/
├── metrics/
│   ├── adjacent_pair_metrics.csv
│   ├── boundary_actions.csv
│   ├── output_manifest.csv
│   ├── pdf_summary.csv
│   └── overlap_assembly_failures.csv
└── summary.json
```

A folder is created under `Extracted image` for **every input PDF**, including PDFs for which the classifier found no drawing pages.

### Delivery labels

| Prefix | Meaning |
|---|---|
| `SINGLE` | A complete drawing page that did not require stitching. |
| `MERGED` | Two or more pages confidently assembled into one plate. |
| `HOLD__...` | The image is delivered, but a plausible neighbouring relationship remains uncertain and should be reviewed. |

`HOLD` is not a failed extraction and does not remove content. The `audit/reviews` sheets make those cases easier to inspect.

## Method summary

The algorithm combines three kinds of evidence:

- local SIFT feature matches and RANSAC geometry;
- continuity near facing page boundaries;
- local translation consensus around the proposed seam.

It first assembles high-confidence overlaps, then tests unresolved neighbouring components. General vetoes reject transformations that look like rotated/scaled duplicate figures, direction-conflicting partial matches, or implausibly oversized homographies. These are general geometric rules; there are no filename-specific exceptions.

## Validation guarantees

Before a run is accepted, the program checks that:

- every predicted drawing page appears in the delivery exactly once;
- no drawing page appears in two outputs;
- every input PDF has a corresponding delivery folder;
- no TXT sidecars were generated.

Run statistics and the algorithm version are written to `summary.json`.

## Validated benchmark

On 23 October 2026, the packaged command-line implementation was rerun from the original PDFs and page-classifier CSV across the full 1,000-PDF development set. It produced:

- 3,106 source pages;
- 1,527 predicted drawing pages across 902 PDFs;
- 989 delivery images, including 460 multi-page assemblies;
- 64 `HOLD` images across 28 PDFs;
- 98 correctly created empty delivery folders for PDFs with no predicted drawing pages;
- 38 automatic local seams and 41 automatic iterative bridges;
- zero overlap-assembly failures;
- exact-once delivery of every predicted drawing page;
- zero differences across all 989 manifest rows when compared with the previously reviewed reference run.

The complete extraction and assembly run took approximately **18 minutes** on the development computer. This includes 150-DPI drawing-page rendering, matching, assembly, delivery writing, metrics, and review sheets; it does not include the preceding page-classifier run. Runtime will vary with CPU performance, OpenCV build, PDF complexity, storage speed, and the number of candidate drawing-page pairs. These results demonstrate reproduction of the reviewed development run, not universal accuracy on unseen collections.

## Scope and limitations

- The current thresholds were developed on scanned Swedish historical patents, mainly from 1895–1900. Other countries, scan pipelines, page sizes, and periods should be audited before large-scale use.
- Similar mechanical figures can still create ambiguous local matches. Such cases are preserved as separate `HOLD` outputs instead of being force-merged.
- The program does not identify `Fig. 1`, `Fig. 2`, and similar semantic labels.
- It does not split a multi-figure drawing page into separate figures.
- Classification errors from the upstream page classifier carry into this step; inspect empty PDF folders and low-confidence classifier predictions when completeness matters.

## Reproducibility and GitHub use

All operational code is contained in this folder. No trained stitching model or external parameter file is required. For repeatable results, commit the source files and pin dependency versions in your deployment environment. OpenCV feature matching is deterministic for these fixed inputs in the tested setup, but very small numerical differences can occur between OpenCV builds or platforms.

Do not commit source patent PDFs, generated output folders, or prediction files unless their licences and repository size are appropriate. The included `.gitignore` excludes these locations by default.
