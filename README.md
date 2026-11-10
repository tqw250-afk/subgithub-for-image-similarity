# Swedish–Finnish Patent Figure Retrieval

This package contains two complete, bilingual workflows in both Jupyter Notebook and Python formats:

```text
notebooks/
  01_finnish_page_classifier.ipynb
  02_embedding_and_patent_retrieval.ipynb

scripts/
  01_finnish_page_classifier.py
  02_embedding_and_patent_retrieval.py
```

The first workflow builds and validates the Finnish drawing-page classifier. The second starts from the classifier outputs and runs through drawing-page segmentation, DINOv3 embedding, Swedish–Finnish retrieval, multi-figure pooling, and Drawing-page fallback.

SIFT/RANSAC computation and later geometric reranking are not part of this release. The reviewed hard-case tables used by the pooling analysis are retained as fixed reference inputs.

## Included files

```text
README.md
RUN_ALL_VERIFICATION.md
requirements.txt
.gitignore

notebooks/
  01_finnish_page_classifier.ipynb
  02_embedding_and_patent_retrieval.ipynb

scripts/
  01_finnish_page_classifier.py
  02_embedding_and_patent_retrieval.py

reference_annotations/
  figure_pair_ai_annotations.csv
  hard_case_summary.csv
```

The notebook and Python versions contain the same processing stages. The notebooks are intended for step-by-step review and Run All execution; the scripts provide the same workflow for command-line use.

## How the bundled CSV files are used

The two CSV files under `reference_annotations/` are small evaluation inputs, not training data and not embedding indexes.

- `figure_pair_ai_annotations.csv` maps reviewed annotation IDs to Swedish and Finnish patent/figure IDs. Part 7 merges this mapping with the human-review decisions stored in the workflow and exports `human_review_feedback_round1.csv`.
- `hard_case_summary.csv` identifies the difficult query patents used in the pooling rank-change table.

Neither file changes the DINOv3 vectors, the Finnish page classifier, the YOLO segmentation, or the initial cross-country baseline. They only support the reviewed hard-case analysis. The public copies contain only the fields read by the code; local image paths and contact-sheet paths have been removed.

Other CSV files are generated during Run All and pass data between stages. For example, page predictions feed segmentation, segmentation manifests feed embedding, and patent-level pooling scores feed Drawing-page fallback. These generated files are intentionally not committed because they contain workspace-specific paths and can be reproduced from the notebooks.

## Workflow 1 — Finnish page classifier

`01_finnish_page_classifier` contains three integrated parts:

1. render the matched patent pages and create full-page DINOv3 embeddings;
2. train the Finnish drawing-page classifier;
3. run the unseen-patent blind validation.

The final classifier uses a 768-dimensional full-page DINOv3 embedding, StandardScaler, and balanced logistic regression. OCR, handwriting recognition, patent metadata, and OpenCV geometry are not final prediction features.

Main outputs:

```text
embedding_atm/finnish_page_classifier_pilot/
embedding_atm/finnish_page_classifier_v1/
embedding_atm/finnish_page_classifier_v1_blind_validation/
```

Final model:

```text
embedding_atm/finnish_page_classifier_v1/models/finnish_page_classifier_dinov3_candidate_v1.joblib
```

## Workflow 2 — Embedding and patent retrieval

`02_embedding_and_patent_retrieval` contains eight integrated parts:

1. segment Finnish drawing pages with YOLO;
2. filter Finnish crops using DINOv3;
3. build the Finnish figure index;
4. segment the matched Swedish drawing pages;
5. build the Swedish figure index;
6. run the DINO-only cross-country baseline;
7. compare patent-level multi-figure pooling methods;
8. add the Drawing-page fallback channel.

Drawing-page fallback embeds every drawing page as five views:

```text
full
top_left
top_right
bottom_left
bottom_right
```

The four large regions cross the centre line by 5% on each side, producing 10% overlap around the page centre.

Final outputs:

```text
embedding_atm/cross_country_retrieval/tables/page_region_fallback/
```

Important files:

- `crop_page_fusion_scores.csv`
- `crop_page_fusion_metrics.csv`
- `known_pair_crop_page_ranks.csv`
- `segmentation_problem_rank_changes.csv`
- `page_region_fallback_summary.json`

## Required workspace layout

Patent PDFs, trained model weights, and generated embedding arrays are not included in the GitHub package. Use the following layout:

```text
workspace/
  Swe-Fin-matched-patents/
  sample_data/
  swe_pre_process_pipeline-main/
    swe_pre_process_pipeline-main/
      artifacts/
        best.pt
        page_classifier_v2.joblib
      src/
        page_classifier.py
  embedding_atm/
    github/                      # This package
    ...                          # Generated outputs and caches
```

The code searches upward for a parent directory containing `embedding_atm`, so the files can be launched from the workspace root or from inside the package.

## Installation

Python 3.10–3.13 is recommended:

```bash
python -m pip install -r embedding_atm/github/requirements.txt
```

The first DINOv3 run may download pretrained `timm` weights. A GPU is optional; CPU execution is supported but slower.

## Running the notebooks

Run the two notebooks in this order:

1. `notebooks/01_finnish_page_classifier.ipynb` — Run All;
2. `notebooks/02_embedding_and_patent_retrieval.ipynb` — Run All.

The second notebook requires the completed classifier outputs from the first notebook.

## Running the Python versions

From the workspace root:

```bash
python embedding_atm/github/scripts/01_finnish_page_classifier.py
python embedding_atm/github/scripts/02_embedding_and_patent_retrieval.py
```

The scripts use a non-interactive Matplotlib backend, so preview generation does not block command-line execution.

## Reproducibility notes

- DINOv3 embeddings are L2-normalized before cosine-similarity search.
- Rendering and embedding caches are reused only when their keys match.
- Output summaries include input hashes where applicable.
- Known matches are defined at the patent level, not necessarily at the individual-figure level.
- The tested fusion weights are exploratory and still require validation on an independent patent collection.

See `RUN_ALL_VERIFICATION.md` for the execution check and verified final scale.

