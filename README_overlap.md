# Patent Cross-Page Stitcher & Deduplication Pipeline (Stage A)

This module mathematically assesses sequential historical patent pages to detect physical diagram overlaps, rotate misaligned continuations, drop redundant duplicates, and fuse matching pages into single, unified graphical sheets.

## 🚀 Pipeline Architecture (Waterfall Logic)

The algorithm evaluates sequential patent pages (N vs N+1) using a three-phase geometric waterfall:

1. **Standard Edge Overlap (Normal & Reverse):** Slices the inner 40% edges of the canvases and searches for continuous structural lines (using ORB feature extraction and RANSAC affine transforms). If it finds a match, it merges them.
2. **Full-Canvas Deduplication:** If the edge check fails, it extracts 10,000 ORB features across the full canvas and calculates a 2D Partial Affine projection. If >90% of Page B physically lands inside the borders of Page A, Page B is flagged as a `DUPLICATE_OR_VARIANT` and permanently dropped to prevent data pollution.
3. **Vertical & Rotated Continuations:** If the full-canvas projection determines that Page B protrudes into new, unrepresented spatial territory, Page B is physically rotated and permanently stitched to Page A.

## 📂 Output Structure

For each processed patent, the pipeline creates a clean, unified output directory ready for downstream Figure Extraction:

```text
patent_cross_page_results/
├── 7918/
│   ├── final_ready_canvases/         # <-- DOWNSTREAM TARGET FOLDER
│   │   ├── 7918_page_6.png           # (Surviving standalone pages)
│   │   └── sheet_pages_0004_0005.png # (Massive stitched canvases)
│   ├── metadata.json                 # (Local patent processing metrics)
│   └── overlap/                      # (Generated ONLY if --save-debug is passed)
└── overall_summary.json              # (Global execution metrics & run time)


## ⚙️ Usage

Run the pipeline from the command line, pointing it at your folder of extracted patent sketch images.

#Standard Run (Fast, Silent, Clean output):

```bash
python patent_cross_page_pipeline_v5_final.py --input-dir sketches_from_7000_pdfs --output-dir 
```

# patent_cross_page_results
# Debug Run (Generates visual RANSAC bounding boxes and green inlier lines):

```bash
python patent_cross_page_pipeline_v5_final.py --input-dir sketches_from_7000_pdfs --output-dir patent_cross_page_results --save-debug
```

# 📊 Metrics Tracking

The overall_summary.json file in the root output folder will automatically track execution speeds (in seconds and minutes) and provide a categorical breakdown of pipeline actions (e.g., pairs tested, edges accepted, rotated continuations, duplicates dropped, and standalone pages successfully preserved).



{
  "execution_metrics": {
    "execution_time_seconds": 1957.87,
    "execution_time_minutes": 32.63
  },
  "dataset_composition": {
    "total_patents": 6670,
    "total_extracted_pages": 12948
  },
  "overlap_results": {
    "adjacent_pairs_tested": 6278,
    "edge_overlap_accepted": 4236,
    "vertical_rotated_continuations": 143,
    "duplicates_dropped": 220,
    "stitched_runs_created": 4018,
    "standalone_pages_preserved": 4544
  }
}
