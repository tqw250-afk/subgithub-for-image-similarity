# Run All verification — passed

Verified on 2026-11-10. The integrated package passed in both forms:

- `01_finnish_page_classifier`: one complete notebook and one matching Python script;
- `02_embedding_and_patent_retrieval`: one complete notebook and one matching Python script.

Both integrated notebooks completed with Run All and no error outputs. Both matching Python scripts then completed from the command line. The final Drawing-page fallback CSV and JSON files were reproduced.

## Verified final scale

```text
Swedish drawing pages:       214
Finnish drawing pages:       151
Swedish page-region vectors: 1,070
Finnish page-region vectors:   755
Evaluable known patent pairs:   89
```

## Verified final exploratory result

The best tested fusion on the current known pairs is `hybrid_page50`, which gives equal weight to standardized crop and page-region channels:

```text
Recall@1:  0.7640
Recall@3:  0.8652
Recall@5:  0.9101
Recall@10: 0.9438
Recall@20: 0.9663
MRR:       0.8245
```

These values are exploratory results on known pairs, not an independently validated final model claim.

