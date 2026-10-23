# subgithub for UPHG

This repository is intended to resolve image-related issues in the History Patent project.

## Branches

- `master`: the main integration branch.
- `Qinwu`: Qinwu's working branch for uploading and updating documents.
- `Garveet`: Garveet's working branch for uploading and updating documents.

Please upload work to your personal branch. Changes can be merged into `master` when they are ready.

## Page classifier

The `Qinwu` branch contains a downloadable classifier for separating text and
drawing pages in scanned Swedish patent documents from 1895–1900. See the
[page-classifier documentation](page_classifier/README.md) for the method,
training data, limitations, benchmark notebook, and command-line instructions.

## Page extractor

The `page_extractor` package consumes the page-classifier CSV, preserves every
predicted drawing page, and assembles pages that are confidently parts of the
same scanned plate. It includes an independently runnable notebook and command-
line script, complete delivery folders, audit sheets, metrics, and exact-once
page-coverage validation. See the
[page-extractor documentation](page_extractor/README.md) for installation,
usage, output labels, the validated 1,000-PDF benchmark, and current
limitations.
