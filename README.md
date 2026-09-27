# Anonymous review materials

This repository contains the manuscript, supplementary material, and aggregate result tables for a learning-analytics submission. It does not contain student-level records or model-training code.

- `main.pdf` and `supplementary.pdf` are the readable manuscript and supplement.
- `main.tex`, `supplementary.tex`, `references.bib`, the class/style files, `figures/`, and `tables/` are the LaTeX sources.
- `supplement-generated/` contains the aggregate result tables, applicability records, and status files referenced by the supplement. Missing or unsupported comparisons are documented there rather than represented as zero.

To rebuild the PDFs with Tectonic, run `tectonic main.tex` and `tectonic supplementary.tex` from the repository root. The included PDFs are the submission snapshots.
