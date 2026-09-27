# Anonymous review materials

This repository contains the manuscript, supplementary material, aggregate result tables, and experiment code for a learning-analytics submission. It does not contain student-level records or generated learner trajectories.

- `main.pdf` and `supplementary.pdf` are the readable manuscript and supplement.
- `main.tex`, `supplementary.tex`, `references.bib`, the class/style files, `figures/`, and `tables/` are the LaTeX sources.
- `supplement-generated/` contains the aggregate result tables, applicability records, and status files referenced by the supplement. Missing or unsupported comparisons are documented there rather than represented as zero.
- `code/` contains the generator and evaluation pipeline, dataset-preparation scripts, configs, tests, and curated JSON report snapshots. Start with [`code/README.md`](code/README.md) and [`code/experiments/README.md`](code/experiments/README.md) for dependencies and reproduction commands. Raw datasets must be obtained separately from their original sources.

To rebuild the PDFs with Tectonic, run `tectonic main.tex` and `tectonic supplementary.tex` from the repository root. The included PDFs are the submission snapshots.
