# Dataset Files

This directory is intentionally data-only and ignored by git.

Download the shared dataset bundle from OneDrive and place/extract it here so the layout is:

```text
experiments/data/datasets/
  assistments/
    raw/
    train.csv
    test.csv
    metadata.json
  ednet/
    raw/
    train.csv
    test.csv
    metadata.json
  oulad/
    raw/
    train.csv
    test.csv
    metadata.json
  oulad_weekly/
    train.csv
    test.csv
    metadata.json
```

The tracked Python code for loading and preprocessing these files lives in `experiments/data/dataset_scripts/`.

`oulad_weekly` is derived from `oulad/raw/oulad.zip`. It retains explicit
zero-activity weeks up to withdrawal/course completion and carries learner-level
dropout, failure, course, and demographic attributes. KDD Cup 2010 is not part
of the locked three-dataset paper benchmark; its adapter remains optional
development support only.
