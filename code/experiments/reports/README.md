# Reference reports

This directory contains tracked, immutable report snapshots for audit and
publication. The pipeline never writes new runs here.

- `standard/` contains the active RQ1-only evaluation snapshots for the
  unmodified generators. Each seed report is linked by SHA-256 to both its
  archived source report and the unchanged standard synthetic CSV. The
  multi-seed summaries contain only metrics whose names begin with `rq1.`.
- `standard_vs_tail_targeted/` is the destination for the curated final reports.
  These completed reports compare the standard baseline with the corrected RQ2
  tail-targeted arm.
- `archive/` is an optional, git-ignored local workspace for superseded
  development snapshots. It is never an active paper-results source.

Read [`RESULTS_GUIDE.md`](RESULTS_GUIDE.md) for the active three-dataset matrix,
the exact meaning of every report block and statistic, RQ1/RQ2 interpretation,
dataset-specific targets and tail groups, and known model-collapse outcomes.

The active RQ1 snapshots can be reproduced without training or evaluation:

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/promote_standard_reports.py
```

This command extracts only `publication.metrics.rq1` and standard-arm
diagnostics. It never promotes the superseded RQ2/tail-targeting blocks from
the mixed historical reports.

On the original development machine, the active standard OULAD CSVs are first
migrated to current derived outcomes with
`scripts/migrate_oulad_standard_outcomes.py`. This changes no generator-produced
behavior and therefore leaves RQ1 metric values unchanged; rerunning the
promotion command refreshes only artifact/provenance hashes in the RQ1
snapshots.

Local evaluation reports and manifests are written to the git-ignored
`experiments/runs/standard_vs_tail_targeted/` workspace. Raw synthetic CSVs are
stored separately under `experiments/outputs/standard/` and
`experiments/outputs/tail_targeted/`.

After every selected dataset/model cell has completed its strict audit, promote
the final RQ1/RQ2 snapshots without rerunning a generator or evaluator:

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/promote_final_reports.py
```

The promotion command accepts `--datasets`, `--models`, and `--seeds`. It rejects
failed or incomplete reports, requires the locked publication schema, validates
the dataset/model/seed identity and SHA-256 of both the standard and
tail-targeted synthetic artifacts, embeds their complete generation metadata,
rewrites repository-local absolute paths as portable relative paths, and writes
only to `standard_vs_tail_targeted/`. Promotion never changes metric values.

Optional CSV exports for every curated dataset/model cell can then be generated
locally with:

```bash
PYTHONPATH=experiments .venv/bin/python experiments/scripts/reports_to_csv.py \
  --expected-seeds 20260703 20260704 20260705
```
