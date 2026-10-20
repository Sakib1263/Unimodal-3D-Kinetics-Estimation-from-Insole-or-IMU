# Contributing

This repository is maintained for the Km2Kn and Insole2Kn3D studies. Read the [article correspondence](docs/PAPERS.md) and [data contract](docs/DATA_CONTRACT.md) before changing scientific behavior.

## Verify a change

Save source and documentation as UTF-8 with LF line endings, as specified in `.editorconfig`. Use plain punctuation in the English documentation and keep notebook source free of copied rendering artifacts. The repository validator checks for common encoding errors.

Use a separate Python 3.12 environment. Install `python -m pip install -e ".[maintain]"`, then run:

```bash
python -m ruff check .
python scripts/validate_repository.py
python -m kinetics demo
```

For notebook changes, execute the affected notebook from a fresh kernel in demo mode, enable its one-epoch training path, inspect the figures, and run its saved-model reload comparison. Keep validation outputs outside the maintained source tree. Source notebooks should have cleared execution outputs before sharing. A regression probe should reproduce a meaningful failure or scientific-contract violation; remove temporary fixtures and generated checks after preserving their results externally.

For a data-loader, model or evaluation change, verify the affected shape, labels, units, participant isolation, inverse transform and serialization behavior with representative inputs. Synthetic success does not validate article-level results. Changes to preprocessing, architecture or evaluation require a separate real-data comparison before scientific conclusions.

## Describe the change

Explain the triggering problem, the resulting behavior, the affected study/notebook and the checks performed. Record any change to source signals, normalization, feature selection, windows, splits, architectures, checkpoint selection or metrics. Preserve the exact configuration and commit used for an earlier reported result.

For a bug report, provide the code version, configuration, environment, traceback and smallest reproducible example. Identify the target, foot, reference source, units and split for a data issue. Prefer synthetic examples when source data cannot be redistributed.

Do not commit datasets, credentials, generated runs or trained checkpoints. Keep missing reference values explicit rather than replacing them with zeros. Preserve original author/license notices and scholarly attribution. Supplied manuscript drafts and publisher PDFs are reference material and are not part of the code release.
