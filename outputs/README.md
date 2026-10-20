# Historical result exports

The existing spreadsheets and CSV files in this directory are preserved research exports. They were not regenerated during pipeline maintenance. Their filenames alone do not establish a complete model, preprocessing, split, checkpoint, or evaluation provenance.

The insole CSV exports relate to the smart insole study; the IMU/Km2Kn workbooks relate to the kinematic reconstruction route. Verify the generating experiment before using any number in a comparison or manuscript. See the [associated articles and notebooks](../docs/PAPERS.md).

New executions write to unique directories under `runs/`, which is ignored by Git. Each run records its configuration, source/code hashes, participant splits, units, model checkpoints, predictions, and metric definitions. Keep those artifacts together when reporting results. Do not relabel the historical files as results from the maintained workflow.
