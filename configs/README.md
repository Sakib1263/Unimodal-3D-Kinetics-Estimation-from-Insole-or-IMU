# Experiment configurations

Each JSON file defines one target, foot, model, and input configuration. Unknown or repeated settings fail validation instead of being ignored. UTF-8 files with a Windows byte-order mark are accepted.

| Configuration | Article route | Default input and target |
| --- | --- | --- |
| [km2kn.json](km2kn.json) | Km2Kn, under journal revision | Six bilateral ankle-region kinematic trajectories â†’ left-foot `Fy`. |
| [insole_grfm.json](insole_grfm.json) | Published smart insole study | Left-foot pressure channels `p1`â€“`p16` â†’ left-foot `Fy`. |
| [insole_cop.json](insole_cop.json) | Published smart insole study | Right-foot pressure channels `p1`â€“`p16` â†’ right-foot treadmill `CoPz`. |
| [source_native_example.json](source_native_example.json) | Corrected preparation example | New MATLAB exports with training-fitted reversible normalization. Source units still require verification. |

These are maintained-workflow examples, not frozen reproductions of the articles. Read [PAPERS.md](../docs/PAPERS.md) for protocol differences and notebook links.

## Settings to review

- **Data:** `data_format`, `data_files`, `dataset_state`, `target_units`, `normalization`, and `allow_legacy_channel_order`. Paths in `configs/` use the repository root. A JSON elsewhere uses its own folder. `project_dir` can override that base; absolute paths are accepted. The same rules apply to outputs.
- **Task:** `study`, `target`, `side`, `target_source`, `input_sides`, and `feature_names`. Available targets are `Fx`, `Fy`, `Fz`, `Mx`, `My`, `Mz`, `CoPx`, and `CoPz`. For insole inputs, `input_sides` controls the pressure streams independently of the target foot.
- **Model:** `model`, `depth`, `width`, `kernel_size`, `cardinality`, `pooling`, and `deep_supervision`. The stored class/method name is the implementation identity; it does not certify correspondence to a paper's named network.
- **Windows:** `window_length`, `sample_rate`, and `train_stride`. Overlap is extracted within individual source records only. Records already equal to `window_length` generate one window regardless of a smaller stride.
- **Selection:** `min_abs_pcc` and `max_pairwise_correlation` are fitted on training records only. Set 0 and 1 respectively to disable target-correlation and redundancy pruning. Constant input channels are removed using the training data. Pairwise pruning is not labeled as multivariate VIF.
- **Evaluation:** `fold_ids` selects distinct outer folds 1â€“5. `validation_subjects` defaults to one separate participant from each training pool. `seed` controls repeatable software randomness; cross-device numerical identity is not promised.
- **Training:** `epochs`, `batch_size`, `learning_rate`, `loss`, `patience`, and `lr_patience`. Supported losses are MAE, Huber, and MSE. Deep supervision uses weight 1 for `out` and 0.5 for each auxiliary head.
- **Artifacts:** `output_dir` controls where new, uniquely named run directories are created; `project_dir` optionally sets the base for data and outputs.

For target-specific experiments, copy the configuration to a descriptive filename and change the task fields. Run `python -m kinetics inspect --config <file>` before training. Keep the configuration with the resulting run artifacts.


## First experiment checklist

1. Copy the appropriate JSON, preferably within `configs/`, and give it a descriptive name.
2. Check `study`, `target`, `side`, `target_source` and the source units. Never claim physical units for a legacy normalized target.
3. Set `data_files` and confirm its metadata. Leave `allow_legacy_channel_order=false` for newly labeled exports; the three historical examples explicitly enable it because their original files may lack labels.
4. Select `feature_names` in the intended order. For pressure inputs, set `input_sides` independently of the target foot; an empty list means the target foot only. Leave `input_sides=[]` for Km2Kn.
5. Use one fold and one epoch for an initial execution, then restore your scientific protocol. A JSON boolean is `true`/`false`, not a quoted string; all list-valued settings must remain JSON arrays.
6. Run `python -m kinetics inspect --config <file>`, then train. Keep the saved configuration with the run.

All omitted fields take defaults from [ExperimentConfig](../kinetics/config.py). Unknown keys, duplicate resolved paths, wrong types and invalid ranges fail early. `window_length` must be divisible by `2**depth`. Fixed-cardinality `MLMRSNet` additionally requires divisibility by `2**(depth + cardinality - 1)` when cardinality is nonzero. `MultiResUNet` requires width at least 6. `cardinality` controls fixed-scope `MLMRSNet`; other multiscale variants use their own depth-dependent scope schedule.
