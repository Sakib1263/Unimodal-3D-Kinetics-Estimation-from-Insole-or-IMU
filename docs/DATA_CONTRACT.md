# Data contract and preprocessing

The repository contains code and historical result exports. The source study dataset is not bundled. Training must use the synchronized, matched participant-trial data associated with the article of interest. The supported participants are P04-P12; source axes are x = anteroposterior, y = vertical, z = mediolateral. Confirm these conventions against the source files before interpreting any axis.

## Array and label requirements

The Python workflow uses `X: (record, time, channel)` and `Y: (record, time, 1)`. Every row must represent one contiguous, synchronized record or an already extracted contiguous window. Rows must never join participants, trials, or separated contacts. Inputs and targets share the same temporal samples. Nonfinite values are rejected rather than replaced with zeros. Complex, text and boolean signal arrays are rejected before conversion. Source target precision is retained; model inputs and normalized optimization targets use float32.

`subject`, `speed`, `mode`, and `record` contain one label per row. Participant IDs are integers 4-12. Speed is stored in the source convention; the provided MATLAB exporter parses source filenames in tenths of km/h. Mode is 0 for walking and 1 for running. Record identifiers should retain the source trial and interval.

Supported targets are `Fx`, `Fy`, `Fz`, `Mx`, `My`, `Mz`, `CoPx`, and `CoPz`. A separate model is trained for each target and foot. `LL` and `RL` denote left and right foot. Vertical CoP and center of mass are not substituted for these eight targets.

## MATLAB v7.3 exports

Place the historical `Data_Preprocessed_Walk.mat` and `Data_Preprocessed_Run.mat` files under `data/`, or change `data_files` in the configuration. MATLAB HDF5 arrays are stored as `(channel,time,record)` and explicitly transposed into the Python contract. Ordinary non-HDF5 MAT files are not accepted by this loader.

| Field | Use |
| --- | --- |
| `data_IMU`, `label_IMU` | Kinematic trajectories for Km2Kn. The four excluded channels are `pelvis_tx`, `pelvis_tz`, `mtp_angle_r`, and `mtp_angle_l`. |
| `data_Insole_LL`, `data_Insole_RL`, `label_Insole` | Insole streams. Only the verified first 16 pressure channels per selected foot are model inputs. |
| `data_Force_FP_LL`, `data_Force_FP_RL`, `label_FP` | Instrumented treadmill targets. CoP must be present here to evaluate treadmill CoP. |
| `subinfo`, `speedinfo` | Participant and speed labels. Both are required. |
| `modeinfo` | Locomotion labels. Historical files may instead use an unambiguous Walk/Run filename. |
| `source_file`, `window_start_sample` | Trial and interval provenance produced by the maintained exporter. |

Labels are decoded from MATLAB cell arrays. The historical-order fallback is available only through explicit `allow_legacy_channel_order: true`, with the expected channel count. It assumes the source order documented in `kinetics/data.py`; it is not a way to repair mismatched labels. If channel labels differ, review the source and implement an explicit, tested mapping. Do not rename channels by position without checking their meaning.

For CoP, `target_source: force_plate` is the default. A six-channel force/moment reference cannot supply CoP. The loader raises an error instead of silently switching to the insole. `target_source: insole` supports an explicitly labeled CoP sensor-reference diagnostic only.

The historical MATLAB script omitted `speedinfo` from its save call. A file without speed labels is incomplete for this workflow; regenerate it from the source rather than inventing labels.

## Scale and units

The original preprocessing applied min-max normalization without saving an invertible scale for each target. Such files must use:

```json
{
  "dataset_state": "legacy_normalized",
  "target_units": "normalized_source_scale_unknown",
  "normalization": "none"
}
```

Multiplication by participant body mass does not invert min-max scaling. The maintained evaluation reports legacy errors only in their stored scale; it does not label them N, N*m, or cm.

For exports that retain the source scale, select `dataset_state: source_native` and `normalization: train_minmax`. The training transform is fitted per channel across training records only. Validation inputs/targets and test inputs use training transforms without clipping. Test references stay in their original source scale. Predictions are restored with the saved training minimum and range. MAE and RMSE then use the supplied source scale, and NRMSE divides by the source-scale training target range. The same NRMSE definition applies when normalization is disabled and to legacy stored scales. A constant training reference produces null NRMSE rather than an arbitrary denominator. Verify `target_units` against the source; the example intentionally uses `source_native_unverified` until that review is complete.

## Corrected MATLAB export

The maintained `Data_Preprocessing.m` is a function for matched `Datastr` source files organized as `inputRoot/<participant>/Walk/*.mat` and `Run/*.mat`. It requires MATLAB with Signal Processing Toolbox and supports modern name-value arguments.

```matlab
Data_Preprocessing("Data", "data", SourceSynchronized=true)
```

It verifies signal dimensions, labels, finite values, sampling rates, and recording durations; resamples in the source time base; extracts aligned windows without additional per-window scaling; and saves speed, mode, source, and interval labels. Optional EMG and center-of-mass streams do not limit the usable duration. Exports are finalized only after the data and attributes are written successfully. Temporary export files are cleaned on failure. Existing outputs are never overwritten. A run with no usable windows fails clearly; a missing locomotion mode produces a warning if the other mode is available. New files are named `Data_Preprocessed_Walk_v2.mat` and `Data_Preprocessed_Run_v2.mat`.

`SourceSynchronized=true` is a user assertion that the input records are already matched and synchronized. Resampling and equal duration cannot establish synchronization. The exporter preserves source channel order and units; it does not calibrate sensors or transform coordinate frames. If the duration mismatch exceeds 0.1 seconds, it stops for review. Source filenames must encode speed as `walk_09.mat`, `run_63.mat`, or the corresponding participant-prefixed form.

Use `configs/source_native_example.json` for the new exports after verifying their units and labels. This export changes the historical preparation procedure and must be evaluated as a maintained protocol, not treated as a reproduction of the reported article results.

## NPZ alternative

NPZ files must contain numeric `X`, `Y`, `subject`, `speed`, and `mode` arrays; Unicode `record` and `feature_names` arrays; scalar numeric `sample_rate`; and scalar string metadata: `study`, `target`, `side`, `dataset_state`, `target_units`, and `target_source`. These metadata values must match the experiment configuration. Pressure feature names must be `LL_p1` through `LL_p16` and/or `RL_p1` through `RL_p16` in configured `input_sides` order. The four excluded kinematic channels are removed consistently for both MATLAB and NPZ inputs before explicit feature selection. Record identifiers must be nonempty and unique within and across input files. Metadata vectors must contain exactly one label per record; matrices are rejected. A `source_native` MATLAB export must declare `normalization_state` and `sample_rate_hz` attributes. Source filenames and mode labels must agree. Source-file and starting-sample provenance must be supplied together; starts are nonnegative integer offsets. Object arrays and pickle loading are not accepted.

Longer continuous records allow training-only overlap through `train_stride`. Split participants before extracting windows. If the source is already segmented into four-second windows, the runner cannot recover the intervening time samples needed for the manuscript's 80% overlap augmentation. It reports that limitation in the split manifest instead of joining unrelated windows.


## Writing an NPZ export

After aligning and verifying the source arrays, export one target/foot at a time. This example is a schema template; `X`, `Y` and the labels must come from your measured source records:

```python
import numpy as np

np.savez_compressed(
    "aligned_km2kn_Fy_LL.npz",
    X=X.astype(np.float32),       # (records, samples, channels)
    Y=Y.astype(np.float32),       # (records, samples, 1), here left-foot Fy
    subject=subject,              # integer vector, participants 4 through 12
    speed=speed,                  # nonnegative vector in the source convention
    mode=mode,                    # 0=walk, 1=run
    record=np.asarray(record_ids, dtype=str),
    feature_names=np.asarray(feature_names, dtype=str),
    sample_rate=256.0,
    study="km2kn", target="Fy", side="LL",
    dataset_state="source_native", target_units="N", target_source="force_plate",
)
```

Only use `target_units="N"` after verifying that the source force is in newtons. Insole exports use `study="insole"` and the pressure names specified above. Set `data_format="npz"`, list the exported files in `data_files`, and match every scalar metadata field in your experiment JSON. Different targets or source scales must use different exports/configurations.

## Raw CSV visualization

This optional path visualizes source signals; it does not prepare training datasets. Files must be named `<participant>_<trial>.csv`, for example `Subj04_walk_09.csv`. Each row is one time sample. Column order is exact:

- **Force (12 columns):** `RFx, RFy, RFz, RMx, RMy, RMz, LFx, LFy, LFz, LMx, LMy, LMz`.
- **Insole (50 columns):** left block followed by right block. Each block contains pressure `LP0`-`LP15` (or `RP0`-`RP15`), then `LAX, LAY, LAZ, LGX, LGY, LGZ, LTF, LCX, LCY` (corresponding `R` names for the right block). These CSV names follow the original source schema; the MATLAB/NPZ model inputs use `p1`-`p16`.

Headerless files use that exact positional schema. With `--header`, names/order are checked before loading; columns are never silently relabeled. Select the contact reference from the same foot as the plotted signal, and verify its threshold and source units:

```bash
python -m kinetics raw-plot data/Force --kind force --column RFy --vertical-column RFy --sample-rate 2048 --threshold 20 --units N --output runs/raw_force.png
python -m kinetics raw-plot data/Insole --kind insole --column LTF --vertical-column LTF --sample-rate 100 --threshold 20 --units N --output runs/raw_insole.png
```

The numeric values are examples requiring source confirmation. Optional `--minimum-seconds` (default 0.1) and `--cutoff` (default 15 Hz, below Nyquist) control contact inclusion and low-pass filtering. The filter is applied before resampling each stance to 101 phase points. Stances are averaged within participant/trial, then participants receive equal weight. Shading is the between-participant sample SD when at least two participants contribute. A single participant has no SD band.
