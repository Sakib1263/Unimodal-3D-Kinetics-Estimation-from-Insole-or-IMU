# Articles and code correspondence

This repository supports two related unimodal studies. Their sensing modalities, named reconstruction networks, preprocessing, and evaluation claims must remain distinguishable.

## Km2Kn manuscript

**Title:** Deep learning-based reconstruction of a complete 3D plantar kinetic profile from optimized inertial kinematics.

**Status:** Manuscript under journal revision as of October 2026. The final publication details and DOI will be added when available.

**Method:** Km2Kn maps processed joint- and segment-kinematic trajectories derived from Xsens inertial motion capture to eight available kinetic waveforms per foot. The manuscript calls the reconstruction backbone Km2KnNet. It compares anatomical regions, target-specific channels, laterality, architecture benchmarks, and component ablations.

**Notebook:** [Km2Kn.ipynb](../Km2Kn.ipynb).

**Configuration:** [km2kn.json](../configs/km2kn.json). Its six selected channels are the bilateral ankle angle, ankle adduction, and subtalar angle trajectories. These are processed kinematic channels, not an assertion that an independently configured sparse IMU system produces equivalent trajectories.

The manuscript describes record-level min-max normalization before resampling/windowing, 1,024 samples at 256 Hz, 80% training overlap, five outer participant folds, a 10% random validation sample from training data before augmentation, and separate target-specific models. It also describes gait-cycle evaluation and a training-reference mapping for dimensional predictions. The historical local notebook does not establish all these operations, and its existing normalized MAT files do not retain enough scale metadata to recreate every dimensional result.

## Insole2Kn3D: published smart insole article

**Title:** 3D Foot Kinetics Estimation From Distributed VGRF From Smart Insoles via 1D Domain Transformation.

**Authors:** Sakib Mahmud, Muhammad E. H. Chowdhury, and Faycal Bensaali.

**Publication:** IEEE Journal of Biomedical and Health Informatics, volume 30, issue 3, pages 2656-2667, March 2026. First published online September 5, 2025.

**DOI:** [10.1109/JBHI.2025.3605296](https://doi.org/10.1109/JBHI.2025.3605296).

**Method:** Distributed plantar vGRF from smart insoles is mapped to instrumented-treadmill-reference 3D GRF, 3D GRM, and two horizontal CoP trajectories using target-specific 1D reconstruction models. The paper introduces Ke2KeNet and evaluates plantar-region relevance.

**Notebook:** [Insole2Kn_3DGRFMCoP.ipynb](../Insole2Kn_3DGRFMCoP.ipynb) unifies forces, moments and CoP. Select one of eight targets and either foot in Section 2; all subsequent steps are shared.

**Configuration:** [insole.json](../configs/insole.json), with left-foot `Fy` as the default. Each run retains its selected target, foot and input channels in the saved configuration.

The article describes 1,024-sample windows at 256 Hz, segment min-max normalization, five outer participant folds, a random 10% validation subset of the training set, MAE validation loss, an initial learning rate of 3e-4, 20-epoch early-stopping patience, and a 10-epoch learning-rate patience. Historical code paths do not by themselves certify the reported dimensional errors or exact network implementation.

The article's original repository URL, [Insole-to-3DGRFMCoP](https://github.com/Sakib1263/Insole-to-3DGRFMCoP), resolves to this repository under its current name.

## Maintained implementation and remaining correspondence gaps

| Topic | Maintained behavior | Relation to article reproduction |
| --- | --- | --- |
| Outer test folds | P11/P12, P09/P10, P07/P08, P05/P06, P04 | Preserves the documented participant schedule. |
| Validation | One held-out participant from each outer training pool by default | Stricter than the articles' random 10% training-sample holdout; not numerically interchangeable. |
| Preprocessing | Explicit source-scale contract; optional training-fitted min-max transform | Corrected protocol. Record/window normalization in each article must be reproduced separately when required. |
| Overlap | Only within individual supplied records, after participant splitting | Pre-windowed exports cannot recover missing continuous samples for 80% overlap. |
| Kinetic reference | Instrumented treadmill/force plates for forces, moments and CoP | The selected target must be present in the reference export and correctly labeled. |
| Network identity | Explicit implemented class/method, e.g. `MLMRSNet.IMUNet` | Not automatically equated with Km2KnNet or Ke2KeNet. A component/equation/checkpoint audit remains necessary. |
| PSPNet | Newly maintained 1D pyramid-pooling baseline | Replaces calls to a nonexistent `.PSPNet()` method; not a claim to recover an original trained benchmark. |
| Dimensional errors | Inverse of a saved training transform, otherwise stored source scale | Body mass alone cannot recover discarded min-max parameters. |
| Evaluation | Signed, sample-aligned window metrics with participant-balanced summaries | Full gait-cycle segmentation, grouping, and manuscript result verification remain separate. |
| Existing spreadsheets and CSVs | Retained without modification | Historical evidence, not regenerated results or certification of the revised workflow. |

Software fixes should not be interpreted as verification of the published or submitted numerical results. Before claiming reproduction, match the actual source data, preprocessing, model components, trained checkpoint, units, splits, and evaluation representation for the article of interest.

## How to cite

Cite the article whose method is used and the exact repository commit or release. If both routes are used, acknowledge both studies with their current publication status. Km2Kn's final citation will be added when confirmed.

```bibtex
@article{mahmud2026insole3dkinetics,
  author = {Mahmud, Sakib and Chowdhury, Muhammad E. H. and Bensaali, Faycal},
  title = {{3D} Foot Kinetics Estimation From Distributed {VGRF} From Smart Insoles via {1D} Domain Transformation},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year = {2026},
  volume = {30},
  number = {3},
  pages = {2656--2667},
  month = mar,
  doi = {10.1109/JBHI.2025.3605296}
}
```
