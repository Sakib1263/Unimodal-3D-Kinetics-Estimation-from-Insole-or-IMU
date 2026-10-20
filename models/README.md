# Reconstruction models

The model modules are shared research implementations for 1D dense sequence reconstruction. The maintained runner uses an explicit registry in [kinetics/modeling.py](../kinetics/modeling.py), validates full-resolution output shape, and selects the head named `out` during inference.

| Configuration name | Implemented method |
| --- | --- |
| `IMUNet` | `MLMRSNet.IMUNet()` |
| `MLMRSNet` | `MLMRSNet.MLMRSNet()` |
| `MLMRSNetV2` | `MLMRSNet.MLMRSNetV2()` |
| `LDNet` | `MLMRSNet.LDNet()` |
| `UNet`, `UNetPP`, `UNet3P`, `MultiResUNet` | Corresponding methods of `unet_variants.UNet` |
| `FPN` | `fpn.FPN.FPN()` |
| `PSPNet` | `pspnet.PSPNet.PSPNet()` |

The registry deliberately uses implementation names. **Do not equate `IMUNet` with Km2KnNet or Ke2KeNet without the component, equation, and checkpoint correspondence audit described in [PAPERS.md](../docs/PAPERS.md).** A passing shape/gradient test verifies software behavior, not article equivalence.

Deep-supervision targets are derived from actual output lengths, including full-length auxiliary heads. Model output order is not used to select the primary prediction. Each registered model is checked with deep supervision both enabled and disabled on small synthetic tensors.

[layers.py](layers.py) provides registered, serializable operational convolution/transposed-convolution, DropBlock1D, and channel-reduction layers. Operational layers validate polynomial order and create all trainable layers during initialization/build. DropBlock supports inference and training without removed Keras backend APIs, shares masks across channels correctly when requested, and handles a fully dropped mask without division by zero.

This directory contains the registered architecture families and reusable layers. Check historical checkpoints against the current model definitions before loading them.

Construct models through `build_model(config, num_channels)`, rather than calling architecture methods directly. Configuration validation checks dimensions, the wrapper exposes a dictionary of named heads, and the runner applies losses and checkpoint selection consistently. The primary head uses a linear activation so signed force/moment predictions are preserved. The total validation loss drives checkpoint selection: primary weight 1 and auxiliary weight 0.5 each.

The registry was checked with both deep-supervision settings on small gradient steps and at 1,024-sample resolution with model save/reload. Those checks cover implementation behavior; research-data performance and GPU behavior require separate evaluation. The layers module remains available for operational-layer/regularization experiments, but its existence does not imply that every registered architecture uses those layers.

Original author/license comments are retained. Cite the original methodological sources for reused architectures. The repository license does not replace scholarly attribution.
