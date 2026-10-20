"""Explicit model dispatch and targets derived from actual output tensors."""

import importlib
import numpy as np

MODEL_REGISTRY = {
    "IMUNet": ("MLMRSNet", "MLMRSNet", "IMUNet"),
    "MLMRSNet": ("MLMRSNet", "MLMRSNet", "MLMRSNet"),
    "MLMRSNetV2": ("MLMRSNet", "MLMRSNet", "MLMRSNetV2"),
    "LDNet": ("MLMRSNet", "MLMRSNet", "LDNet"),
    "UNet": ("unet_variants", "UNet", "UNet"),
    "UNetPP": ("unet_variants", "UNet", "UNetPP"),
    "UNet3P": ("unet_variants", "UNet", "UNet3P"),
    "MultiResUNet": ("unet_variants", "UNet", "MultiResUNet"),
    "FPN": ("fpn", "FPN", "FPN"),
    "PSPNet": ("pspnet", "PSPNet", "PSPNet"),
}


def build_model(config, num_channels):
    import tensorflow as tf

    if type(num_channels) is not int or num_channels < 1:
        raise ValueError("num_channels must be a positive integer.")
    if config.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {config.model}. Choose {sorted(MODEL_REGISTRY)}.")
    module_name, class_name, method_name = MODEL_REGISTRY[config.model]
    cls = getattr(importlib.import_module(f"models.{module_name}"), class_name)
    kwargs = dict(problem_type="Regression", output_nums=1, ds=int(config.deep_supervision), ae=0)
    if module_name == "MLMRSNet":
        kwargs.update(cardinality=config.cardinality, pooling_type=config.pooling)
        # Fixed-cardinality scopes must divide every encoder/bottleneck resolution.
        if config.model == "MLMRSNet" and config.cardinality:
            multiple = 2 ** (config.depth + config.cardinality - 1)
            if config.window_length % multiple:
                raise ValueError(f"MLMRSNet scope alignment requires window_length divisible by {multiple}.")
    instance = cls(
        config.window_length, config.depth, num_channels, config.width, config.kernel_size, **kwargs
    )
    original = getattr(instance, method_name)()
    if len(set(original.output_names)) != len(original.output_names):
        raise ValueError("Architecture output names must be unique.")
    outputs = dict(zip(original.output_names, original.outputs, strict=True))
    if "out" not in outputs or outputs["out"].shape[1:] != (config.window_length, 1):
        raise ValueError("Architecture must expose an out head with (window_length,1) samples.")
    # Named outputs remove architecture-dependent output-order assumptions.
    return tf.keras.Model(inputs=original.inputs[0], outputs=outputs, name=f"{config.model}_reconstruction")


def prepare_targets(model, y):
    y = np.asarray(y)
    if y.ndim != 3 or not y.size or not np.isfinite(y).all():
        raise ValueError("Targets must be a finite nonempty (window,time,channel) array.")
    result = {}
    for name, tensor in zip(model.output_names, model.outputs, strict=True):
        length, channels = tensor.shape[1:]
        if channels != y.shape[2] or length is None or length < 1 or y.shape[1] % length:
            raise ValueError(f"Target cannot be aligned to output {name}: {tensor.shape}.")
        factor = y.shape[1] // length
        result[name] = y.reshape(len(y), length, factor, y.shape[2]).mean(axis=2).astype(np.float32)
    return result


def compile_model(model, config):
    import tensorflow as tf

    losses = {
        "mae": tf.keras.losses.MeanAbsoluteError,
        "mse": tf.keras.losses.MeanSquaredError,
        "huber": tf.keras.losses.Huber,
    }
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={name: losses[config.loss]() for name in model.output_names},
        loss_weights={name: 1.0 if name == "out" else 0.5 for name in model.output_names},
        metrics={name: [tf.keras.metrics.MeanSquaredError()] for name in model.output_names},
        jit_compile=False,
    )
    return model
