"""Serializable operational and regularization layers shared by model variants."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kinetics")
class Oper1D(tf.keras.layers.Layer):
    convolution = tf.keras.layers.Conv1D

    def __init__(self, filters, kernel_size, padding="same", strides=1, activation=None, q=1, **kwargs):
        super().__init__(**kwargs)
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError("q must be a positive integer.")
        self.filters, self.kernel_size = filters, kernel_size
        self.padding, self.strides, self.q = padding, strides, q
        self.activation = tf.keras.activations.get(activation)
        self.all_layers = [
            self.convolution(
                filters, kernel_size, strides=strides, padding=padding, name=f"polynomial_{i + 1}"
            )
            for i in range(q)
        ]

    def build(self, input_shape):
        for layer in self.all_layers:
            layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        terms = [
            layer(inputs if i == 0 else tf.pow(inputs, i + 1)) for i, layer in enumerate(self.all_layers)
        ]
        return self.activation(tf.add_n(terms))

    def compute_output_shape(self, input_shape):
        return self.all_layers[0].compute_output_shape(input_shape)

    def get_config(self):
        return {
            **super().get_config(),
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "strides": self.strides,
            "q": self.q,
            "activation": tf.keras.activations.serialize(self.activation),
        }


@tf.keras.utils.register_keras_serializable(package="kinetics")
class Oper1DTranspose(Oper1D):
    convolution = tf.keras.layers.Conv1DTranspose


@tf.keras.utils.register_keras_serializable(package="kinetics")
class ChannelReduction(tf.keras.layers.Layer):
    def __init__(self, reduction="mean", **kwargs):
        super().__init__(**kwargs)
        if reduction not in {"mean", "max"}:
            raise ValueError("reduction must be mean or max.")
        self.reduction = reduction

    def call(self, inputs):
        operation = tf.reduce_mean if self.reduction == "mean" else tf.reduce_max
        return operation(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 1)

    def get_config(self):
        return {**super().get_config(), "reduction": self.reduction}


@tf.keras.utils.register_keras_serializable(package="kinetics")
class DropBlock1D(tf.keras.layers.Layer):
    def __init__(self, block_size, keep_prob, sync_channels=False, **kwargs):
        super().__init__(**kwargs)
        if (
            type(block_size) is not int
            or block_size < 1
            or type(keep_prob) not in (int, float)
            or not 0 < keep_prob <= 1
            or type(sync_channels) is not bool
        ):
            raise ValueError("block_size must be positive and 0 < keep_prob <= 1.")
        self.block_size, self.keep_prob, self.sync_channels = block_size, keep_prob, sync_channels

    def build(self, input_shape):
        if input_shape[1] is not None and input_shape[1] < self.block_size:
            raise ValueError("DropBlock block_size cannot exceed the sequence length.")
        super().build(input_shape)

    def call(self, inputs, training=None):
        def dropped():
            shape = tf.shape(inputs)
            length = tf.cast(shape[1], inputs.dtype)
            gamma = (1 - self.keep_prob) / self.block_size * length / (length - self.block_size + 1)
            positions = tf.range(shape[1])
            valid = tf.logical_and(
                positions >= (self.block_size - 1) // 2, positions < shape[1] - self.block_size // 2
            )
            mask_shape = tf.stack([shape[0], shape[1], 1 if self.sync_channels else shape[2]])
            seed = tf.cast(tf.random.uniform(mask_shape, dtype=inputs.dtype) < gamma, inputs.dtype)
            seed *= tf.cast(valid[None, :, None], inputs.dtype)
            mask = 1 - tf.nn.max_pool1d(seed, self.block_size, strides=1, padding="SAME")
            mask = tf.broadcast_to(mask, shape)
            scale = tf.math.divide_no_nan(tf.cast(tf.size(inputs), inputs.dtype), tf.reduce_sum(mask))
            return inputs * mask * scale

        if training is None or training is False or self.keep_prob == 1:
            return inputs
        if training is True:
            return dropped()
        return tf.cond(tf.cast(training, tf.bool), dropped, lambda: tf.identity(inputs))

    def get_config(self):
        return {
            **super().get_config(),
            "block_size": self.block_size,
            "keep_prob": self.keep_prob,
            "sync_channels": self.sync_channels,
        }
