"""A maintained 1D pyramid-pooling baseline, distinct from the papers' named models."""

import tensorflow as tf


class PSPNet:
    def __init__(
        self,
        length,
        model_depth,
        num_channel,
        model_width,
        kernel_size,
        problem_type="Regression",
        output_nums=1,
        ds=0,
        ae=0,
    ):
        if problem_type != "Regression" or ae:
            raise ValueError("This baseline supports dense regression without an autoencoder.")
        if min(length, model_depth, num_channel, model_width, kernel_size, output_nums) < 1:
            raise ValueError("Model dimensions must be positive.")
        if length % (2**model_depth):
            raise ValueError("Length must be divisible by 2**depth.")
        self.length, self.depth, self.channels = length, model_depth, num_channel
        self.width, self.kernel, self.outputs, self.ds = model_width, kernel_size, output_nums, ds

    def PSPNet(self):
        inputs = tf.keras.Input((self.length, self.channels))
        x = inputs
        for level in range(self.depth):
            x = tf.keras.layers.Conv1D(
                self.width * (2**level), self.kernel, padding="same", activation="relu"
            )(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
        branches = [x]
        length = int(x.shape[1])
        for bins in (1, 2, 4):
            if length < bins or length % bins:
                continue
            branch = tf.keras.layers.AveragePooling1D(length // bins)(x)
            branch = tf.keras.layers.Conv1D(self.width, 1, activation="relu")(branch)
            branches.append(tf.keras.layers.UpSampling1D(length // bins)(branch))
        x = tf.keras.layers.Concatenate()(branches)
        outputs = []
        for level in reversed(range(self.depth)):
            if self.ds:
                outputs.append(tf.keras.layers.Conv1D(self.outputs, 1, name=f"level{level + 1}")(x))
            x = tf.keras.layers.UpSampling1D(2)(x)
            x = tf.keras.layers.Conv1D(self.width, self.kernel, padding="same", activation="relu")(x)
        outputs.append(tf.keras.layers.Conv1D(self.outputs, 1, name="out")(x))
        return tf.keras.Model(inputs, outputs if self.ds else outputs[0])
