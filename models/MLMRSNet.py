"""Multiscale 1D regression architectures used by the explicit model registry.

Construct through kinetics.modeling.build_model so output names and dimensions
are checked consistently. Paper equivalence is discussed in docs/PAPERS.md.
"""

# Author: Sakib Mahmud
# MIT License

import tensorflow as tf


def Conv_Block(inputs, model_width, kernel, multiplier, use_batchnorm=True):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width * multiplier, kernel, padding="same")(inputs)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x


def trans_conv1D(inputs, model_width, multiplier, kernel_size, level):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width * multiplier, kernel_size, strides=level, padding="same")(
        inputs
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x


def Concat_Block(input1, *argv):
    # Concatenation Block from the KERAS Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = tf.keras.layers.concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs, size=2):
    # 1D UpSampling Block
    up = tf.keras.layers.UpSampling1D(size=size)(inputs)

    return up


def Feature_Extraction_Block(inputs, model_width, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    shape = inputs.shape
    latent = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(feature_number, name="features")(latent)
    latent = tf.keras.layers.Dense(model_width * shape[1])(latent)
    latent = tf.keras.layers.Reshape((shape[1], model_width))(latent)

    return latent


def mix_pool_layer(pool_size):
    """Combine max and mean pooling, then learn a channel projection."""

    def apply(x):
        return tf.keras.layers.Conv1D(int(x.shape[-1]), 1)(
            tf.keras.layers.Add()(
                [
                    tf.keras.layers.MaxPooling1D(3, strides=pool_size, padding="same")(x),
                    tf.keras.layers.AveragePooling1D(3, strides=pool_size, padding="same")(x),
                ]
            )
        )

    return apply


def MSP_Unit(inputs, level, conv_filters, multiplier, pooling_type="mix"):
    """Pool one temporal scope, restore its length through two paths, then fuse."""
    pool_size = up_size = level
    x = []
    if pooling_type == "avg":
        x = tf.keras.layers.AveragePooling1D(3, strides=pool_size, padding="same")(inputs)
    elif pooling_type == "max":
        x = tf.keras.layers.MaxPooling1D(3, strides=pool_size, padding="same")(inputs)
    elif pooling_type == "mix":
        x = mix_pool_layer(pool_size)(inputs)
    x = Conv_Block(x, conv_filters, 1, multiplier, use_batchnorm=True)
    x1 = trans_conv1D(x, conv_filters, multiplier, 4, up_size)
    x2 = tf.keras.layers.UpSampling1D(size=up_size)(x)
    x = tf.keras.layers.concatenate([x1, x2], axis=-1)
    x = Conv_Block(x, conv_filters, 1, 1, use_batchnorm=False)

    return x


def MRP_Block(inputs, conv_filters, multiplier, pooling_type="mix", cardinality=5):
    """Fuse the input with multiple pooled scopes and parallel 3/5/7 convolutions."""
    x = []
    if cardinality == 0:
        x = inputs
    else:
        for ii in range(0, cardinality):
            x_temp = MSP_Unit(inputs, 2**ii, conv_filters, multiplier, pooling_type=pooling_type)
            if ii == 0:
                x = tf.keras.layers.concatenate([inputs, x_temp], axis=-1)
            else:
                x = tf.keras.layers.concatenate([x, x_temp], axis=-1)
    x3 = Conv_Block(x, conv_filters, 3, multiplier, use_batchnorm=False)
    x5 = Conv_Block(x, conv_filters, 5, multiplier, use_batchnorm=False)
    x7 = Conv_Block(x, conv_filters, 7, multiplier, use_batchnorm=False)
    out = Conv_Block(
        tf.keras.layers.concatenate([x3, x5, x7], axis=-1), conv_filters, 1, multiplier, use_batchnorm=True
    )
    return out


class MLMRSNet:
    """Build registered multiscale variants; use the shared regression wrapper."""

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
        cardinality=5,
        pooling_type="avg",
        feature_number=1024,
        is_transconv=True,
    ):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # kernel_size: Kernel or Filter Size of the Convolutional Layers
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # cardinality: Number of MSP layers or units per MRP Block
        # pooling type: 'average', 'max' or 'mix'
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        # is_transconv: Transposed Convolution or Interpolation based UpSampling
        if min(length, model_depth, num_channel, model_width, kernel_size, output_nums) < 1:
            raise ValueError("Model dimensions must be positive.")
        if length % (2**model_depth):
            raise ValueError("Signal length must be divisible by 2**model_depth.")
        if cardinality < 0 or pooling_type not in {"avg", "max", "mix"}:
            raise ValueError("Invalid scope cardinality or pooling type.")
        self.length = length
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.cardinality = cardinality
        self.pooling_type = pooling_type
        self.feature_number = feature_number
        self.is_transconv = is_transconv

    def MLMRSNet(self):
        """Use fixed-cardinality multiscale blocks in the encoder and decoder."""
        if (
            self.length == 0
            or self.model_depth == 0
            or self.model_width == 0
            or self.num_channel == 0
            or self.kernel_size == 0
        ):
            raise ValueError("Model dimensions must be positive.")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = MRP_Block(
                pool,
                self.model_width,
                2 ** (i - 1),
                pooling_type=self.pooling_type,
                cardinality=self.cardinality,
            )
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = MRP_Block(
            pool,
            self.model_width,
            2**self.model_depth,
            pooling_type=self.pooling_type,
            cardinality=self.cardinality,
        )

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            layer_num = self.model_depth - j
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f"level{layer_num}")(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = Concat_Block(
                    trans_conv1D(deconv, self.model_width, 2 ** (layer_num - 1), 1, 2),
                    convs_list[layer_num - 1],
                )
            elif not self.is_transconv:
                deconv = Concat_Block(upConv_Block(deconv), convs_list[layer_num - 1])
            deconv = MRP_Block(
                deconv,
                self.model_width,
                2 ** (layer_num - 1),
                pooling_type=self.pooling_type,
                cardinality=self.cardinality,
            )

        # Output
        outputs = []
        if self.problem_type == "Classification":
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation="softmax", name="out")(deconv)
        elif self.problem_type == "Regression":
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation="linear", name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def MLMRSNetV2(self):
        """Use dense cross-level connections and depth-dependent scope counts."""
        if (
            self.length == 0
            or self.model_depth == 0
            or self.model_width == 0
            or self.num_channel == 0
            or self.kernel_size == 0
        ):
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(0, self.model_depth):
            if i > 0:
                for k in range(1, i):
                    conv = convs["conv%s" % (k - 1)]
                    conv = tf.keras.layers.MaxPooling1D(pool_size=(2 ** (i - k)))(conv)
                    pool = tf.keras.layers.concatenate([pool, conv], axis=-1)
            conv = MRP_Block(
                pool,
                self.model_width,
                2**i,
                pooling_type=self.pooling_type,
                cardinality=self.model_depth - i + 1,
            )
            convs["conv%s" % (i - 1)] = conv
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = MRP_Block(
            pool,
            self.model_width,
            2**self.model_depth,
            pooling_type=self.pooling_type,
            cardinality=1,
        )
        convs["conv%s" % self.model_depth] = conv

        # Decoding
        deconv = conv
        deconvs = {}
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connections_all = convs_list[self.model_depth - j - 1]
            for k in range(0, self.model_depth - j - 1):
                skip_connection = convs_list[k]
                skip_connection = tf.keras.layers.MaxPooling1D(
                    pool_size=(2 ** ((self.model_depth - j) - k - 1))
                )(skip_connection)
                skip_connections_all = tf.keras.layers.concatenate(
                    [skip_connections_all, skip_connection], axis=-1
                )
            deconv_tot = upConv_Block(deconv, size=2**1)
            deconv_tot = tf.keras.layers.Activation("sigmoid")(deconv_tot)
            deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
            if j > 0:
                for m in range(0, j):
                    deconv = deconvs["deconv%s" % m]
                    deconv = upConv_Block(deconv, size=(2 ** (j - m)))
                    deconv = tf.keras.layers.Activation("sigmoid")(deconv)
                    deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
            deconv = MRP_Block(
                deconv_tot,
                self.model_width,
                self.model_depth + 1,
                pooling_type=self.pooling_type,
                cardinality=j + 1,
            )
            deconvs["deconv%s" % j] = deconv
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, 2, name=f"level{self.model_depth - j}")(deconv)
                levels.append(level)

        # Output
        outputs = []
        if self.problem_type == "Classification":
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation="softmax", name="out")(deconv)
        elif self.problem_type == "Regression":
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation="linear", name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def LDNet(self):
        """Use a nested decoder with multiscale encoder features."""
        if (
            self.length == 0
            or self.model_depth == 0
            or self.model_width == 0
            or self.num_channel == 0
            or self.kernel_size == 0
        ):
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = MRP_Block(
                pool,
                self.model_width,
                2 ** (i - 1),
                pooling_type=self.pooling_type,
                cardinality=self.model_depth - i + 1,
            )
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = MRP_Block(
            pool,
            self.model_width,
            2 ** (i - 1),
            pooling_type=self.pooling_type,
            cardinality=0,
        )
        convs["conv%s" % (self.model_depth + 1)] = conv

        # Decoding
        convs_list = list(convs.values())
        if self.D_S == 1:
            level = tf.keras.layers.Conv1D(1, 1, name=f"level{self.model_depth}")(convs_list[0])
            levels.append(level)

        deconv = []
        deconvs = {}
        deconvs_skip = {}

        for i in range(1, (self.model_depth + 1)):
            for j in range(0, (self.model_depth - i + 1)):
                if i == 1:
                    skip_connection = convs_list[j]
                    if self.is_transconv:
                        deconv = trans_conv1D(convs_list[j + 1], self.model_width, 2**j, 2, 2)
                    elif not self.is_transconv:
                        deconv = upConv_Block(convs_list[j + 1])
                    deconv = Concat_Block(deconv, skip_connection)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2**j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2**j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                    if (i + j) == (self.model_depth):
                        deconvs_skip["deconv_skip%s" % i] = deconv
                elif i > 1:
                    deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                    for k in range(2, i):
                        deconv_temp = deconvs["deconv%s%s" % (j, k)]
                        deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                    skip_connection = convs_list[j]
                    if self.is_transconv:
                        deconv = trans_conv1D(
                            deconvs["deconv%s%s" % ((j + 1), (i - 1))], self.model_width, 2**j, 2, 2
                        )
                    elif not self.is_transconv:
                        deconv = upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))])
                    deconv = Concat_Block(deconv, deconv_tot, skip_connection)
                    if (i + j) == (self.model_depth) and (j != (self.model_depth - 1)):
                        for m in range(1, i - 1):
                            temp_deconv = upConv_Block(deconvs_skip["deconv_skip%s" % m], size=(2 ** (i - m)))
                            deconv = tf.keras.layers.concatenate([deconv, temp_deconv], axis=-1)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2**j)
                    deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2**j)
                    deconvs["deconv%s%s" % (j, i)] = deconv
                    if (i + j) == (self.model_depth):
                        deconvs_skip["deconv_skip%s" % i] = deconv
                if (self.D_S == 1) and (j == 0) and (i < self.model_depth):
                    level = tf.keras.layers.Conv1D(1, 1, name=f"level{self.model_depth - i}")(
                        deconvs["deconv%s%s" % (j, i)]
                    )
                    levels.append(level)

        deconv = deconvs["deconv%s%s" % (0, self.model_depth)]

        # Output
        outputs = []
        if self.problem_type == "Classification":
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation="softmax", name="out")(deconv)
        elif self.problem_type == "Regression":
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation="linear", name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def IMUNet(self):
        """Use multiscale encoding with full-scale decoder feature aggregation."""
        if (
            self.length == 0
            or self.model_depth == 0
            or self.model_width == 0
            or self.num_channel == 0
            or self.kernel_size == 0
        ):
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = MRP_Block(
                pool,
                self.model_width,
                2 ** (i - 1),
                pooling_type=self.pooling_type,
                cardinality=self.model_depth - i + 1,
            )
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = MRP_Block(
            pool,
            self.model_width,
            2**self.model_depth,
            pooling_type=self.pooling_type,
            cardinality=0,
        )

        # For Deep Supervision
        if self.problem_type == "Classification":
            level = tf.keras.layers.Conv1D(1, 1, activation="sigmoid", name=f"level{self.model_depth}")(conv)
        elif self.problem_type == "Regression":
            level = tf.keras.layers.Conv1D(1, 1, activation="linear", name=f"level{self.model_depth}")(conv)
        levels.append(level)

        # Decoding
        deconv = conv
        deconvs = {}
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connections_all = convs_list[self.model_depth - j - 1]
            for k in range(0, self.model_depth - j - 1):
                skip_connection = convs_list[k]
                skip_connection = tf.keras.layers.MaxPooling1D(
                    pool_size=(2 ** ((self.model_depth - j) - k - 1))
                )(skip_connection)
                skip_connection = Conv_Block(skip_connection, self.model_width, self.kernel_size, 2**0)
                skip_connections_all = tf.keras.layers.concatenate(
                    [skip_connections_all, skip_connection], axis=-1
                )
            deconv_tot = MRP_Block(
                deconv,
                self.model_width,
                2**0,
                pooling_type=self.pooling_type,
                cardinality=0,
            )
            deconv_tot = upConv_Block(deconv_tot, size=2**1)
            deconv_tot = tf.keras.layers.Activation("sigmoid")(deconv_tot)
            deconv_tot = tf.keras.layers.concatenate([skip_connections_all, deconv_tot], axis=-1)
            if j > 0:
                for m in range(0, j):
                    deconv = upConv_Block(deconvs["deconv%s" % m], size=(2 ** (j - m)))
                    deconv = tf.keras.layers.Activation("sigmoid")(deconv)
                    deconv_tot = tf.keras.layers.concatenate([deconv_tot, deconv], axis=-1)
            deconv = MRP_Block(
                deconv_tot,
                self.model_width,
                2 ** (self.model_depth - j),
                pooling_type=self.pooling_type,
                cardinality=0,
            )
            deconvs["deconv%s" % j] = deconv
            if self.D_S == 1 and (j < (self.model_depth - 1)):
                # For Deep Supervision
                if self.problem_type == "Classification":
                    level = tf.keras.layers.Conv1D(
                        1, 1, activation="sigmoid", name=f"level{self.model_depth - j - 1}"
                    )(deconv)
                elif self.problem_type == "Regression":
                    level = tf.keras.layers.Conv1D(
                        1, 1, activation="linear", name=f"level{self.model_depth - j - 1}"
                    )(deconv)
                levels.append(level)
        # Output
        outputs = []
        if self.problem_type == "Classification":
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation="sigmoid", name="out")(deconv)
        elif self.problem_type == "Regression":
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation="linear", name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.reverse()
            levels.append(outputs)
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model
