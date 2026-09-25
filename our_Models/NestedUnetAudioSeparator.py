import tensorflow as tf

import Models.InterpolationLayer
import Utils
from Utils import LeakyReLU
import numpy as np
import Models.OutputLayer

'''Alias Settings'''
tf.variable_scope = tf.compat.v1.variable_scope
tf.image.resize_bilinear = tf.compat.v1.image.resize_bilinear


class NestedUnetAudioSeparator(tf.keras.Model):
    '''
    U-Net separator network for audio separation or speech enhancement.
    Uses valid convolutions, so it predicts for the centre part of the input - 
    only certain input and output shapes are therefore possible (see getpadding 
    function)
    '''

    def __init__(self, model_config, **kwargs):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        :param context: Determine whether to discard those padding for 
                        consistent size
        :param upsampling: Determine whether upsampling is learned
        :param deep_supervised: Detemine whether to supervise over outputs of 
                                all nested Unet
        '''
        super().__init__(**kwargs)
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"] # 24
        self.num_increase_filters = model_config["num_increase_filters"]
        self.filter_size = model_config["filter_size"] # 15
        self.merge_filter_size = model_config["merge_filter_size"] # 5
        self.input_filter_size = model_config["input_filter_size"] # 15, middle
        self.output_filter_size = model_config["output_filter_size"] # 1, last
        self.upsampling = model_config["upsampling"]
        self.output_type = model_config["output_type"]
        self.context = model_config["context"]
        self.padding = "valid" if model_config["context"] else "same"
        self.source_names = model_config["source_names"]
        self.num_sources = len(self.source_names)
        self.num_channels = 1 if model_config["mono_downmix"] else 2
        self.output_activation = model_config["output_activation"] # tf.tanh
        self.deep_supervised = model_config["deep_supervised"]

        if self.output_activation == "tanh":
            self.out_activation_fn = tf.tanh
        elif self.output_activation == "linear":
            self.out_activation_fn = None
        else:
            raise NotImplementedError

        # Down-convolutions
        self.down_convs = []
        for i in range(self.num_layers):
            filter_num = self.num_initial_filters + self.num_increase_filters * i
            self.down_convs.append(
                tf.keras.layers.Conv1D(
                    filter_num,
                    self.filter_size,
                    strides=1,
                    activation=LeakyReLU,
                    padding=self.padding,
                    name=f"downconv_{i}"
                )
            )

        filter_num = self.num_initial_filters + self.num_increase_filters * (self.num_layers - 1)
        self.bottleneck_conv = tf.keras.layers.Conv1D(
            filter_num,
            self.filter_size,
            activation=LeakyReLU,
            padding=self.padding,
            name=f"downconv_{self.num_layers}"
        )

        # Nested upconvs and interpolation layers
        self.up_conv_dict = {}
        self.up_interp_dict = {}
        for i in range(self.num_layers):
            for j in range(1, self.num_layers + 1 - i):
                key = f"{i}_{j}"
                if j != self.num_layers:
                    f_num = self.num_initial_filters + self.num_increase_filters * (j - 1)
                    self.up_conv_dict[key] = tf.keras.layers.Conv1D(
                        f_num,
                        self.merge_filter_size,
                        activation=LeakyReLU,
                        padding=self.padding,
                        name=f"upconv_{key}"
                    )
                if self.upsampling == 'learned':
                    self.up_interp_dict[key] = Models.InterpolationLayer.LearnedInterpolationLayer(
                        padding=self.padding,
                        level=key,
                        name=f"interp_{key}"
                    )

        # Final output convs for nested stages
        self.final_convs = {}
        for i in range(1, self.num_layers + 1):
            if not self.deep_supervised and i != self.num_layers:
                continue
            self.final_convs[str(i)] = tf.keras.layers.Conv1D(
                self.num_initial_filters + self.num_increase_filters * (self.num_layers - 1),
                self.merge_filter_size,
                activation=LeakyReLU,
                padding=self.padding,
                name=f"final_conv_{i}"
            )

        # Output layers
        if self.output_type == "direct":
            if self.deep_supervised:
                self.out_layers = [
                    Models.OutputLayer.IndependentOutputLayer(
                        self.source_names,
                        self.num_channels,
                        self.output_filter_size,
                        self.padding,
                        self.out_activation_fn,
                        name=f"out_layer_{i}"
                    )
                    for i in range(self.num_layers)
                ]
            else:
                self.out_layers = Models.OutputLayer.IndependentOutputLayer(
                    self.source_names,
                    self.num_channels,
                    self.output_filter_size,
                    self.padding,
                    self.out_activation_fn,
                    name="out_layer"
                )
        elif self.output_type == "difference":
            if self.deep_supervised:
                self.out_layers = [
                    Models.OutputLayer.DifferenceOutputLayer(
                        self.source_names,
                        self.num_channels,
                        self.output_filter_size,
                        self.padding,
                        self.out_activation_fn,
                        name=f"out_layer_{i}"
                    )
                    for i in range(self.num_layers)
                ]
            else:
                self.out_layers = Models.OutputLayer.DifferenceOutputLayer(
                    self.source_names,
                    self.num_channels,
                    self.output_filter_size,
                    self.padding,
                    self.out_activation_fn,
                    name="out_layer"
                )
        else:
            raise NotImplementedError

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input 
        and output, so that the Unet works and has the given shape as output 
        shape
        :param shape: Desired output shape 
        :return: Input_shape, output_shape, where each is a list 
                 [batch_size, time_steps, channels]
        '''

        if self.context:
            rem = float(shape[1])
            rem = rem + self.output_filter_size - 1

            for i in range(self.num_layers):
                rem = rem + self.merge_filter_size - 1
                rem = (rem + 1.) / 2.

            x = np.asarray(np.ceil(rem), dtype=np.int64)
            assert(x >= 2)

            output_shape = x
            input_shape = x
            input_shape = input_shape + self.filter_size - 1

            for i in range(self.num_layers):
                output_shape = 2*output_shape - 1
                output_shape = output_shape - self.merge_filter_size + 1

                input_shape = 2*input_shape - 1
                if i < self.num_layers - 1:
                    input_shape = input_shape + self.filter_size - 1
                else:
                    input_shape = input_shape + self.input_filter_size - 1

            output_shape = output_shape - self.output_filter_size + 1

            input_shape = np.asarray([shape[0], input_shape, self.num_channels])
            output_shape = np.asarray([shape[0], output_shape, self.num_channels])
            return input_shape, output_shape
        else:
            input_shape = np.asarray([shape[0], shape[1], self.num_channels])
            output_shape = input_shape
            return input_shape, output_shape

    def call(self, inputs, training=False, return_spectrogram=False):
        '''
        Forward pass of Nested U-Net (Wave-U-Net++ / J-Net)
        :param inputs: Input batch of mixtures, 3D tensor [batch_size, num_samples, num_channels]
        :return: If deep_supervised: list of dicts. Otherwise: dict of source estimates.
        '''
        enc_outputs = list()
        current_layer = inputs

        # Down-convolution
        for i in range(self.num_layers):
            current_layer = self.down_convs[i](current_layer)
            enc_outputs.append(current_layer)
            current_layer = current_layer[:, ::2, :]

        current_layer = self.bottleneck_conv(current_layer)
        enc_outputs.append(current_layer)
        assert(len(enc_outputs) == self.num_layers + 1)

        all_enc_outputs = list()
        all_enc_outputs.append(enc_outputs)

        # Upconvolution
        for i in range(self.num_layers):
            sub_enc_outputs = list()
            for j in range(1, self.num_layers + 1 - i):
                key = f"{i}_{j}"
                current_layer_up = all_enc_outputs[i][j]
                if j != self.num_layers:
                    for k in range(i):
                        current_layer_up = Utils.crop_and_concat(
                            all_enc_outputs[k+1][j],
                            current_layer_up,
                            match_feature_dim=False
                        )
                    current_layer_up = self.up_conv_dict[key](current_layer_up)

                # UPSAMPLING
                current_layer_up = tf.expand_dims(current_layer_up, axis=1)
                if self.upsampling == 'learned':
                    current_layer_up = self.up_interp_dict[key](current_layer_up)
                else:
                    width = current_layer_up.shape[2]
                    if self.context:
                        current_shape = [1, width * 2 - 1]
                        current_layer_up = tf.compat.v1.image.resize_bilinear(
                            current_layer_up,
                            current_shape,
                            align_corners=True
                        )
                    else:
                        current_shape = [1, width * 2]
                        current_layer_up = tf.compat.v1.image.resize_bilinear(
                            current_layer_up,
                            current_shape
                        )
                current_layer_up = tf.squeeze(current_layer_up, axis=1)
                sub_enc_outputs.append(current_layer_up)

            all_enc_outputs.append(sub_enc_outputs)

        # Reconnect/concatenate the most shallow layer together to form the input of last conv
        final_outputs = list()
        for i in range(1, self.num_layers + 1):
            if not self.deep_supervised and i != self.num_layers:
                continue
            current_layer = all_enc_outputs[i][0]

            for j in range(i):
                current_layer = Utils.crop_and_concat(
                    all_enc_outputs[j][0],
                    current_layer,
                    match_feature_dim=False
                )

            current_layer = self.final_convs[str(i)](current_layer)
            final_outputs.append(current_layer)

        for i in range(self.num_layers - 1):
            if not self.deep_supervised:
                continue
            final_outputs[i] = Utils.crop(
                final_outputs[i],
                final_outputs[-1].shape.as_list(),
                match_feature_dim=False
            )

        if self.output_type == "direct":
            if self.deep_supervised:
                return [self.out_layers[i](final_outputs[i]) for i in range(self.num_layers)]
            else:
                return self.out_layers(final_outputs[-1])
        elif self.output_type == "difference":
            cropped_input = Utils.crop(
                inputs,
                final_outputs[-1].shape.as_list(),
                match_feature_dim=False
            )
            if self.deep_supervised:
                cropped_inputs = [cropped_input] * self.num_layers
                return [
                    self.out_layers[i](cropped_inputs[i], final_outputs[i], training=training)
                    for i in range(self.num_layers)
                ]
            else:
                return self.out_layers(cropped_input, final_outputs[-1], training=training)
        else:
            raise NotImplementedError

    def get_output(self, input, training=False, return_spectrogram=False, reuse=True):
        '''
        Backward-compatible wrapper for TF1 graph calls
        '''
        return self(input, training=training, return_spectrogram=return_spectrogram)

