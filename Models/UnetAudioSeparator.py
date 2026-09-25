import tensorflow as tf

import Models.InterpolationLayer
import Utils
from Utils import LeakyReLU
import numpy as np
import Models.OutputLayer

'''
Alias settings:
    Since tensorflow keep changing the relative path/location of some certain 
    functions, thus, we simply set some aliases to avoid annoying tensorflow 
    deprecation warning
'''
tf.variable_scope = tf.compat.v1.variable_scope
tf.image.resize_bilinear = tf.compat.v1.image.resize_bilinear
tf.variable_scope = tf.compat.v1.variable_scope

class UnetAudioSeparator(tf.keras.Model):
    '''
    U-Net separator network for singing voice separation.
    Uses valid convolutions, so it predicts for the centre part of the input - only certain input and output shapes are therefore possible (see getpadding function)
    '''

    def __init__(self, model_config, **kwargs):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        '''
        super().__init__(**kwargs)
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.num_increase_filters = model_config["num_increase_filters"]
        self.filter_size = model_config["filter_size"]
        self.merge_filter_size = model_config["merge_filter_size"]
        self.input_filter_size = model_config["input_filter_size"]
        self.output_filter_size = model_config["output_filter_size"]
        self.upsampling = model_config["upsampling"]
        self.output_type = model_config["output_type"]
        self.context = model_config["context"]
        self.padding = "valid" if model_config["context"] else "same"
        self.source_names = model_config["source_names"]
        self.num_channels = 1 if model_config["mono_downmix"] else 2
        self.output_activation = model_config["output_activation"]

        if self.output_activation == "tanh":
            self.out_activation_fn = tf.tanh
        elif self.output_activation == "linear":
            self.out_activation_fn = None
        else:
            raise NotImplementedError

        # Build layers
        self.down_convs = []
        for i in range(self.num_layers):
            self.down_convs.append(
                tf.keras.layers.Conv1D(
                    self.num_initial_filters + (self.num_increase_filters * i),
                    self.filter_size,
                    strides=1,
                    activation=LeakyReLU,
                    padding=self.padding,
                    name=f"downconv_{i}"
                )
            )

        self.bottleneck_conv = tf.keras.layers.Conv1D(
            self.num_initial_filters + (self.num_increase_filters * self.num_layers),
            self.filter_size,
            activation=LeakyReLU,
            padding=self.padding,
            name=f"downconv_{self.num_layers}"
        )

        self.up_interpolations = []
        self.up_convs = []
        for i in range(self.num_layers):
            if self.upsampling == 'learned':
                self.up_interpolations.append(
                    Models.InterpolationLayer.LearnedInterpolationLayer(
                        padding=self.padding,
                        level=str(i),
                        name=f"interp_{i}"
                    )
                )
            else:
                self.up_interpolations.append(None)

            self.up_convs.append(
                tf.keras.layers.Conv1D(
                    self.num_initial_filters + (self.num_increase_filters * (self.num_layers - i - 1)),
                    self.merge_filter_size,
                    activation=LeakyReLU,
                    padding=self.padding,
                    name=f"upconv_{i}"
                )
            )

        if self.output_type == "direct":
            self.out_layer = Models.OutputLayer.IndependentOutputLayer(
                self.source_names,
                self.num_channels,
                self.output_filter_size,
                self.padding,
                self.out_activation_fn,
                name="out_layer"
            )
        elif self.output_type == "difference":
            self.out_layer = Models.OutputLayer.DifferenceOutputLayer(
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
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape 
        :return: Input_shape, output_shape, where each is a list [batch_size, time_steps, channels]
        '''

        if self.context:
            rem = float(shape[1])
            rem = rem - self.output_filter_size + 1

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

            input_shape = np.concatenate([[shape[0]], [input_shape], [self.num_channels]])
            output_shape = np.concatenate([[shape[0]], [output_shape], [self.num_channels]])

            return input_shape, output_shape
        else:
            return [shape[0], shape[1], self.num_channels], [shape[0], shape[1], self.num_channels]

    def call(self, inputs, training=False, return_spectrogram=False):
        '''
        Forward pass of U-Net
        :param inputs: Input batch of mixtures, 3D tensor [batch_size, num_samples, num_channels]
        :return: Dict of source estimates. Each item is a 3D tensor [batch_size, num_out_samples, num_channels]
        '''
        enc_outputs = list()
        current_layer = inputs

        # Down-convolution
        for i in range(self.num_layers):
            current_layer = self.down_convs[i](current_layer)
            enc_outputs.append(current_layer)
            current_layer = current_layer[:, ::2, :]

        current_layer = self.bottleneck_conv(current_layer)

        # Upconvolution
        for i in range(self.num_layers):
            current_layer = tf.expand_dims(current_layer, axis=1)
            if self.upsampling == 'learned':
                current_layer = self.up_interpolations[i](current_layer)
            else:
                width = current_layer.shape[2]
                if self.context:
                    current_layer = tf.compat.v1.image.resize_bilinear(current_layer, [1, width * 2 - 1], align_corners=True)
                else:
                    current_layer = tf.compat.v1.image.resize_bilinear(current_layer, [1, width * 2])
            current_layer = tf.squeeze(current_layer, axis=1)

            current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer, match_feature_dim=False)
            current_layer = self.up_convs[i](current_layer)

        current_layer = Utils.crop_and_concat(inputs, current_layer, match_feature_dim=False)

        if self.output_type == "direct":
            return self.out_layer(current_layer)
        elif self.output_type == "difference":
            cropped_input = Utils.crop(inputs, current_layer.shape.as_list(), match_feature_dim=False)
            return self.out_layer(cropped_input, current_layer, training=training)
        else:
            raise NotImplementedError

    def get_output(self, input, training=False, return_spectrogram=False, reuse=True):
        '''
        Backward-compatible wrapper for TF1 graph calls
        '''
        return self(input, training=training, return_spectrogram=return_spectrogram)
