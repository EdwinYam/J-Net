import tensorflow as tf

import Utils

class IndependentOutputLayer(tf.keras.layers.Layer):
    def __init__(self, source_names, num_channels, filter_width, padding, activation, **kwargs):
        super().__init__(**kwargs)
        self.source_names = source_names
        self.num_channels = num_channels
        self.filter_width = filter_width
        self.layer_padding = padding
        self.activation = activation
        self.conv_layers = {
            name: tf.keras.layers.Conv1D(
                filters=num_channels,
                kernel_size=filter_width,
                activation=activation,
                padding=padding,
                name=f"conv1d_out_{name}"
            )
            for name in source_names
        }

    def call(self, featuremap):
        return {name: self.conv_layers[name](featuremap) for name in self.source_names}


class DifferenceOutputLayer(tf.keras.layers.Layer):
    def __init__(self, source_names, num_channels, filter_width, padding, activation, **kwargs):
        super().__init__(**kwargs)
        self.source_names = source_names
        self.num_channels = num_channels
        self.filter_width = filter_width
        self.layer_padding = padding
        self.activation = activation
        self.conv_layers = {
            name: tf.keras.layers.Conv1D(
                filters=num_channels,
                kernel_size=filter_width,
                activation=activation,
                padding=padding,
                name=f"conv1d_out_{name}"
            )
            for name in source_names[:-1]
        }

    def call(self, input_mix, featuremap, training=False):
        outputs = dict()
        sum_source = None
        for name in self.source_names[:-1]:
            out = self.conv_layers[name](featuremap)
            outputs[name] = out
            sum_source = out if sum_source is None else sum_source + out

        target_shape = sum_source.shape.as_list() if hasattr(sum_source, 'shape') else sum_source.get_shape().as_list()
        last_source = Utils.crop(input_mix, target_shape) - sum_source
        last_source = Utils.AudioClip(last_source, training)
        outputs[self.source_names[-1]] = last_source
        return outputs


def independent_outputs(featuremap, source_names, num_channels, filter_width, padding, activation):
    layer = IndependentOutputLayer(source_names, num_channels, filter_width, padding, activation)
    return layer(featuremap)

def difference_output(input_mix, featuremap, source_names, num_channels, filter_width, padding, activation, training=False):
    layer = DifferenceOutputLayer(source_names, num_channels, filter_width, padding, activation)
    return layer(input_mix, featuremap, training=training)

def _independent_outputs_(featuremap, num_sources, num_channels):
    outputs = list()
    for _ in range(num_sources):
        conv = tf.keras.layers.Conv1D(num_channels, 1, activation=tf.tanh, padding='valid')
        outputs.append(conv(featuremap))
    return outputs

def _difference_output_(input_mix, featuremap, num_sources, num_channels):
    outputs = list()
    last_source = input_mix
    for _ in range(num_sources - 1):
        conv = tf.keras.layers.Conv1D(num_channels, 1, activation=tf.tanh, padding='valid')
        out = conv(featuremap)
        outputs.append(out)
        last_source = last_source - out
    outputs.append(last_source)
    return outputs


