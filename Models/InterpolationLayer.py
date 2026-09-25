import tensorflow as tf


class LearnedInterpolationLayer(tf.keras.layers.Layer):
    '''
    Implements a trainable upsampling layer by interpolation by a factor of two, from N samples to N*2 - 1.
    Interpolation of intermediate feature vectors v_1 and v_2 (of dimensionality F) is performed by
     w \\cdot v_1 + (1-w) \\cdot v_2, where \\cdot is point-wise multiplication, and w an F-dimensional weight vector constrained to [0,1]
    :param padding: "valid" or "same"
    :param level: identifier for layer weights
    '''
    def __init__(self, padding="valid", level="0", **kwargs):
        super().__init__(**kwargs)
        assert padding.lower() in ("valid", "same")
        self.layer_padding = padding.upper()
        self.level = str(level)
        self.interp_weights = None

    def build(self, input_shape):
        features = input_shape[-1]
        self.interp_weights = self.add_weight(
            name="interp_" + self.level,
            shape=[features],
            dtype=tf.float32,
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: [batch_size, 1, width, features]
        weights_scaled = tf.nn.sigmoid(self.interp_weights)
        counter_weights = 1.0 - weights_scaled
        diag_w = tf.expand_dims(tf.linalg.diag(weights_scaled), axis=0)
        diag_cw = tf.expand_dims(tf.linalg.diag(counter_weights), axis=0)
        conv_weights = tf.expand_dims(tf.concat([diag_w, diag_cw], axis=0), axis=0)

        intermediate_vals = tf.nn.conv2d(inputs, conv_weights, strides=[1, 1, 1, 1], padding=self.layer_padding)

        intermediate_vals = tf.transpose(intermediate_vals, [2, 0, 1, 3])
        out = tf.transpose(inputs, [2, 0, 1, 3])
        num_entries = out.shape[0] if out.shape[0] is not None else tf.shape(out)[0]
        out = tf.concat([out, intermediate_vals], axis=0)

        num_outputs = (2 * num_entries - 1) if self.layer_padding == "VALID" else 2 * num_entries
        indices = list()
        for idx in range(int(num_outputs)):
            if idx % 2 == 0:
                indices.append(idx // 2)
            else:
                indices.append(int(num_entries) + idx // 2)
        out = tf.gather(out, indices)
        return tf.transpose(out, [1, 2, 0, 3])


def learned_interpolation_layer(input, padding, level):
    '''
    Functional wrapper for backward compatibility with TF1 graph calls.
    '''
    layer = LearnedInterpolationLayer(padding=padding, level=level)
    return layer(input)