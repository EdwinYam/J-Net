import tensorflow as tf
from Utils import LeakyReLU
import functools

class UnetSpectrogramSeparator(tf.keras.Model):
    '''
    U-Net separator network for singing voice separation.
    Takes in the mixture magnitude spectrogram and return estimates of the accompaniment and voice magnitude spectrograms.
    Uses "same" convolutions like in original paper
    '''

    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.mono = model_config["mono_downmix"]
        self.source_names = model_config["source_names"]

        assert len(self.source_names) == 2
        assert self.mono

        # Spectrogram settings
        self.frame_len = 1024
        self.hop = 768

        # Create sub-layers per source
        self.down_convs = {name: [] for name in self.source_names}
        self.down_bns = {name: [] for name in self.source_names}
        self.up_convs = {name: [] for name in self.source_names}
        self.up_bns = {name: [] for name in self.source_names}
        self.dropouts = {name: [] for name in self.source_names}
        self.mask_convs = {}

        for name in self.source_names:
            for i in range(self.num_layers):
                self.down_convs[name].append(
                    tf.keras.layers.Conv2D(
                        self.num_initial_filters * (2 ** i),
                        [5, 5],
                        strides=[2, 2],
                        padding='same',
                        name=f"down_conv_{name}_{i}"
                    )
                )
                self.down_bns[name].append(
                    tf.keras.layers.BatchNormalization(name=f"down_bn_{name}_{i}")
                )

            for i in range(self.num_layers - 1):
                self.up_convs[name].append(
                    tf.keras.layers.Conv2DTranspose(
                        self.num_initial_filters * (2 ** (self.num_layers - i - 2)),
                        [5, 5],
                        strides=[2, 2],
                        padding="same",
                        name=f"up_conv_{name}_{i}"
                    )
                )
                self.up_bns[name].append(
                    tf.keras.layers.BatchNormalization(name=f"up_bn_{name}_{i}")
                )
                if i < 3:
                    self.dropouts[name].append(tf.keras.layers.Dropout(0.5, name=f"dropout_{name}_{i}"))
                else:
                    self.dropouts[name].append(None)

            self.mask_convs[name] = tf.keras.layers.Conv2DTranspose(
                1,
                [5, 5],
                strides=[2, 2],
                activation=tf.nn.sigmoid,
                padding="same",
                name=f"mask_conv_{name}"
            )

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape
        :return: Padding along each axis (total): (Input frequency, input time)
        '''
        return [shape[0], shape[1], 1], [shape[0], shape[1], 1]

    def call(self, inputs, training=False, return_spectrogram=False):
        '''
        Forward pass of spectrogram U-Net
        :param inputs: Input batch of mixtures, 3D tensor [batch_size, num_samples, 1], mono raw audio
        :param training: Training mode flag
        :param return_spectrogram: Whether to return spectrogram magnitudes or raw audio
        :return: Dictionary of source estimates
        '''
        window = functools.partial(tf.signal.hann_window, periodic=True)
        inv_window = tf.signal.inverse_stft_window_fn(self.hop, forward_window_fn=window)

        assert inputs.shape[-1] == 1
        stfts = tf.signal.stft(
            tf.squeeze(inputs, 2),
            frame_length=self.frame_len,
            frame_step=self.hop,
            fft_length=self.frame_len,
            window_fn=window
        )
        mix_mag = tf.abs(stfts)
        mix_angle = tf.math.angle(stfts)

        mix_mag_norm = tf.math.log1p(tf.expand_dims(mix_mag, 3))
        mix_mag_norm = mix_mag_norm[:, :, :-1, :]

        mags = dict()
        for name in self.source_names:
            enc_outputs = list()
            current_layer = mix_mag_norm

            for i in range(self.num_layers):
                current_layer = self.down_convs[name][i](current_layer)
                current_layer = self.down_bns[name][i](current_layer, training=training)
                current_layer = LeakyReLU(current_layer)
                if i < self.num_layers - 1:
                    enc_outputs.append(current_layer)

            for i in range(self.num_layers - 1):
                current_layer = self.up_convs[name][i](current_layer)
                current_layer = self.up_bns[name][i](current_layer, training=training)
                current_layer = tf.nn.relu(current_layer)
                current_layer = tf.concat([enc_outputs[-i-1], current_layer], axis=3)
                if self.dropouts[name][i] is not None:
                    current_layer = self.dropouts[name][i](current_layer, training=training)

            mask = self.mask_convs[name](current_layer)
            mask = tf.pad(mask, [(0,0), (0,0), (0, 1), (0,0)], mode="CONSTANT", constant_values=0.5)
            mask = tf.squeeze(mask, 3)

            mags[name] = tf.multiply(mix_mag, mask)

        if return_spectrogram:
            return mags
        else:
            audio_out = dict()
            for source_name in list(mags.keys()):
                stft = tf.multiply(tf.complex(mags[source_name], 0.0), tf.exp(tf.complex(0.0, mix_angle)))
                audio = tf.signal.inverse_stft(stft, self.frame_len, self.hop, self.frame_len, window_fn=inv_window)
                audio = tf.expand_dims(audio, 2)
                audio_out[source_name] = audio
            return audio_out

    def get_output(self, input, training=False, return_spectrogram=False, reuse=True):
        return self(input, training=training, return_spectrogram=return_spectrogram)

