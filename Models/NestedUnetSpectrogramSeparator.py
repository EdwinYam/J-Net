import tensorflow as tf
from Utils import LeakyReLU
import functools

class NestedUnetSpectrogramSeparator(tf.keras.Model):
    '''
    U-Net separator network for singing voice separation.
    Takes in the mixture magnitude spectrogram and return estimates of the 
    accompaniment and voice magnitude spectrograms, or return the clean audio 
    for speech enhancement
    Uses "same" convolutions like in original paper
    '''

    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.mono = model_config["mono_downmix"]
        self.source_names = model_config["source_names"]
        self.num_sources = len(self.source_names)
        self.deep_supervised = model_config["deep_supervised"]

        assert len(self.source_names) == 2
        assert self.mono

        # Spectrogram settings
        self.frame_len = 1024
        self.hop = 768

        # Build sub-layers per source
        self.down_convs = {name: [] for name in self.source_names}
        self.down_bns = {name: [] for name in self.source_names}
        self.up_conv_dict = {name: {} for name in self.source_names}
        self.up_bn_dict = {name: {} for name in self.source_names}
        self.dropouts = {name: {} for name in self.source_names}
        self.mask_convs = {name: {} for name in self.source_names}

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

            for i in range(self.num_layers):
                for j in range(self.num_layers - i):
                    key = f"{i}_{j}"
                    filter_num = self.num_initial_filters * (2 ** j)
                    self.up_conv_dict[name][key] = tf.keras.layers.Conv2DTranspose(
                        filter_num,
                        [5, 5],
                        strides=[2, 2],
                        padding='same',
                        name=f"up_conv_{name}_{key}"
                    )
                    self.up_bn_dict[name][key] = tf.keras.layers.BatchNormalization(name=f"up_bn_{name}_{key}")
                    if j < 3:
                        self.dropouts[name][key] = tf.keras.layers.Dropout(0.5, name=f"dropout_{name}_{key}")
                    else:
                        self.dropouts[name][key] = None

            for i in range(self.num_layers):
                if i == self.num_layers - 1 or self.deep_supervised:
                    self.mask_convs[name][str(i)] = tf.keras.layers.Conv2DTranspose(
                        1,
                        [5, 5],
                        strides=[2, 2],
                        activation=tf.nn.sigmoid,
                        padding='same',
                        name=f"mask_conv_{name}_{i}"
                    )

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input 
        and output, so that the Unet works and has the given shape as output 
        shape
        :param shape: Desired output shape
        :return: Padding along each axis (total): (Input frequency, input time)
        '''
        return [shape[0], shape[1], 1], [shape[0], shape[1], 1]

    def call(self, inputs, training=False, return_spectrogram=False):
        '''
        Forward pass of nested spectrogram U-Net
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
                enc_outputs.append(current_layer)

            all_enc_outputs = list()
            final_outputs = list()
            all_enc_outputs.append(enc_outputs)

            for i in range(self.num_layers - 1):
                sub_enc_outputs = list()
                for j in range(self.num_layers - 1 - i):
                    key = f"{i}_{j}"
                    current_layer_up = all_enc_outputs[i][j + 1]
                    current_layer_up = self.up_conv_dict[name][key](current_layer_up)
                    current_layer_up = self.up_bn_dict[name][key](current_layer_up, training=training)
                    current_layer_up = tf.nn.relu(current_layer_up)

                    concat_output = [all_enc_outputs[k][j] for k in range(i + 1)]
                    concat_output.append(current_layer_up)
                    current_layer_up = tf.concat(concat_output, axis=3)

                    if self.dropouts[name][key] is not None:
                        current_layer_up = self.dropouts[name][key](current_layer_up, training=training)
                    sub_enc_outputs.append(current_layer_up)
                    if j == 0:
                        final_outputs.append(current_layer_up)

                all_enc_outputs.append(sub_enc_outputs)

            source_mags = list()
            for i in range(len(final_outputs)):
                if i == len(final_outputs) - 1 or self.deep_supervised:
                    mask = self.mask_convs[name][str(i)](final_outputs[i])
                    mask = tf.pad(mask, [(0, 0), (0, 0), (0, 1), (0, 0)], mode='CONSTANT', constant_values=0.5)
                    mask = tf.squeeze(mask, 3)
                    source_mags.append(tf.multiply(mix_mag, mask))

            if len(source_mags) == 1:
                source_mags = source_mags[0]
            mags[name] = source_mags

        if return_spectrogram:
            return mags
        else:
            audio_out = dict()
            for source_name in list(mags.keys()):
                if self.deep_supervised:
                    stft = tf.multiply(tf.complex(mags[source_name][-1], 0.0), tf.exp(tf.complex(0.0, mix_angle)))
                    audio = tf.signal.inverse_stft(stft, self.frame_len, self.hop, self.frame_len, window_fn=inv_window)
                    audio = tf.expand_dims(audio, 2)
                    audio_out[source_name] = audio
                else:
                    audios = list()
                    for i in range(self.num_layers):
                        stft = tf.multiply(tf.complex(mags[source_name][i], 0.0), tf.exp(tf.complex(0.0, mix_angle)))
                        audio = tf.signal.inverse_stft(stft, self.frame_len, self.hop, self.frame_len, window_fn=inv_window)
                        audio = tf.expand_dims(audio, 2)
                        audios.append(audio)
                    if len(audios) == 1:
                        audios = audios[0]
                    audio_out[source_name] = audios

            return audio_out

    def get_output(self, input, training=False, return_spectrogram=False, reuse=True):
        return self(input, training=training, return_spectrogram=return_spectrogram)
