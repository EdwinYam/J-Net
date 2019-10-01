import tensorflow as tf

from Utils import LeakyReLU
import functools
from tensorflow.contrib.signal import hann_window

'''Alias settings'''
tf.variable_scope = tf.compat.v1.variable_scope
tf.image.resize_bilinear = tf.compat.v1.image.resize_bilinear

class NestedUnetSpectrogramSeparator:
    '''
    U-Net separator network for singing voice separation.
    Takes in the mixture magnitude spectrogram and return estimates of the 
    accompaniment and voice magnitude spectrograms, or return the clean audio 
    for speech enhancement
    Uses "same" convolutions like in original paper
    '''

    def __init__(self, model_config):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network
        '''
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.mono = model_config["mono_downmix"]
        self.source_names = model_config["source_names"]
        self.num_sources = len(self.source_names)
        self.deep_supervised = model_config["deep_supervised"]

        assert(len(self.source_names) == 2) 
        # Only use for acc/voice separation for now, since model gets too big 
        # otherwise
        assert(self.mono) # Only mono

        # Spectrogram settings
        self.frame_len = 1024
        self.hop = 768

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input 
        and output, so that the Unet works and has the given shape as output 
        shape
        :param shape: Desired output shape
        :return: Padding along each axis (total): (Input frequency, input time)
        '''

        return [shape[0], shape[1], 1], [shape[0], shape[1], 1]

    def get_output(self, input, training, return_spectrogram=False, reuse=True):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch of mixtures, 3D tensor 
                      [batch_size, num_samples, 1], mono raw audio
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :Param return_spectrogram: Whether to output the spectrogram estimate or
                                   convert it to raw audio and return that
        :return: U-Net output: If return_spectrogram: Accompaniment and voice 
        magnitudes as length-two list with two 4D tensors. Otherwise Two 3D 
        tensors containing the raw audio estimates
        '''
        # Setup STFT computation
        window = functools.partial(hann_window, periodic=True)
        inv_window = tf.contrib.signal.inverse_stft_window_fn(self.hop, forward_window_fn=window)
        with tf.variable_scope("separator", reuse=reuse):
            # Compute spectrogram
            assert(input.get_shape().as_list()[2] == 1) 
            # Model works ONLY on mono
            stfts = tf.contrib.signal.stft(tf.squeeze(input, 2), 
                                           frame_length=self.frame_len, 
                                           frame_step=self.hop, 
                                           fft_length=self.frame_len, 
                                           window_fn=window)
            mix_mag = tf.abs(stfts)
            mix_angle = tf.angle(stfts)

            # Input for network
            mix_mag_norm = tf.log1p(tf.expand_dims(mix_mag, 3))
            mix_mag_norm = mix_mag_norm[:,:,:-1,:] 
            # Cut off last frequency bin to make number of frequency bins 
            # divisible by 2

            mags = dict()
            for name in self.source_names: 
                # One U-Net for each source as per Jansson et al
                # For speech enhancement, we only need one loop to output the
                # very only clean source
                enc_outputs = list()
                current_layer = mix_mag_norm

                # Down-convolution: Repeat pool-conv
                for i in range(self.num_layers):
                    assert(current_layer.get_shape().as_list()[1] % 2 == 0 and current_layer.get_shape().as_list()[2] % 2 == 0)

                    current_layer = tf.layers.conv2d(current_layer, 
                                                     self.num_initial_filters*(2**i), 
                                                     [5, 5], 
                                                     strides=[2,2], 
                                                     activation=None, 
                                                     padding='same',
                                                     name='downconv_{}'.format(i))
                    current_layer = tf.contrib.layers.batch_norm(current_layer, 
                                                                 activation_fn=LeakyReLU, 
                                                                 is_training=training)
                    enc_outputs.append(current_layer)

                all_enc_outputs = list()
                final_outputs = list()
                all_enc_outputs.append(enc_outputs)
                # Upconvolution
                for i in range(self.num_layers):
                    sub_enc_outputs = list()
                    for j in range(self.num_layers - i):
                        # Repeat: Up-convolution (transposed conv with stride), 
                        # copy-and-crop feature map from down-ward path, 
                        # convolution to combine both feature maps
                        currnet_layer_up = all_enc_outputs[i][j]
                        filter_num = self.num_initial_filters*(2**(j))
                        current_layer_up = tf.layers.conv2d_transpose(current_layer_up,
                                                                      filter_num,
                                                                      [5,5],
                                                                      strides=[2,2],
                                                                      activation=None,
                                                                      padding='same')
                        current_layer_up = tf.contrib.layers.batch_norm(current_layer_up,
                                                                        is_training=training,
                                                                        activation_fn=tf.nn.relu)
                        sub_enc_outputs.append(current_layer_up)
                        concat_output = [ all_enc_outputs[k][j] for k in range(i+1) ]
                        concat_output.append(current_layer_up)
                        current_layer_up = tf.concat(concat_output, axis=3)
                        if j < 3:
                            current_layer_up = tf.layers.dropout(current_layer_up, training=training)
                        if j == 0:
                            final_outputs.append(current_layer_up)
                    all_enc_outputs.append(sub_enc_outputs)

                # Compute mask
                source_mags = list()
                for i in range(self.num_layers):
                    if i == self.num_layers - 1 or self.deep_supervised:
                        mask = tf.layers.conv2d_transpose(final_outpus[i],
                                                           1, [5,5],
                                                           strides=[2,2],
                                                           activation=tf.nn.sigmoid,
                                                           padding='same') 
                        mask = tf.pad(mask,
                                      [(0,0), (0,0), (0,1), (0,0)],
                                      mode='CONSTANT',
                                      constant_values=0.5) 
                
                        # Pad last frequency bin of mask that is missing since 
                        # we removed it in the input
                        mask = tf.squeeze(mask, 3)

                       # Compute source magnitudes
                        source_mags.append(tf.multiply(mix_mag, mask))

                if len(source_mags) == 1:
                    source_mags = source_mags[0]
                mags[name] = source_mags

            if return_spectrogram:
                # For speech enhancement, we return only clean speech source,
                # and it's a list of output if self.deep_supervised == True
                return mags
            else:
                audio_out = dict()
                # Reconstruct audio
                for source_name in list(mags.keys()):
                    if self.deep_supervised:
                        stft = multiply(tf.complex(mags[source_name][-1], 0.0),
                                        tf.exp(tf.complex(0.0, mix_angle)))
                        audio = tf.contrib.signal.inverse_stft(stft,
                                                               self.frame_len,
                                                               self.hop,
                                                               self.frame_len,
                                                               window_fn=inv_window)
                        audio = tf.expand_dims(audio, 2)

                    else:
                        audios = lists()
                        for i in range(self.num_layers):
                            stft = tf.multiply(tf.complex(mags[source_name][i], 0.0), 
                                               tf.exp(tf.complex(0.0, mix_angle)))
                            audio = tf.contrib.signal.inverse_stft(stft, 
                                                                   self.frame_len, 
                                                                   self.hop, 
                                                                   self.frame_len, 
                                                                   window_fn=inv_window)
                            # Reshape to [batch_size, samples, 1]
                            audio = tf.expand_dims(audio, 2)
                            audios.append(audio)
                        audio_out[source_name] = audios

                return audio_out
