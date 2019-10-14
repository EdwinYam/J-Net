import tensorflow as tf

import Models.InterpolationLayer
import Utils
from Utils import LeakyReLU
import numpy as np
import Models.OutputLayer

'''Alias Settings'''
tf.variable_scope = tf.compat.v1.variable_scope
tf.image.resize_bilinear = tf.compat.v1.image.resize_bilinear


class NestedUnetAudioSeparator:
    '''
    U-Net separator network for audio separation or speech enhancement.
    Uses valid convolutions, so it predicts for the centre part of the input - 
    only certain input and output shapes are therefore possible (see getpadding 
    function)
    '''

    def __init__(self, model_config):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        :param context: Determine whether to discard those padding for 
                        consistent size
        :param upsampling: Determine whether upsampling is learned
        :param deep_supervised: Detemine whether to supervise over outputs of 
                                all nested Unet
        '''
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
        self.frozen_downsample = model_config["frozen_downsample"]

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
            # Check if desired shape is possible as output shape - go from 
            # output shape towards lowest-res feature map
            rem = float(shape[1]) # Cut off batch size number and channel

            # Output filter size
            rem = rem + self.output_filter_size - 1

            # Upsampling blocks
            for i in range(self.num_layers):
                rem = rem + self.merge_filter_size - 1
                rem = (rem + 1.) / 2. # out = in + in - 1 <=> in = (out+1) / 2

            # Round resulting feature map dimensions up to nearest integer
            x = np.asarray(np.ceil(rem),dtype=np.int64)
            assert(x >= 2)

            # Compute input and output shapes based on lowest-res feature map
            output_shape = x
            input_shape = x

            # Extra conv
            input_shape = input_shape + self.filter_size - 1

            # Go from centre feature map through up- and downsampling blocks
            for i in range(self.num_layers):
                output_shape = 2*output_shape - 1 # Upsampling
                output_shape = output_shape - self.merge_filter_size + 1 # Conv

                input_shape = 2*input_shape - 1 # Decimation
                if i < self.num_layers - 1:
                    input_shape = input_shape + self.filter_size - 1 # Conv
                else:
                    input_shape = input_shape + self.input_filter_size - 1

            # Output filters
            output_shape = output_shape - self.output_filter_size + 1

            input_shape = np.asarray([shape[0], input_shape, self.num_channels])
            output_shape = np.asarray([shape[0], output_shape, self.num_channels])
            # input_shape = np.concatenate([[shape[0]], [input_shape], [self.num_channels]])
            # output_shape = np.concatenate([[shape[0]], [output_shape], [self.num_channels]])
            return input_shape, output_shape
        else:
            input_shape = np.asarray([shape[0], shape[1], self.num_channels])
            output_shape = input_shape
            return input_shape, output_shape

    def get_output(self, input, training, return_spectrogram=False, reuse=True, use_discriminator=False):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch of mixtures, 3D tensor 
                      [batch_size, num_samples, num_channels]
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :return: U-Net output: List of source estimates. Each item is a 3D tensor 
                               [batch_size, num_out_samples, num_channels]
        '''
        with tf.variable_scope("separator", reuse=reuse):
            enc_outputs = list()
            current_layer = input

            # Down-convolution: Repeat strided conv
            for i in range(self.num_layers):
                filter_num = self.num_initial_filters + self.num_increase_filters * i 
                current_layer = tf.layers.conv1d(current_layer, 
                                                 filter_num, 
                                                 self.filter_size, 
                                                 strides=1, 
                                                 activation=LeakyReLU, 
                                                 padding=self.padding,
                                                 name='downconv_{}'.format(i),
                                                 trainable=self.frozen_downsample) # out = in - filter + 1

                enc_outputs.append(current_layer)
                current_layer = current_layer[:,::2,:] 
                # Decimate by factor of 2 
                # out = (in-1)/2 + 1

            filter_num = self.num_initial_filters + self.num_increase_filters * (self.num_layers-1)
            current_layer = tf.layers.conv1d(current_layer, 
                                             filter_num,
                                             self.filter_size,
                                             activation=LeakyReLU,
                                             padding=self.padding,
                                             name='downconv_{}'.format(self.num_layers),
                                             trainable=self.frozen_downsample) 
            # One more conv here since we need to compute features after last 
            # decimation
            enc_outputs.append(current_layer)

            ####
            if use_discriminator:
                return enc_outputs
            
            assert(len(enc_outputs) == self.num_layers + 1)
            current_layer_up = current_layer

            # Feature map here shall be X along one dimension
            all_enc_outputs = list()
            all_enc_outputs.append(enc_outputs)
            # Upconvolution
            for i in range(self.num_layers):
                # Store the output of sub upsampling network, and there are
                # (self.num_layers) sub networks in WaveUnet++ 
                sub_enc_outputs = list()
                for j in range(1, self.num_layers + 1 - i):
                    # The first stored encoded output is not used but passed to
                    # final conv layer for final output
                    current_layer_up = all_enc_outputs[i][j]
                    if j != self.num_layers:
                        # Deepest encoded output needs upsampling only, no need
                        # to be passed to any conv layer
                        # Other encoded output needs to be crop_and_concat"ed"
                        # and be passed to conv layer
                        for k in range(i):
                            assert(enc_outputs[j].get_shape().as_list()[1] == all_enc_outputs[k+1][j].get_shape().as_list()[1] or self.context)
                            #No cropping should be necessary unless we are using context
                            current_layer_up = Utils.crop_and_concat(all_enc_outputs[k+1][j],
                                                                     current_layer_up,
                                                                     match_feature_dim=False)
                        
                        filter_num = self.num_initial_filters + self.num_increase_filters * (j-1)
                        current_layer_up = tf.layers.conv1d(current_layer_up, 
                                                            filter_num,
                                                            self.merge_filter_size,
                                                            activation=LeakyReLU,
                                                            padding=self.padding,
                                                            name='upconv_{}_{}'.format(i,j))
                    
                    #UPSAMPLING
                    current_layer_up = tf.expand_dims(current_layer_up, axis=1)
                    if self.upsampling == 'learned':
                        # Learned interpolation between two neighbouring time positions 
                        # by using a convolution filter of width 2, and inserting the 
                        # responses in the middle of the two respective inputs
                        current_layer_up = Models.InterpolationLayer.learned_interpolation_layer(current_layer_up, self.padding, '{}_{}'.format(i,j))
                    else:
                        if self.context:
                            current_shape = [1, current_layer_up.get_shape().as_list()[2]*2 - 1]
                            current_layer_up = tf.image.resize_bilinear(current_layer_up, 
                                                                        current_shape,
                                                                        align_corners=True)
                        else:
                            current_shape = [1, current_layer_up.get_shape().as_list()[2]*2]
                            current_layer_up = tf.image.resize_bilinear(current_layer_up, 
                                                                        current_shape) # out = in + in - 1
 
                    current_layer_up = tf.squeeze(current_layer_up, axis=1)
                    sub_enc_outputs.append(current_layer_up)
                    # UPSAMPLING FINISHED

                all_enc_outputs.append(sub_enc_outputs)

            for m in range(len(all_enc_outputs)):
                for n in range(len(all_enc_outputs[m])):
                    print('   [unet++] {}_{}: {}'.format(m, n, all_enc_outputs[m][n].get_shape().as_list()))

            # Reconnect/concatenate the most swallow layer together to form the 
            # input of last conv
            final_outputs = list()
            for i in range(1,self.num_layers+1):
                if (not self.deep_supervised and i != self.num_layers):
                     continue
                current_layer = all_enc_outputs[i][0]

                for j in range(i):
                    current_layer = Utils.crop_and_concat(all_enc_outputs[j][0], 
                                                          current_layer, 
                                                          match_feature_dim=False)

                current_layer = tf.layers.conv1d(current_layer,
                                                 self.num_initial_filters+self.num_increase_filters*(self.num_layers-1),
                                                 self.merge_filter_size,
                                                 activation=LeakyReLU,
                                                 padding=self.padding) 
                                                 # out = in - filter + 1
                
                # The final_output which pass less conv layers tends to be 
                # longer, thus, we need to crop those output (or not need to
                # crop?)
                # TODO
                final_outputs.append(current_layer)
            
            for i in range(self.num_layers - 1):
                if not self.deep_supervised:
                    continue
                assert(final_outputs[i].get_shape().as_list()[1] > final_outputs[-1].get_shape().as_list()[1] or not self.context)
                print('    [unet++] {}th subnet final output shape: {}'.format(i+1, final_outputs[i].get_shape().as_list()))
                final_outputs[i] = Utils.crop(final_outputs[i],
                                              final_outputs[-1].get_shape().as_list(),
                                              match_feature_dim=False)
            print('    [unet++] Final output shape: {}'.format(final_outputs[-1]))

            # Output layer
            # Determine output activation function
            # (tf.tanh) seem to be better than (linear)
            if self.output_activation == "tanh":
                out_activation = tf.tanh
            elif self.output_activation == "linear":
                out_activation = lambda x: Utils.AudioClip(x, training)
            else:
                raise NotImplementedError

            if self.output_type == "direct":
                if self.deep_supervised:
                    return [Models.OutputLayer.independent_outputs(final_outputs[i],
                                                                   self.source_names,
                                                                   self.num_channels, 
                                                                   self.output_filter_size,
                                                                   self.padding, 
                                                                   out_activation) for i in range(self.num_layers)]
                else:
                    return Models.OutputLayer.independent_outputs(final_outputs[-1],
                                                                  self.source_names,
                                                                  self.num_channels,
                                                                  self.output_filter_size,
                                                                  self.padding,
                                                                  out_activation)
            elif self.output_type == "difference":
                cropped_input = Utils.crop(input,
                                           final_outputs[-1].get_shape().as_list(), 
                                           match_feature_dim=False)
                if self.deep_supervised:
                    cropped_inputs = [cropped_input] * self.num_layers
                    return [Models.OutputLayer.difference_output(cropped_inputs[i], 
                                                                 final_outputs[i], 
                                                                 self.source_names, 
                                                                 self.num_channels, 
                                                                 self.output_filter_size, 
                                                                 self.padding, 
                                                                 out_activation, 
                                                                 training) for i in range(self.num_layers)]
                else:
                    return Models.OutputLayer.difference_output(cropped_input,
                                                                final_outputs[-1],
                                                                self.source_names,
                                                                self.num_channels,
                                                                self.output_filter_size,
                                                                self.padding,
                                                                out_activation,
                                                                training)
            else:
                raise NotImplementedError
